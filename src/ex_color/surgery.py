"""
Utilities for structural model edits like pruning an ablation.

This is designed to work with bottlenecked MLPs in this repo, e.g. CNColorMLP,
where a latent "bottleneck" sits between a producing Linear and a consuming
Linear (optionally with normalization in between, e.g. L2Norm).

Two variants are provided:
 - prune: removes selected latent dimensions by shrinking surrounding Linear
   layers (deleting output rows from the producer and input columns from the
   consumer). This returns a new model with reduced latent dimensionality.
 - ablate: zeros out parameters contributing to selected latent dimensions
   (producer rows and consumer columns) without changing shapes.

Both functions operate on a deepcopy of the provided model to avoid mutation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence, TypeVar

import torch
import torch.nn as nn

M = TypeVar('M', bound=nn.Module)


@dataclass
class _Located:
    name: str
    module: nn.Linear


def _get_module_and_parent(model: nn.Module, path: str) -> tuple[nn.Module, nn.Module | None, str | None]:
    """
    Return (module, parent_module, child_name_in_parent) for a dotted path.

    If the path refers to the model itself, parent will be None and child_name None.
    """
    if not path:
        return model, None, None

    # Resolve module
    try:
        module = model.get_submodule(path)
    except AttributeError as e:
        raise AttributeError(f"No submodule at path '{path}'") from e

    # Resolve parent and child name using path split
    if '.' in path:
        parent_path, child_name = path.rsplit('.', 1)
        parent = model.get_submodule(parent_path)
    else:
        parent = model
        child_name = path
    return module, parent, child_name


def _all_linears_by_shape(model: nn.Module) -> tuple[list[_Located], list[_Located]]:
    """
    Return (producers, consumers) where

    - producers: Linear with out_features == k (to be filled later)
    - consumers: Linear with in_features == k (to be filtered later)

    We return all and filter by k in callers to avoid duplicate traversals.
    """
    prods: list[_Located] = []
    cons: list[_Located] = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            prods.append(_Located(name, mod))
            cons.append(_Located(name, mod))
    return prods, cons


def _device_dtype_like(t: torch.Tensor) -> dict:
    return {'device': t.device, 'dtype': t.dtype}


def _uniquely_select(items: list[_Located], predicate, kind: str, k: int) -> _Located:
    matches = [it for it in items if predicate(it.module)]
    if len(matches) == 0:
        raise RuntimeError(f'Could not locate a unique {kind} Linear with feature size {k}.')
    if len(matches) > 1:
        names = ', '.join(it.name or '<root>' for it in matches)
        raise RuntimeError(
            f'Ambiguous {kind} Linear candidates for k={k}: {names}. '
            f'Specify a different layer_id or refactor to make this unique.'
        )
    return matches[0]


def _find_surrounding_linears(model: nn.Module, layer_path: str) -> tuple[_Located, _Located, int]:
    """
    Locate the nearest Linear before and after the given layer in module traversal order.

    Returns (prev_linear, next_linear, k) where k == prev.out_features == next.in_features.
    """
    # Ensure the target exists
    model.get_submodule(layer_path)

    mods = list(model.named_modules())
    names = [n for n, _ in mods]
    try:
        idx = names.index(layer_path)
    except ValueError as e:
        raise AttributeError(f"Layer '{layer_path}' not found in model modules") from e

    # Search backward for nearest Linear
    prev_loc: _Located | None = None
    for i in range(idx - 1, -1, -1):
        n, m = mods[i]
        if isinstance(m, nn.Linear):
            prev_loc = _Located(n, m)
            break

    # Search forward for nearest Linear
    next_loc: _Located | None = None
    for i in range(idx + 1, len(mods)):
        n, m = mods[i]
        if isinstance(m, nn.Linear):
            next_loc = _Located(n, m)
            break

    if prev_loc is None or next_loc is None:
        return _fallback_unique_linears(model)

    prev = prev_loc.module
    next_ = next_loc.module
    if prev.out_features != next_.in_features:
        raise RuntimeError(
            'Nearest Linear layers around the target do not agree on feature size: '
            f'prev(out)={prev.out_features}, next(in)={next_.in_features}'
        )
    k = prev.out_features
    return prev_loc, next_loc, k


def _fallback_unique_linears(model: nn.Module) -> tuple[_Located, _Located, int]:
    prods, cons = _all_linears_by_shape(model)
    prod_ks = {it.module.out_features for it in prods}
    cons_ks = {it.module.in_features for it in cons}
    shared_list = sorted(prod_ks.intersection(cons_ks))
    if not shared_list:
        raise RuntimeError('No matching producer/consumer Linear layers found around the target layer.')
    for k in shared_list:
        n_prods = [it for it in prods if it.module.out_features == k]
        n_cons = [it for it in cons if it.module.in_features == k]
        if len(n_prods) == 1 and len(n_cons) == 1:
            return n_prods[0], n_cons[0], k
    k = shared_list[0]
    prev_loc = _uniquely_select(prods, lambda m: isinstance(m, nn.Linear) and m.out_features == k, 'producer', k)
    next_loc = _uniquely_select(cons, lambda m: isinstance(m, nn.Linear) and m.in_features == k, 'consumer', k)
    return prev_loc, next_loc, k


def _locate_adjacent_linears(model: nn.Module, layer_path: str) -> tuple[_Located, _Located, int]:
    return _find_surrounding_linears(model, layer_path)


def _replace_submodule(model: nn.Module, path: str, new_module: nn.Module) -> None:
    _, parent, child_name = _get_module_and_parent(model, path)
    if parent is None or child_name is None:
        raise RuntimeError('Replacing the root module is not supported by this utility.')
    # Works for both plain Modules and Sequentials
    parent._modules[child_name] = new_module


def _validate_dims(dims: Sequence[int], k: int) -> list[int]:
    uniq = sorted({int(d) for d in dims})
    if any(d < 0 or d >= k for d in uniq):
        raise IndexError(f'dims out of range for width {k}: {uniq}')
    return uniq


def _compute_keep_indices(k: int, drop: Sequence[int]) -> list[int]:
    drop_set = set(drop)
    return [i for i in range(k) if i not in drop_set]


def ablate(model: M, layer_id: str, dims: Sequence[int]) -> M:
    """
    Return a copy of model where the selected latent dims are effectively nulled.

    Implementation: zero out producer rows and consumer columns for the given dims.
    Shapes remain unchanged.
    """
    new_model: M = copy.deepcopy(model)

    prev_loc, next_loc, k = _locate_adjacent_linears(new_model, layer_id)
    drop = _validate_dims(dims, k)
    if len(drop) == 0:
        return new_model
    prev = prev_loc.module
    next_ = next_loc.module

    # Zero producer rows and corresponding biases
    with torch.no_grad():
        prev.weight.data[drop, :] = 0
        if prev.bias is not None:
            prev.bias.data[drop] = 0
        # Zero consumer columns
        next_.weight.data[:, drop] = 0

    return new_model


def prune(model: M, layer_id: str, dims: Sequence[int]) -> M:
    """
    Return a copy of model with selected latent dims fully removed.

    This reduces the latent width k by len(dims) by:
        - Removing the rows from the producer Linear's weight/bias
        - Removing the columns from the consumer Linear's weight
    """
    new_model: M = copy.deepcopy(model)

    prev_loc, next_loc, k = _locate_adjacent_linears(new_model, layer_id)
    drop = _validate_dims(dims, k)
    if len(drop) == 0:
        return new_model
    new_k = k - len(drop)
    if new_k <= 0:
        raise ValueError('Ablation would result in zero latent width; at least one dimension must remain.')

    prev = prev_loc.module
    next_ = next_loc.module

    keep = _compute_keep_indices(k, drop)

    # Create new producer with reduced out_features
    prev_like = _device_dtype_like(prev.weight)
    new_prev = nn.Linear(prev.in_features, new_k, bias=(prev.bias is not None)).to(**prev_like)
    with torch.no_grad():
        new_prev.weight.copy_(prev.weight[keep, :])
        if prev.bias is not None:
            new_prev.bias.copy_(prev.bias[keep])

    # Create new consumer with reduced in_features
    next_like = _device_dtype_like(next_.weight)
    new_next = nn.Linear(new_k, next_.out_features, bias=(next_.bias is not None)).to(**next_like)
    with torch.no_grad():
        new_next.weight.copy_(next_.weight[:, keep])
        if next_.bias is not None:
            new_next.bias.copy_(next_.bias)

    # Replace modules in-place within the copied model
    _replace_submodule(new_model, prev_loc.name, new_prev)
    _replace_submodule(new_model, next_loc.name, new_next)

    return new_model


__all__ = ['prune', 'ablate']
