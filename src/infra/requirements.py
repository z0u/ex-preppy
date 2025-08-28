import logging
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Literal, overload, Callable
from types import ModuleType

import modal

log = logging.getLogger(__name__)


@overload
def freeze(*packages: str, dev: bool = False, local: bool = False) -> list[str]: ...


@overload
def freeze(*, group: str | list[str]) -> list[str]: ...


@overload
def freeze(*, all: bool, dev: bool = False, local: bool = False) -> list[str]: ...


def freeze(
    *packages, all: bool = False, dev: bool = False, local: bool = False, group: str | list[str] | None = None
) -> list[str]:
    """
    Get the requirements for specified packages and their dependencies.

    Args:
        *packages: Names of packages to get requirements for.
        all: If True, freeze all packages. Packages must not be specified if this is True.
        dev: If True, include development dependencies.
        local: If True, include dependencies from the 'local' group.
        group: If specified, only include packages from the given group(s).

    Returns:
        A list of package specifications in the format 'package==version'.

    """
    cmd = ['uv', '--offline', 'tree']

    if all:
        if packages:
            raise ValueError("Cannot specify packages when 'all' is True")
        package_opts = []
    elif group is not None:
        if isinstance(group, str):
            group = [group]
        if not group:
            return []
        package_opts = [f'--group={g}' for g in group]
        cmd.extend(package_opts)
    else:
        if not packages:
            return []
        package_opts = [f'--package={pkg}' for pkg in packages]
        cmd.extend([f'--package={pkg}' for pkg in packages])

    result = subprocess.run(cmd + ['--no-dedupe', '--all-groups'], text=True, capture_output=True, check=True)
    available_deps = parse_uv_tree_output(result.stdout, ignore_first=True)

    constraints = ['--no-dedupe']
    if not group:
        constraints.append('--all-groups')
    if not dev:
        constraints.append('--no-dev')
    if not local:
        constraints.append('--no-group=local')

    result = subprocess.run(cmd + constraints + package_opts, text=True, capture_output=True, check=True)
    selected_deps = parse_uv_tree_output(result.stdout, ignore_first=all or bool(group))

    log.info(f'Selected {len(selected_deps)} of {len(available_deps)} dependencies')
    log.debug('Dependencies: %s', selected_deps)
    return selected_deps


def uv_freeze(
    *packages: str,
    groups: list[str] | None = None,
    not_groups: list[str] | None = None,
    only_groups: list[str] | None = None,
    all_groups: bool = False,
    indexes: list[str] | None = None,
    python_version: str | None = None,
    python_platform: str | None = None,
    only_run_locally: bool = True,
) -> list[str]:
    if only_run_locally and not modal.is_local():
        log.info('Skipping package-freezing: not running locally')
        return []

    cmd = ['uv', '--offline', 'tree']

    result = subprocess.run(cmd + ['--no-dedupe', '--all-groups'], text=True, capture_output=True, check=True)
    all_deps = parse_uv_tree_output(result.stdout, ignore_first=True)

    opts: list[str | tuple[str, ...]] = []
    opts += [('--package', pkg) for pkg in packages]
    opts += [('--group', g) for g in (groups or [])]
    opts += [('--not-group', g) for g in (not_groups or [])]
    opts += [('--only-group', g) for g in (only_groups or [])]
    opts += [('--index', i) for i in (indexes or [])]
    if all_groups:
        opts += ['--all-groups']
    if python_version:
        opts += [('--python-version', python_version)]
    if python_platform:
        opts += [('--python-platform', python_platform)]

    # Flatten tuples
    opts = [(opt,) if isinstance(opt, str) else opt for opt in opts]
    flat_opts = [opt for sublist in opts for opt in sublist]

    result = subprocess.run(cmd + flat_opts, text=True, capture_output=True, check=True)
    selected_deps = parse_uv_tree_output(result.stdout, ignore_first=bool(groups))
    log.info(f'Selected {len(selected_deps)} of {len(all_deps)} dependencies')
    log.debug('Dependencies: %s', selected_deps)
    return selected_deps


def modnames(*modules: ModuleType | Literal['self']) -> list[str]:
    """
    Convert a list of modules into a list of names.

    'self' is special: it will be converted into the name of the calling module.
    """
    ps = set[str]()
    for mod in modules:
        if mod == 'self':
            try:
                mod = get_calling_module()
            except RuntimeError as e:
                raise ValueError('Module "self" has no name') from e
        if mod.__name__ == '__main__':
            log.warning('Using __main__ as a requirement')
        ps.add(mod.__name__)
    return sorted(ps)


def get_calling_module() -> ModuleType:
    """
    Search up the stack to find the module that called this function.

    Ignores this module.
    """
    import inspect

    frame = inspect.currentframe()
    while frame:
        mod = inspect.getmodule(frame)
        if mod and mod.__name__ != __name__:
            return mod
        frame = frame.f_back
    raise RuntimeError('Could not find calling module')


def parse_uv_tree_output(output: str, ignore_first: bool) -> list[str]:
    """Parse the output of 'uv tree' command to extract package specifications."""
    requirements: set[str] = set()

    lines = output.strip().split('\n')
    if ignore_first:
        lines = lines[1:]

    # Regular expression to extract package name and version
    # This matches lines like "package v1.2.3" with or without tree characters
    # https://packaging.python.org/en/latest/specifications/name-normalization/#name-format
    name_pattern = r'([A-Z0-9]|[A-Z0-9][A-Z0-9._-]*[A-Z0-9])'
    pattern = name_pattern + r' v([^\s]+)'

    for line in lines:
        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            pkg_name = match.group(1)
            version = match.group(2)
            requirements.add(f'{pkg_name}=={version}')

    return sorted(requirements)


def _dir_contains_python(path: Path) -> bool:
    """Return True if directory looks like a Python package (has __init__.py or any .py)."""
    if (path / '__init__.py').exists():
        return True
    return any(path.rglob('*.py'))


def _scan_src_packages(root_dir: Path) -> list[str]:
    """Discover top-level packages and modules under src/ (UV namespace discovery)."""
    src_dir = root_dir / 'src'
    if not src_dir.is_dir():
        return []
    names: set[str] = set()
    for child in src_dir.iterdir():
        n = child.name
        if n.startswith('.') or n.startswith('_'):
            continue
        if child.is_dir() and _dir_contains_python(child):
            names.add(n)
        elif child.is_file() and child.suffix == '.py' and child.stem != '__init__':
            names.add(child.stem)
    return sorted(names)


def _packages_from_uv_config(pyproject: dict, root_dir: Path) -> list[str]:
    """Load explicit packages from tool.uv.build-backend.packages if provided."""
    backend = pyproject.get('tool', {}).get('uv', {}).get('build-backend', {})
    pkgs = backend.get('packages') if isinstance(backend, dict) else None
    if not isinstance(pkgs, list):
        return []
    names: set[str] = set()
    for entry in pkgs:
        if not isinstance(entry, str):
            continue
        # try as relative to project root
        path = root_dir / entry
        if path.is_dir():
            names.add(path.name)
            continue
        if path.is_file() and path.suffix == '.py' and path.stem != '__init__':
            names.add(path.stem)
            continue
        # try relative to src/
        src_path = root_dir / 'src' / entry
        if src_path.is_dir():
            names.add(src_path.name)
        elif src_path.is_file() and src_path.suffix == '.py' and src_path.stem != '__init__':
            names.add(src_path.stem)
    return sorted(names)


def _packages_from_hatch(pyproject: dict, root_dir: Path) -> list[str]:
    """Load packages from Hatch config if present (legacy fallback)."""
    hatch_packages = (
        pyproject.get('tool', {})
        .get('hatch', {})
        .get('build', {})
        .get('targets', {})
        .get('wheel', {})
        .get('packages', [])
    )
    if not isinstance(hatch_packages, list):
        return []
    paths = [root_dir / d for d in hatch_packages if isinstance(d, str)]
    directories = [path for path in paths if path.is_dir()]
    return sorted(path.name for path in directories)


def project_packages() -> list[str]:
    """
    Determine first-party package/module names for this repo.

    Strategy (in order):
    - If using UV build backend with `namespace = true`, scan `src/` for top-level
      Python packages (directories) and modules (single .py files). This mirrors UV's
      auto-discovery behavior and avoids coupling to a specific backend.
    - Else, fall back to Hatch's `tool.hatch.build.targets.wheel.packages` if present.

    Returns:
        Sorted list of top-level import names (e.g., ['ex_color', 'infra']).
    """
    root_dir = find_project_root()

    pyproject_path = root_dir / 'pyproject.toml'
    log.debug(f'Loading {pyproject_path}')
    with open(pyproject_path, 'rb') as f:
        pyproject = tomllib.load(f)

    # 1) Prefer UV build-backend with namespace discovery: scan src/
    uv_backend = pyproject.get('tool', {}).get('uv', {}).get('build-backend', {})
    namespace_enabled = bool(uv_backend.get('namespace', False)) if isinstance(uv_backend, dict) else False

    strategies: list[tuple[str, Callable[[], list[str]]]] = []
    if namespace_enabled:
        strategies.append(('src scan', lambda: _scan_src_packages(root_dir)))
    # Try explicit UV packages if provided
    strategies.append(('uv config', lambda: _packages_from_uv_config(pyproject, root_dir)))
    # Fallback to Hatch config
    strategies.append(('hatch config', lambda: _packages_from_hatch(pyproject, root_dir)))

    for label, fn in strategies:
        packages = fn()
        if packages:
            log.info(f'Found {len(packages)} local packages via {label}: {", ".join(packages)}')
            log.debug('Packages: %s', packages)
            return packages

    log.info('No recognizable package configuration; returning empty list')
    return []


def find_project_root() -> Path:
    """
    Find the project root directory containing pyproject.toml.

    Returns:
        Path to the project root directory.

    Raises:
        FileNotFoundError: If pyproject.toml cannot be found.
    """
    current = Path.cwd()

    # Try a few times going up the directory tree
    for _ in range(10):
        if (current / 'pyproject.toml').exists():
            return current

        parent = current.parent
        if parent == current:  # Reached the file system root
            break
        current = parent

    raise FileNotFoundError(f'Could not find pyproject.toml from {current}')


def get_project_name() -> str:
    """
    Get the project name from pyproject.toml.

    Returns:
        The project name as defined in pyproject.toml.

    Raises:
        FileNotFoundError: If pyproject.toml cannot be found.
        KeyError: If 'name' is not defined in the project section.
    """
    root_dir = find_project_root()
    pyproject_path = root_dir / 'pyproject.toml'

    with open(pyproject_path, 'rb') as f:
        pyproject = tomllib.load(f)

    return pyproject['project']['name']  # type: ignore[no-any-return]
