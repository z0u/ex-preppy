import tempfile
import pytest
import torch
from torch import Tensor

from ex_color.intervention.linear_mapper import LinearMapper
from ex_color.intervention.bounded_falloff import BoundedFalloff
from ex_color.intervention.bezier_mapper import BezierMapper, FastBezierMapper
from ex_color.intervention.suppression import Suppression
from ex_color.intervention.repulsion import Repulsion


# ---- helpers ----


@pytest.fixture(autouse=True)
def set_torch_seed():
    # Prevent nondeterministic test failures
    torch.manual_seed(123)


def unit(v: Tensor) -> Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)


# ---- mapper behavior ----


def test_linear_mapper_basic():
    m = LinearMapper(a=0.2, b=0.6)
    x = torch.tensor([0.0, 0.1, 0.2, 0.6, 0.9])
    y = m(x)
    # unchanged below a
    assert torch.allclose(y[:2], x[:2])
    # decreased (or equal) at and above a
    assert (y[2:] <= x[2:] + 1e-6).all()
    # endpoints map to themselves
    assert pytest.approx(float(m(torch.tensor([0.2])))) == 0.2
    assert pytest.approx(float(m(torch.tensor([1.0])))) == 0.6


def test_bounded_falloff_basic():
    m = BoundedFalloff(a=0.3, b=0.8, power=2)
    x = torch.tensor([0.0, 0.2, 0.3, 0.6, 0.9])
    y = m(x)
    # below a -> 0
    assert torch.allclose(y[:3], torch.tensor([0.0, 0.0, 0.0]))
    # above a -> in (0, b]
    assert (y[3:] > 0).all() and (y[3:] <= 0.8 + 1e-6).all()


@pytest.mark.parametrize('cls', [BezierMapper, FastBezierMapper])
def test_bezier_monotonic_and_bounds(cls):
    m = cls(a=0.2, b=0.7)
    x = torch.linspace(0, 1, 101)
    y = m(x)
    # unchanged below a
    assert torch.allclose(y[x <= 0.2], x[x <= 0.2])
    # within [0,1]
    assert (y >= 0 - 1e-6).all() and (y <= 1 + 1e-6).all()
    # non-decreasing
    assert (y[1:] >= y[:-1] - 1e-5).all()


# ---- mapper serialization ----


@pytest.mark.parametrize(
    'm',
    [
        LinearMapper(0.1, 0.9),
        BoundedFalloff(0.2, 0.7, power=1.5),
        BezierMapper(0.2, 0.6),
        FastBezierMapper(0.2, 0.6),
    ],
)
def test_mapper_full_serialization_roundtrip(m):
    """Test full PyTorch serialization (torch.save/load) for all mapper types."""
    with tempfile.NamedTemporaryFile() as f:
        # Save complete module
        torch.save(m, f.name)
        # Load and verify behavior matches
        m2 = torch.load(f.name, weights_only=False)

    x = torch.linspace(0, 1, 11)
    assert torch.allclose(m(x), m2(x), atol=1e-5)


def test_mapper_state_dict_serialization():
    """Test state_dict serialization for mappers with complex internal state."""
    # Test with FastBezierMapper which has lookup tables as buffers
    m1 = FastBezierMapper(0.1, 0.8, lookup_resolution=500)

    # Save and restore state
    state = m1.state_dict()
    m2 = FastBezierMapper(0.1, 0.8, lookup_resolution=500)  # Same params
    m2.load_state_dict(state)

    # Should have identical behavior
    x = torch.linspace(0, 1, 21)
    assert torch.allclose(m1(x), m2(x), atol=1e-6)

    # Internal buffers should match
    assert torch.allclose(m1.x_lookup, m2.x_lookup)
    assert torch.allclose(m1.y_lookup, m2.y_lookup)


def test_mapper_device_movement():
    """Test that buffers move correctly with .to() calls."""
    m = BezierMapper(0.2, 0.7)

    # Check initial device
    assert m.P0.device == torch.device('cpu')

    # Test that forward pass works
    x = torch.tensor([0.5, 0.8])
    y1 = m(x)

    # Move to CPU explicitly (should be no-op but tests the mechanism)
    m_cpu = m.to('cpu')
    y2 = m_cpu(x)
    assert torch.allclose(y1, y2)

    # If CUDA available, test GPU movement
    if torch.cuda.is_available():
        m_gpu = m.to('cuda')
        x_gpu = x.to('cuda')
        y_gpu = m_gpu(x_gpu)
        assert y_gpu.device.type == 'cuda'
        assert torch.allclose(y1, y_gpu.cpu(), atol=1e-6)


# ---- interventions behavior ----


def test_suppression_projects_along_concept():
    E = 4
    concept = unit(torch.randn(E))
    # Use identity-like gate: gate(x)=x for x>0 (and 0 for x<=0)
    falloff = LinearMapper(0.0, 1.0)
    sup = Suppression(concept, falloff)

    # choose vectors with positive and negative dot
    v = unit(torch.randn(8, E))
    out = sup(v)

    # For a=0, gate(x)=x (identity on [0,1]), so remove dot^2 along the concept for positive dots only
    dots = v @ concept
    proj = (dots.clamp_min(0) * dots)[:, None] * concept
    expected = v - proj
    assert torch.allclose(out, expected, atol=1e-5)


def test_repulsion_rotation_and_noop_for_negative():
    E = 5
    concept = unit(torch.randn(E))
    mapper = LinearMapper(0.2, 0.8)
    rep = Repulsion(concept, mapper)

    # build batch: first half negative dot, second half positive
    pos = unit(torch.randn(4, E))
    pos = unit(pos + concept)  # bias to positive dot
    neg = unit(torch.randn(4, E))
    neg = unit(neg - concept)  # bias to negative dot
    x = torch.cat([neg, pos], dim=0)

    y = rep(x)
    dots_in = x @ concept
    dots_out = y @ concept

    # negative dot inputs unchanged
    assert torch.allclose(y[:4], x[:4], atol=1e-5)

    # positive dot outputs moved toward mapper(dots)
    target = mapper(dots_in[4:])
    assert torch.allclose(dots_out[4:], target.clamp(-1 + 1e-6, 1 - 1e-6), atol=1e-3)
    # unit norm preserved
    assert torch.allclose(y.norm(dim=1), torch.ones(y.size(0)), atol=1e-4)


# ---- interventions serialization ----


def test_repulsion_full_serialization_with_mapper():
    """Test that Repulsion correctly serializes its mapper submodule."""
    concept = unit(torch.randn(6))
    mapper = BezierMapper(0.1, 0.5)
    rep = Repulsion(concept, mapper)

    with tempfile.NamedTemporaryFile() as f:
        torch.save(rep, f.name)
        rep2 = torch.load(f.name, weights_only=False)

    # Verify behavior matches exactly
    x = unit(torch.randn(10, 6))
    assert torch.allclose(rep(x), rep2(x), atol=1e-5)

    # Verify mapper was preserved
    assert isinstance(rep2.mapper, BezierMapper)
    assert rep2.mapper.a == rep.mapper.a
    assert rep2.mapper.b == rep.mapper.b


def test_suppression_full_serialization_with_falloff():
    """Test that Suppression correctly serializes its falloff submodule."""
    concept = unit(torch.randn(6))
    falloff = BoundedFalloff(0.2, 0.7, power=1.3)
    sup = Suppression(concept, falloff)

    with tempfile.NamedTemporaryFile() as f:
        torch.save(sup, f.name)
        sup2 = torch.load(f.name, weights_only=False)

    # Verify behavior matches exactly
    x = unit(torch.randn(10, 6))
    assert torch.allclose(sup(x), sup2(x), atol=1e-5)

    # Verify falloff was preserved
    assert isinstance(sup2.falloff, BoundedFalloff)
    assert sup2.falloff.a == sup.falloff.a
    assert sup2.falloff.b == sup.falloff.b
    assert sup2.falloff.power == sup.falloff.power


def test_intervention_state_dict_includes_submodules():
    """Test that state_dict automatically includes mapper submodules."""
    concept = unit(torch.randn(4))
    mapper = FastBezierMapper(0.1, 0.5, lookup_resolution=100)
    rep = Repulsion(concept, mapper)

    state = rep.state_dict()

    # State dict should include concept_vector buffer
    assert 'concept_vector' in state

    # State dict should include mapper submodule state with proper prefixes
    mapper_keys = [k for k in state.keys() if k.startswith('mapper.')]
    assert len(mapper_keys) > 0  # Should have mapper.P0, mapper.P1, etc.

    # Verify we can reconstruct completely
    rep2 = Repulsion(
        concept.clone(),
        FastBezierMapper(0.0, 1.0, lookup_resolution=100),  # match lookup shape
    )
    rep2.load_state_dict(state)

    # Should behave identically
    x = unit(torch.randn(5, 4))
    assert torch.allclose(rep(x), rep2(x), atol=1e-6)


def test_intervention_device_movement_with_submodules():
    """Test that interventions and their submodules move devices together."""
    concept = unit(torch.randn(3))
    mapper = BezierMapper(0.2, 0.6)
    rep = Repulsion(concept, mapper)

    # Initially on CPU
    assert rep.concept_vector.device == torch.device('cpu')
    assert rep.mapper.P0.device == torch.device('cpu')

    # Move to CPU explicitly (no-op but tests the mechanism)
    rep_cpu = rep.to('cpu')
    assert rep_cpu.concept_vector.device == torch.device('cpu')
    assert rep_cpu.mapper.P0.device == torch.device('cpu')

    # Test GPU movement if available
    if torch.cuda.is_available():
        rep_gpu = rep.to('cuda')
        assert rep_gpu.concept_vector.device.type == 'cuda'
        assert rep_gpu.mapper.P0.device.type == 'cuda'

        # Should still work correctly
        x = unit(torch.randn(3, 3).cuda())
        y = rep_gpu(x)
        assert y.device.type == 'cuda'
