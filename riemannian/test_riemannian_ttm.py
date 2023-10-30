from dataclasses import dataclass

import torch as t
import numpy as np
from einops import einsum
from hypothesis import settings, given, note
from hypothesis import strategies as st
from tntorch import TTMatrix

from riemannian.optimizer import RiemannianOptimizer
from riemannian.riemannian_ttm import RiemannianTTMCores
from riemannian.riemannian_ttm_linear import RiemannianTTMLinear


@dataclass
class TTMParams:
    input_dims: list[int]
    output_dims: list[int]
    ranks: list[int]

    def randn(self):
        tensor = t.randn(np.prod(self.input_dims), np.prod(self.output_dims))
        return TTMatrix(tensor, self.ranks, self.input_dims, self.output_dims)

    def riemannian_randn(self, rank: int = None):
        if rank is None:
            rank = np.max(self.ranks)
        return RiemannianTTMCores(self.randn(), rank)

    def make_full_rank(self):
        dims = [i * o for i, o in zip(self.input_dims, self.output_dims)]
        self.ranks = np.minimum(np.cumprod(dims)[:-1], np.cumprod(dims[::-1])[:-1][::-1]).tolist()

    def linear_layer(self, bias: bool):
        return RiemannianTTMLinear(
            np.prod(self.input_dims),
            np.prod(self.output_dims),
            np.max(self.ranks),
            self.input_dims,
            self.output_dims,
            bias=bias,
        )


@st.composite
def ttm_strategy(draw):
    n_dims = draw(st.integers(2, 4))
    return TTMParams(
        input_dims=draw(st.lists(st.integers(1, 4), min_size=n_dims, max_size=n_dims)),
        output_dims=draw(st.lists(st.integers(1, 4), min_size=n_dims, max_size=n_dims)),
        ranks=draw(st.lists(st.integers(1, 4), min_size=n_dims - 1, max_size=n_dims - 1)),
    )



@settings(deadline=None)
@given(ttm_params=ttm_strategy())
def test_initialization_is_the_same(ttm_params: TTMParams):
    ttm = ttm_params.randn()
    riemannian_ttm = RiemannianTTMCores(ttm, None)
    assert t.allclose(riemannian_ttm.ttmatrix().torch(), ttm.torch(), rtol=1e-4, atol=1e-4)


@settings(deadline=None)
@given(ttm_params=ttm_strategy())
def test_initialization_orthogonality(ttm_params: TTMParams):
    ttm = ttm_params.riemannian_randn()
    for i in range(ttm.n_dims - 1):
        assert t.allclose(einsum(ttm.us[i], ttm.us[i], 'r d x, r d y -> x y'), t.eye(ttm.us[i].shape[2]), rtol=1e-4, atol=1e-4)
    for i in range(1, ttm.n_dims):
        assert t.allclose(einsum(ttm.vs[i], ttm.vs[i], 'x d r, y d r -> x y'), t.eye(ttm.vs[i].shape[0]), rtol=1e-4, atol=1e-4)


@settings(deadline=None)
@given(ttm_params=ttm_strategy())
def test_reparameterize_without_changes(ttm_params: TTMParams):
    riemannian_ttm = ttm_params.riemannian_randn()
    before = riemannian_ttm.torch()
    riemannian_ttm.reparameterize()
    after = riemannian_ttm.torch()
    assert t.allclose(before, after, rtol=1e-4, atol=1e-4)


@settings(deadline=None)
@given(ttm_params=ttm_strategy())
def test_reparameterize_multiply_minus_one(ttm_params: TTMParams):
    riemannian_ttm = ttm_params.riemannian_randn()
    before = riemannian_ttm.torch()
    riemannian_ttm.deltas[0].data *= -1
    riemannian_ttm.reparameterize()
    after = riemannian_ttm.torch()
    assert t.allclose(before, -after, rtol=1e-4, atol=1e-4)


@settings(deadline=None)
@given(ttm_params=ttm_strategy())
def test_one_zero_grad_step(ttm_params: TTMParams):
    riemannian_ttm = ttm_params.riemannian_randn()
    optimizer = RiemannianOptimizer([riemannian_ttm], 1e-1)

    before = riemannian_ttm.torch()
    for delta in riemannian_ttm.deltas:
        delta.grad = t.zeros_like(delta)
    optimizer.step()
    after = riemannian_ttm.torch()
    assert t.allclose(before, after, rtol=1e-4, atol=1e-4)


@settings(deadline=None)
@given(ttm_params=ttm_strategy())
def test_sqr_norm_loss_in_one_step(ttm_params: TTMParams):
    riemannian_ttm = ttm_params.riemannian_randn()
    optimizer = RiemannianOptimizer([riemannian_ttm], 0.5)

    (riemannian_ttm.torch()**2).sum().backward()
    optimizer.step()
    result = riemannian_ttm.torch()
    assert t.allclose(result, t.zeros_like(result), rtol=1e-4, atol=1e-4)


@settings(deadline=None)
@given(ttm_params=ttm_strategy())
def test_compare_with_SGD_for_full_rank(ttm_params: TTMParams):
    ttm_params.make_full_rank()
    riemannian_ttm = ttm_params.riemannian_randn()
    full_matrix = t.nn.Parameter(riemannian_ttm.torch())

    def complex_loss(matr: t.Tensor):
        coeffs = t.log(t.arange(matr.shape[0]) + 1)[:, None] + t.log(t.arange(matr.shape[1]) + 1)[None, :]
        return ((matr * coeffs)**2).sum()

    optimizer = RiemannianOptimizer([riemannian_ttm], 0.5)
    complex_loss(riemannian_ttm.torch()).backward()
    optimizer.step()

    optimizer = t.optim.SGD([full_matrix], 0.5)
    complex_loss(full_matrix).backward()
    optimizer.step()

    assert t.allclose(riemannian_ttm.torch(), full_matrix, rtol=1e-3, atol=1e-3)


@settings(deadline=None)
@given(ttm_params=ttm_strategy(), bias=st.booleans())
def test_compare_linear_layers(ttm_params: TTMParams, bias: bool):
    ttm_params.make_full_rank()
    ttm_linear = ttm_params.linear_layer(bias=bias)
    linear = t.nn.Linear(np.prod(ttm_params.input_dims), np.prod(ttm_params.output_dims), bias=bias)
    ttm_linear.set_from_linear(linear)

    x = t.randn(2**10, np.prod(ttm_params.input_dims))
    assert t.allclose(ttm_linear(x), linear(x), atol=1e-4, rtol=1e-4)


@settings(deadline=None)
@given(
    dims=st.lists(st.integers(1, 4), min_size=2, max_size=4),
    rank=st.integers(2, 3)
)
def test_linear_with_identity_matrix(dims: list[int], rank: int):
    ttm_linear = RiemannianTTMLinear(np.prod(dims), np.prod(dims), rank, dims, dims)
    linear = t.nn.Linear(np.prod(dims), np.prod(dims))
    linear.weight.data = t.eye(np.prod(dims))
    ttm_linear.set_from_linear(linear)

    x = t.randn(2**10, np.prod(dims))
    assert t.allclose(ttm_linear(x), linear(x), atol=1e-4, rtol=1e-4)


@settings(deadline=None)
@given(ttm_params=ttm_strategy(), normalize_grad=st.booleans())
def test_compare_with_SGD(ttm_params: TTMParams, normalize_grad: bool):
    riemannian_linear = ttm_params.linear_layer(bias=False)
    linear = t.nn.Linear(np.prod(ttm_params.input_dims), np.prod(ttm_params.output_dims), bias=False)
    linear.weight.data = riemannian_linear.cores.torch().T

    def simple_loss(layer):
        matrix = layer(t.eye(np.prod(ttm_params.input_dims)))
        return (matrix**2).sum()

    def assert_compare():
        assert t.allclose(simple_loss(riemannian_linear), simple_loss(linear), atol=1e-4, rtol=1e-4)
        assert t.allclose(riemannian_linear.cores.torch(), linear.weight.T, atol=1e-4, rtol=1e-4)

    assert_compare()

    optimizer = RiemannianOptimizer.from_module(riemannian_linear, lr=1e-2, normalize_grad=normalize_grad)
    simple_loss(riemannian_linear).backward()
    optimizer.step()

    optimizer = t.optim.SGD(linear.parameters(), 1e-2)
    simple_loss(linear).backward()
    if normalize_grad:
        linear.weight.grad /= t.sqrt((linear.weight.grad**2).sum())
    optimizer.step()

    assert_compare()
