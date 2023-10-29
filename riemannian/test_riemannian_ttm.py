from dataclasses import dataclass

import torch as t
import numpy as np
from einops import einsum
from hypothesis import settings, given, note
from hypothesis import strategies as st
from tntorch import TTMatrix

from riemannian.optimizer import RiemannianOptimizer
from riemannian.riemannian_ttm import RiemannianTTMCores


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


#
# @settings(deadline=None)
# @given(ttm_params=ttm_strategy())
# def test_cores_from_all_deltas(ttm_params: TTMParams):
#     ttm = ttm_params.randn()
#     riemannian_ttm = RiemannianTTMCores(ttm, None)
#     for i in range(riemannian_ttm
#     assert t.allclose(riemannian_ttm.ttmatrix().torch(), ttm.torch(), rtol=1e-4, atol=1e-4)



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
