from math import sqrt
from typing import List, Callable

import torch as t
from torch import nn
import tntorch as tn

from riemannian.forward_backward import einsum_forward
from riemannian.riemannian_ttm import RiemannianTTMCores


class RiemannianTTMLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        input_dims: List[int],
        output_dims: List[int],
        bias: bool = True,
        device=None,
        dtype=None,
        forward_fn: Callable = einsum_forward
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.forward_fn = forward_fn

        # Initialize weights from uniform[-1 / sqrt(in_features), 1 / sqrt(in_features)]
        factory_kwargs = {"device": device, "dtype": dtype}
        init = t.rand(in_features, out_features, **factory_kwargs)
        init = (2 * init - 1) / sqrt(in_features)

        self.from_matrix(init)

        if bias:
            init = t.rand(out_features, **factory_kwargs)
            init = (2 * init - 1) / sqrt(out_features)
            self.bias = nn.Parameter(init)
        else:
            self.register_parameter('bias', None)

    def from_matrix(self, matrix: t.Tensor):
        ttm = tn.TTMatrix(matrix, [self.rank] * (len(self.input_dims) - 1), self.input_dims, self.output_dims)
        self.cores = RiemannianTTMCores(ttm, self.rank)

    def forward(self, x: t.Tensor):
        res = self.forward_fn(self.cores.get_cores(), x)

        new_shape = x.shape[:-1] + (self.out_features,)
        res = res.reshape(*new_shape)

        if self.bias is not None:
            res += self.bias

        return res

    def set_weight(self, new_weights: t.Tensor):
        # in regular linear layer weights are transposed, so we transpose back
        new_weights = new_weights.clone().detach().T

        shape = t.Size((self.in_features, self.out_features))
        assert new_weights.shape == shape, f"Expected shape {shape}, got {new_weights.shape}"

        self.from_matrix(new_weights)

    def set_from_linear(self, linear: nn.Linear):
        self.set_weight(linear.weight.data)
        self.bias = nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None
