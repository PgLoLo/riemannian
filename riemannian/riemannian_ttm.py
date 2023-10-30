import tntorch
import torch as t
from tntorch import TTMatrix
from torch import nn


class RiemannianTTMCores(nn.Module):
    def __init__(self, ttm: TTMatrix, rank: int):
        assert len(ttm.cores) > 1, 'One-core TTM is not supported'

        super().__init__()

        self.rank = rank
        self.dims = [core.shape[1:3] for core in ttm.cores]
        self.us = BufferList([t.ones(1)] * len(ttm.cores))
        self.vs = BufferList([t.ones(1)] * len(ttm.cores))

        self.from_tt(ttm.flatten())

    @property
    def n_dims(self) -> int:
        return len(self.deltas)

    @property
    def input_dims(self) -> list[int]:
        return [dim[0] for dim in self.dims]

    @property
    def output_dims(self) -> list[int]:
        return [dim[1] for dim in self.dims]

    def get_cores(self) -> list[t.Tensor]:
        cores = cores_from_deltas(self.us.to_list(), self.vs.to_list(), self.deltas)
        return [
            core.reshape(core.shape[0], inp_dim, out_dim, core.shape[-1])
            for core, inp_dim, out_dim in zip(cores, self.input_dims, self.output_dims)
        ]

    def reparameterize(self):
        tt = self.ttmatrix().flatten()
        tt.round_tt(rmax=self.rank)
        self.from_tt(tt)

    def from_tt(self, tt: tntorch.Tensor):
        for i in range(tt.dim() - 1):
            tt.left_orthogonalize(i)
        for i, core in enumerate(tt.cores):
            self.us[i] = core
        for i in range(tt.dim() - 1, 0, -1):
            tt.right_orthogonalize(i)
        for i, core in enumerate(tt.cores):
            self.vs[i] = core
        self.deltas = nn.ParameterList(
                [nn.Parameter(tt.cores[0])]
                + [
                    nn.Parameter(t.zeros(self.us[i - 1].shape[-1], self.us[i].shape[1], self.vs[i + 1].shape[0]))
                    for i in range(1, tt.dim() - 1)
                ]
                + [nn.Parameter(t.zeros(self.us[-2].shape[2], *self.us[-1].shape[1:]))]
        )

    def ttmatrix(self) -> TTMatrix:
        cores = self.get_cores()
        ranks = [core.shape[2] for core in cores[:-1]]
        return TTMatrix(cores, ranks, self.input_dims, self.output_dims)

    def torch(self) -> t.Tensor:
        return self.ttmatrix().torch()


class BufferList(nn.Module):
    def __init__(self, tensors):
        super().__init__()
        self.len = len(tensors)
        for i, tensor in enumerate(tensors):
            self.register_buffer(self.name(i), tensor)

    def __getitem__(self, i: int) -> t.Tensor:
        if i < 0:
            i = len(self) + i
        return self.get_buffer(self.name(i))

    def __setitem__(self, i: int, tensor: t.Tensor):
        if i < 0:
            i = len(self) + i
        self.__setattr__(self.name(i), tensor)

    def to_list(self):
        return [self[i] for i in range(len(self))]

    def __len__(self):
        return self.len

    def name(self, i: int):
        return f'buffer_{i}'


def cores_from_deltas(us, vs, deltas):
    first_core = t.concatenate([us[0], deltas[0]], dim=2)
    inner_cores = [
        t.concatenate([
            t.concatenate([us[i], deltas[i]], dim=2),
            t.concatenate([t.zeros(*vs[i].shape[:2], deltas[i].shape[2]), vs[i]], dim=2)
        ], dim=0)
        for i in range(1, len(deltas) - 1)
    ]
    last_core = t.concatenate([deltas[-1], vs[-1]], dim=0)

    cores = [first_core] + inner_cores + [last_core]

    return cores