import torch as t
import tntorch as tn
from einops import einsum

from riemannian.riemannian_ttm import RiemannianTTMCores, cores_from_deltas


class NonTensorOptimizer:
    def __init__(self, params, defaults):
        self.defaults = defaults
        self.param_groups = []

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]
        for param_group in param_groups:
            self.add_param_group(param_group)

    def add_param_group(self, param_group):
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.

        Args:
            param_group (dict): Specifies what Tensors should be optimized along with group
                specific optimization options.
        """
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']
        if isinstance(params, t.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            param_group['params'] = list(params)

        for name, default in self.defaults.items():
            param_group.setdefault(name, default)

        params = param_group['params']
        assert len(params) == len(set(params)), (
            "optimizer contains a parameter group with duplicate parameters; "
            "in future, this will cause an error; "
            "see github.com/pytorch/pytorch/issues/40967 for more information"
        )

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)


class RiemannianOptimizer(NonTensorOptimizer):
    def __init__(self, params, lr: float, normalize_grad: bool = False):
        super().__init__(params, {'lr': lr, 'normalize_grad': normalize_grad})

    def step(self, closure=None):
        assert closure is None, 'Not implemented'

        for group in self.param_groups:
            lr = group['lr']
            normalize_grad = group['normalize_grad']

            for param in group['params']:
                n_none_grads = sum(delta.grad is None for delta in param.deltas)
                if n_none_grads == len(param.deltas):
                    continue
                assert n_none_grads == 0, 'Some gradients for deltas are None, but some are not'

                for delta, u in zip(param.deltas[:-1], param.us):
                    delta.grad = delta.grad - einsum(delta.grad, u, u, 'r1 d r2, r1 d x, i j x  -> i j r2')

                if normalize_grad:
                    grad_cores = cores_from_deltas(param.us, param.vs, [delta.grad for delta in param.deltas])
                    tt_grad = tn.Tensor(grad_cores)
                    lr /= t.sqrt(tt_grad.normsq())

                for delta in param.deltas:
                    delta.data -= lr * delta.grad
                param.reparameterize()

    @classmethod
    def from_module(cls, module: t.nn.Module, lr: float, normalize_grad: bool = False):
        submodules = [submodule for _, submodule in module.named_modules() if isinstance(submodule, RiemannianTTMCores)]
        return cls(submodules, lr, normalize_grad)
