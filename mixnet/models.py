import torch
import torch.nn as nn
from torch.autograd import Function
import torch.optim as optim
import timeit
import mixnet._cpp
import numpy as np
if torch.cuda.is_available(): import mixnet._cuda
import numpy.random as npr
#import setproctitle
import random

def get_k(n):
    return int((2 * n) ** 0.5 + 1)


class MixingFunc(Function):
    '''Apply the Mixing method to the input probabilities.

    Args: see MIXNet.

    Impl Note:
        The MIXNet is a wrapper for the MixingFunc,
        handling the initialization and the wrapping of auxiliary variables.
    '''

    @staticmethod
    def forward(ctx, C, z, is_input, max_iter, eps, prox_lam):
        B, n, k = z.size(0), C.size(0), 32  # 32, get_k(C.size(0))
        ctx.prox_lam = prox_lam

        device = 'cuda' if C.is_cuda else 'cpu'
        ctx.g, ctx.gnrm = torch.zeros(B, k, device=device), torch.zeros(B, n, device=device)
        ctx.index = torch.zeros(B, n, dtype=torch.int, device=device)
        ctx.is_input = torch.zeros(B, n, dtype=torch.int, device=device)
        ctx.V = torch.zeros(B, n, k, device=device).normal_()
        # ctx.V = ctx.V / torch.norm(ctx.V, p=2, dim=2, keepdim=True)

        ctx.z = torch.zeros(B, n, device=device)
        ctx.niter = torch.zeros(B, dtype=torch.int, device=device)
        ctx.tmp = torch.zeros(B, n, k, device=device)

        ctx.C = torch.zeros(n, n, device=device)
        # ctx.Cnrms = torch.zeros(n, device=device)
        ctx.z[:] = z.data
        ctx.C[:] = C.data
        ctx.is_input[:] = is_input.data
        perm = torch.randperm(n - 1, dtype=torch.int, device=device)

        mixnet_impl = mixnet._cuda if C.is_cuda else mixnet._cpp
        mixnet_impl.init(perm, ctx.is_input, ctx.index, ctx.z, ctx.V, ctx.tmp)

        mixnet_impl.forward(max_iter, eps,
                            ctx.index, ctx.niter, ctx.C, ctx.z,
                            ctx.V, ctx.gnrm, ctx.g, ctx.tmp)

        return ctx.z.clone()

    @staticmethod
    def backward(ctx, dz):
        B, n, k = dz.size(0), ctx.C.size(0), 32 # 32, get_k(ctx.C.size(0))

        device = 'cuda' if ctx.C.is_cuda else 'cpu'
        ctx.dC = torch.zeros(B, n, n, device=device)
        ctx.U = torch.zeros(B, n, k, device=device)
        ctx.dz = torch.zeros(B, n, device=device)

        ctx.dz[:] = dz.data
        mixnet_impl = mixnet._cuda if ctx.C.is_cuda else mixnet._cpp

        mixnet_impl.backward(ctx.prox_lam,
                             ctx.is_input, ctx.index, ctx.niter, ctx.C, ctx.dC, ctx.z, ctx.dz,
                             ctx.V, ctx.U, ctx.gnrm, ctx.g, ctx.tmp)

        ctx.dC = ctx.dC.sum(dim=0)

        return ctx.dC, ctx.dz, None, None, None, None


def insert_constants(x, pre, n_pre, app, n_app):
    ''' prepend and append torch tensors
    '''
    one = x.new(x.size()[0], 1).fill_(1)
    seq = []
    if n_pre != 0:
        seq.append((pre * one).expand(-1, n_pre))
    seq.append(x)
    if n_app != 0:
        seq.append((app * one).expand(-1, n_app))
    r = torch.cat(seq, dim=1)
    r.requires_grad = False
    return r


class MixNet(nn.Module):
    '''Apply a MixNet layer to complete the input probabilities.

    Args:
        n: Number of input variables.
        aux: Number of auxiliary variables.

        max_iter: Maximum number of iterations for solving
            the inner optimization problem.
            Default: 40
        eps: The stopping threshold for the inner optimizaiton problem.
            The inner Mixing method will stop when the function decrease
            is less then eps times the initial function decrease.
            Default: 1e-4
        prox_lam: The diagonal increment in the backward linear system
            to make the backward pass more stable.
            Default: 1e-2
        weight_normalize: Set true to perform normlization for init weights.
            Default: True

    Inputs: (z, is_input)
        **z** of shape `(batch, n)`:
            Float tensor containing the probabilities (must be in [0,1]).
        **is_input** of shape `(batch, n)`:
            Int tensor indicating which **z** is a input.

    Outputs: z
        **z** of shape `(batch, n)`:
            The prediction probabiolities.

    Attributes: C
        **S** of shape `(n, n)`:
            The learnable equality matrix containing `n` variables.

    Examples:
        >>> mix = mixnet.MixNet(3, aux=5)
        >>> z = torch.randn(2, 3)
        >>> is_input = torch.IntTensor([[1, 1, 0], [1,0,1]])
        >>> pred = mix(z, is_input)
    '''

    def __init__(self, n, aux=0, max_iter=40, eps=1e-4, prox_lam=1e-2, weight_normalize=True):
        super(MixNet, self).__init__()
        self.nvars = n + 1 + aux
        C_t = torch.randn(self.nvars, self.nvars)
        C_t = C_t + C_t.t() - 1
        C_t.fill_diagonal_(0)

        if weight_normalize: C_t = C_t * ((.5 / (self.nvars * 2)) ** 0.5) # extremely important!
        self.C = nn.Parameter(C_t)
        self.aux = aux
        self.max_iter, self.eps, self.prox_lam = max_iter, eps, prox_lam

    def forward(self, z, is_input):
        device = 'cuda' if self.C.is_cuda else 'cpu'
        is_input = insert_constants(is_input.data, 1, 1, 0, self.aux)
        z = torch.cat([torch.ones(z.size(0), 1, device=device), z, torch.zeros(z.size(0), self.aux, device=device)],
                      dim=1)

        z = MixingFunc.apply(self.C, z, is_input, self.max_iter, self.eps, self.prox_lam)

        return z[:, 1:self.C.size(0) - self.aux]