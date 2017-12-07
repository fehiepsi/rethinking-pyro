# coding: utf-8

from collections import OrderedDict

import numpy as np
import scipy.stats as stats

import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
from pyro.infer import SVI
from pyro.optim import Adam


def HPDI(samples, prob=0.89):
    sorted_samples = np.sort(samples)
    mass = int(prob * len(samples))
    lower_i_bound = len(samples) - mass
    widths = sorted_samples[mass:] - sorted_samples[:lower_i_bound]
    lower_i = np.argmin(widths)
    upper_i = lower_i + mass
    return sorted_samples[lower_i], sorted_samples[upper_i]


def PI(samples, prob=0.89):
    return np.percentile(samples, [(1-prob)/2*100, (1+prob)/2*100])


def hessians(ys, xs):
    for x in xs:
        assert x.dim() == 1
    dys = torch.autograd.grad(ys, xs, create_graph=True)
    hs = []
    for i in range(len(xs)):
        for j in range(xs[i].size(0)):
            hs.append(torch.cat(torch.autograd.grad(dys[i][j], xs, retain_graph=True)))
    return torch.stack(hs)


class MAP(object):

    def __init__(self, model, latents):
        self.model = model
        self.latents = OrderedDict(latents)
        self.losses = []
        self.args = ()
        self.kwargs = {}
        self.inference = None

    def coef(self):
        return torch.cat([pyro.param(latent + ".mean") for latent in self.latents])

    def fit(self, *args, **kwargs):
        pyro.clear_param_store()
        self.losses = []
        step = kwargs.pop("step", 1000)
        optim_params = kwargs.pop("optim_params", {})
        self.args = args
        self.kwargs = kwargs
        optim = Adam(optim_params)
        self.inference = SVI(self.model, self.guide, optim, loss="ELBO")
        for _ in range(step):
            self.losses.append(self.inference.step(*self.args, **self.kwargs))
        return self

    def guide(self, *args, **kwargs):
        for latent in self.latents:
            init = self.latents[latent]
            mean = pyro.param(latent+".mean", Variable(torch.Tensor([init]), requires_grad=True))
            pyro.sample(latent, dist.delta, mean)


class Laplace(MAP):

    def precis(self, prob=0.89):
        means = self.coef()
        sds = self.sd()
        z = stats.norm.ppf([(1-prob)/2, (1+prob)/2])
        pre = dict()
        for i, latent in enumerate(self.latents):
            mean = means.data[i]
            sd = sds.data[i]
            pi = mean + sd * z
            pre[latent] = {"Mean": mean, "StdDev": sd, "pi_{}".format(prob): pi}
        return pre

    def extract_samples(self, n=1000):
        mean = self.coef().data.numpy()
        cov = self.vcov().data.numpy()
        return np.random.multivariate_normal(mean=mean, cov=cov, size=n, tol=1e-6)

    def corr(self):
        cov = self.vcov()
        D = torch.diag(1 / self.sd())
        return D.mm(cov).mm(D)

    def vcov(self):
        h = self.hessian()
        return h.inverse()

    def sd(self):
        cov = self.vcov()
        return torch.diag(cov) ** 0.5

    def hessian(self):
        xs = [pyro.param(latent+".mean") for latent in self.latents]
        self.kwargs["callback"] = lambda loss: hessians(loss, xs)
        _, hs = self.inference.loss_and_grads(self.model, self.guide, *self.args, **self.kwargs)
        self.kwargs.pop("callback", None)
        h = hs[0]
        for i in range(1, len(hs)):
            h += hs[i]
        return h / len(hs)

