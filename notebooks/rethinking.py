import logging
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.optim as optim
import pyro.poutine as poutine
from pyro.contrib.autoguide import AutoDelta
from pyro.infer import SVI, Trace_ELBO


class MAP(object):

    def __init__(self, name, model, start={}, **kwargs):
        self._model = model
        self._guide = AutoDelta(model, prefix=name)
        self._kwargs = kwargs
        self._latent_shapes = {}
        for latent, site in poutine.trace(model).get_trace(**kwargs).iter_stochastic_nodes():
            if start == "mean":
                pyro.param("{}_{}".format(self._guide.prefix, latent), site["fn"].mean,
                           constraint=site["fn"].support)
            elif latent in start:
                pyro.param("{}_{}".format(self._guide.prefix, latent), start[latent],
                           constraint=site["fn"].support)
            self._latent_shapes[latent] = site["value"].shape

    def fit(self, lr=1, num_steps=1000):
        svi = SVI(self._model, self._guide, optim.Adam({"lr": lr}), Trace_ELBO())
        losses = []
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("once", "Encountered NaN: loss", UserWarning)
            for _ in range(num_steps):
                losses.append(svi.step(**self._kwargs))
        if w:
            logging.warning(str(w[-1].message)
                            + "\n\nThe start values for the parameters were invalid. This "
                            "could be caused by missing values (NA) in the data or by start "
                            "values outside the parameter constraints. If there are no NA "
                            "values in the data, try using explicit start values.")
        return losses

    def coef(self, full=False):
        mean = self._guide.median()
        if full:
            return torch.cat([m.detach().reshape(-1) for m in mean.values()])
        return {latent: mean[latent].detach() for latent in mean}

    def vcov(self):
        return self._guide.covariance(**self._kwargs)

    def precis(self, prob=0.89, corr=False, digits=2, depth=1):
        return_dict = False
        packed_mean = self.coef(full=True)
        cov = self.vcov()
        packed_std_dev = cov.diag().sqrt()
        left_prob, right_prob = (1 - prob) / 2, (1 + prob) / 2
        packed_quantiles = dist.Normal(packed_mean, packed_std_dev).icdf(
            torch.tensor([[left_prob], [right_prob]]))

        mean, std_dev, quantile_left, quantile_right = {}, {}, {}, {}
        pos = 0
        for latent, shape in self._latent_shapes.items():
            numel = shape.numel()
            if depth == 1 and numel > 1:
                return_dict = True
                next_pos = pos + numel
                mean[latent] = packed_mean[pos:next_pos].tolist()
                std_dev[latent] = packed_std_dev[pos:next_pos].tolist()
                quantile_left[latent] = packed_quantiles[0, pos:next_pos].tolist()
                quantile_right[latent] = packed_quantiles[1, pos:next_pos].tolist()
                pos = next_pos
            else:
                for i in range(numel):
                    name = "{}[{}]".format(latent, i) if numel > 1 else latent
                    mean[name] = packed_mean[pos].tolist()
                    std_dev[name] = packed_std_dev[pos].tolist()
                    quantile_left[name] = packed_quantiles[0, pos].tolist()
                    quantile_right[name] = packed_quantiles[1, pos].tolist()
                    pos = pos + 1

        precis_dict = {"Mean": mean, "StdDev": std_dev,
                       "{:.1f}%".format(left_prob * 100): quantile_left,
                       "{:.1f}%".format(right_prob * 100): quantile_right}
        if return_dict:
            return precis_dict

        precis = pd.DataFrame.from_dict(precis_dict)
        if corr:
            corr_matrix = cov / (packed_std_dev.unsqueeze(1) * packed_std_dev)
            corr_dict = {}
            pos = 0
            for latent, shape in self._latent_shapes.items():
                numel = shape.numel()
                for i in range(numel):
                    name = "{}[{}]".format(latent, i) if numel > 1 else latent
                    corr_dict[name] = corr_matrix[:, pos].tolist()
                    pos = pos + 1
            precis = precis.assign(**corr_dict)
        return precis.astype("float").round(digits).loc[mean.keys(), :]

    def extract_samples(self, n=10000):
        packed_samples = dist.MultivariateNormal(self.coef(full=True), self.vcov()).sample(
            torch.Size([n]))
        samples = {}
        pos = 0
        for latent, shape in self._latent_shapes.items():
            numel = shape.numel()
            samples[latent] = packed_samples[:, pos:(pos + numel)].reshape((n,) + shape)
            pos = pos + numel
        return samples

    def _get_factors_matrix(self, **kwargs):
        factors_list = []
        for factor in list(self._kwargs)[:-1]:
            if factor in kwargs:
                factors_list.append(kwargs[factor])
            else:
                factors_list.append(self._kwargs[factor])
        factors_list.insert(0, torch.ones(factors_list[0].size(0)))
        return torch.stack(factors_list)

    def link(self, n=1000, **kwargs):
        factors_matrix = self._get_factors_matrix(**kwargs)
        coefs_matrix = dist.MultivariateNormal(self.coef(full=True), self.vcov()).sample(
            torch.Size([n]))
        return coefs_matrix[:, :-1].matmul(factors_matrix)

    def sim(self, n=1000, **kwargs):
        factors_matrix = self._get_factors_matrix(**kwargs)
        coefs_matrix = dist.MultivariateNormal(self.coef(full=True), self.vcov()).sample(
            torch.Size([n]))
        mu = coefs_matrix[:, :-1].matmul(factors_matrix)
        sigma = coefs_matrix[:, -1:]
        return dist.Normal(mu, sigma).sample()

    def plot_precis(self, **kwargs):
        precis = self.precis(**kwargs)
        plt.plot(precis["Mean"], precis.index, "ko", fillstyle="none")
        plt.xlabel("Value")
        for latent in precis.index:
            plt.plot(precis.loc[latent, ["5.5%", "94.5%"]], [latent, latent], c="k", lw=1.5)
        plt.gca().invert_yaxis()
        plt.axvline(x=0, c="k", lw=1, alpha=0.2)
        plt.grid(axis="y", linestyle="--")


class LM(MAP):

    def __init__(self, name, start={}, fit_intercept=True, **kwargs):
        self.fit_intercept = fit_intercept
        if "intercept" in kwargs and kwargs["intercept"] == 0:
            self.fit_intercept = False
        self._sigma_name = "{}_sigma".format(name)
        super(LM, self).__init__(name, self._model, start, **kwargs)

    def _model(self, **kwargs):
        y_pred = pyro.sample("intercept", dist.Normal(0, 10)) if self.fit_intercept else 0
        for i, (factor, value) in enumerate(kwargs.items()):
            if i == len(kwargs) - 1:
                return pyro.sample(factor, dist.Normal(y_pred, value.std() / 10), obs=value)
            coef = pyro.sample(factor, dist.Normal(0, 10))
            y_pred = y_pred + coef * value

    def resid(self):
        factors_matrix = self._get_factors_matrix()
        coefs = self.coef(full=True)
        if not self.fit_intercept:
            factors_matrix = factors_matrix[1:]
        return list(self._kwargs.values())[-1] - coefs.matmul(factors_matrix)


def dens(samples, c=None, lw=None, xlab=None):
    bw_factor = (0.75 * samples.size(0)) ** (-0.2)
    bw = 0.5 * bw_factor * samples.std()
    x_min = samples.min()
    x_max = samples.max()
    x = torch.linspace(x_min, x_max, 1000)
    y = dist.Normal(samples, bw).log_prob(x.unsqueeze(-1)).logsumexp(-1).exp()
    y = y / y.sum() * (x.size(0) / (x_max - x_min))
    plt.plot(x.tolist(), y.tolist(), c=c, lw=lw)
    plt.xlabel(xlab)
    plt.ylabel("Density")


def quantile(samples, probs=(0.25, 0.5, 0.75), dim=-1):
    probs = probs if isinstance(probs, (list, tuple)) else (probs,)
    sorted_samples = samples.sort(dim)[0]
    masses = torch.tensor(probs) * samples.size(dim)
    lower_quantiles = sorted_samples.index_select(dim, masses.long().clamp(min=0))
    upper_quantiles = sorted_samples.index_select(dim, masses.long())
    dim = (samples.dim() + dim) if dim < 0 else dim
    t = masses.frac().reshape((len(probs),) + (-1,) * (samples.dim() - dim - 1))
    quantiles = (1 - t) * lower_quantiles + t * upper_quantiles
    return quantiles


def PI(samples, prob=0.89, dim=-1):
    return quantile(samples, ((1 - prob) / 2., (1 + prob) / 2.), dim)


def HPDI(samples, prob=0.89, dim=-1):
    full_size = samples.size(dim)
    mass = int(prob * full_size)
    sorted_samples = samples.sort(dim)[0]
    intervals = (sorted_samples.index_select(dim, torch.tensor(range(mass, full_size)))
                 - sorted_samples.index_select(dim, torch.tensor(range(full_size - mass))))
    start = intervals.argmin(dim)
    indices = torch.stack([start, start + mass], dim)
    return torch.gather(sorted_samples, dim, indices)


def precis(post, prob=0.89, digits=2):
    mean, std_dev, hpd_left, hpd_right = {}, {}, {}, {}
    for latent in post:
        mean[latent] = post[latent].mean().tolist()
        std_dev[latent] = post[latent].std().tolist()
        hpdi = HPDI(post[latent], prob)
        hpd_left[latent] = hpdi[0].tolist()
        hpd_right[latent] = hpdi[1].tolist()
    precis = pd.DataFrame.from_dict({"Mean": mean, "StdDev": std_dev,
                                     "|0.89": hpd_left, "0.89|": hpd_right})
    return precis.astype("float").round(digits).loc[post.keys(), :]


def glimmer(family="normal", **kwargs):
    print("def model({}):".format(", ".join(kwargs.keys())))
    print("    intercept = pyro.sample('Intercept', dist.Normal(0, 10))")
    mu_str = "    mu = intercept + "
    args = list(kwargs)
    for i, latent in enumerate(args[:-1]):
        print("    b_{} = pyro.sample('b_{}', dist.Normal(0, 10))".format(latent, latent))
        mu_str += "b_{} * {}".format(latent, latent)
    print(mu_str)
    print("    sigma = pyro.sample('sigma', dist.HalfCauchy(2))")
    print("    return pyro.sample('{}', dist.Normal(mu, sigma), obs={})"
          .format(args[-1], args[-1]))