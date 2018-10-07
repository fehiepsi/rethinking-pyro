import pandas as pd
import torch

import pyro
import pyro.distributions as dist
import pyro.optim as optim
import pyro.poutine as poutine
from pyro.contrib.autoguide import AutoDelta
from pyro.infer import SVI, Trace_ELBO


class LM(object):

    def __init__(self, fit_intercept=True, **kwargs):
        self.name = kwargs.pop("name", "")
        self.y = kwargs.pop(list(kwargs)[-1])
        self.factors = kwargs
        self.fit_intercept = fit_intercept
        self._sigma = self.y.std()
        
    def _get_coefs(self):
        coefs = {factor: pyro.param(pyro.params.param_with_module_name(self.name, factor),
                                    lambda: torch.tensor(0.)) for factor in self.factors}
        if self.fit_intercept:
            coefs["intercept"] = pyro.param(pyro.params.param_with_module_name(
                self.name, "intercept"), lambda: self.y.mean())
        return coefs

    def _get_y_pred(self):
        coefs = self._get_coefs()
        y_pred = sum([coefs[factor] * self.factors[factor] for factor in self.factors])
        if self.fit_intercept:
            y_pred += coefs["intercept"]
        return y_pred

    def _model(self):
        pyro.sample(pyro.params.param_with_module_name(self.name, "y"),
                    dist.Normal(self._get_y_pred(), self._sigma), obs=self.y)

    def fit(self, lr=1, num_steps=1000):
        svi = SVI(self._model, lambda: None, optim.Adam({"lr": lr}), Trace_ELBO())
        losses = []
        for _ in range(num_steps):
            losses.append(svi.step())
        return losses

    def coef(self):
        coefs = self._get_coefs()
        return {factor: coefs[factor].detach() for factor in coefs}

    def resid(self):
        return (self.y - self._get_y_pred()).detach()


class MAP(object):

    def __init__(self, model, start={}, **kwargs):
        self.model = model
        self._guide = AutoDelta(model, prefix=kwargs.pop("name", "auto"))
        self.kwargs = kwargs
        self._latents = []
        for latent, site in poutine.trace(model).get_trace(**kwargs).iter_stochastic_nodes():
            if latent in start:
                pyro.param("{}_{}".format(self._guide.prefix, latent), self.start[latent],
                           constraint=site["fn"].support)
            self._latents.append(latent)

    def fit(self, lr=1, num_steps=1000):
        svi = SVI(self.model, self._guide, optim.Adam({"lr": lr}), Trace_ELBO())
        losses = []
        for _ in range(num_steps):
            losses.append(svi.step(**self.kwargs))
        return losses

    def coef(self):
        mean = self._guide.median()
        return torch.cat([mean[latent].detach().unsqueeze(0) for latent in mean])

    def vcov(self):
        return self._guide.covariance(**self.kwargs)

    def precis(self, depth=1, prob=0.89, corr=False, digits=2):
        packed_mean = self.coef()
        cov = self.vcov()
        packed_std_dev = cov.diag().sqrt()
        left_prob, right_prob = (1 - prob) / 2, (1 + prob) / 2
        packed_quantiles = dist.Normal(packed_mean, packed_std_dev).icdf(
            torch.tensor([[left_prob], [right_prob]]))

        mean, std_dev, quantile_left, quantile_right = {}, {}, {}, {}
        for i, latent in enumerate(self._latents):
            mean[latent] = packed_mean[i].numpy()
            std_dev[latent] = packed_std_dev[i].numpy()
            quantile_left[latent] = packed_quantiles[0, i].numpy()
            quantile_right[latent] = packed_quantiles[1, i].numpy()

        precis = pd.DataFrame.from_dict({"Mean": mean, "StdDev": std_dev,
            "{:.1f}%".format(left_prob * 100): quantile_left,
            "{:.1f}%".format(right_prob * 100): quantile_right})
        if corr:
            corr_matrix = cov / (packed_std_dev.unsqueeze(1) * packed_std_dev)
            corr_dict = {}
            for i, latent in enumerate(self._latents):
                corr_dict[latent] = corr_matrix[:, i].numpy()
            precis = precis.assign(**corr_dict)
        return precis.astype("float").round(digits)

    def extract_samples(self, n=10000):
        packed_samples = dist.MultivariateNormal(self.coef(), self.vcov()).sample(
            torch.Size([n]))
        samples = {}
        for i, latent in enumerate(self._latents):
            samples[latent] = packed_samples[:, i]
        return samples

    def _get_factors_matrix(self, **kwargs):
        factors_list = []
        for factor in list(self.kwargs)[:-1]:
            if factor in kwargs:
                factors_list.append(kwargs[factor])
            else:
                factors_list.append(self.kwargs[factor])
        factors_list.insert(0, torch.ones(factors_list[0].size(0)))
        return torch.stack(factors_list)
        
    def link(self, n=1000, **kwargs):
        factors_matrix = self._get_factors_matrix(**kwargs)
        coefs_matrix = dist.MultivariateNormal(self.coef(), self.vcov()).sample(
            torch.Size([n]))
        return coefs_matrix[:, :-1].matmul(factors_matrix)
    
    def sim(self, n=1000, **kwargs):
        factors_matrix = self._get_factors_matrix(**kwargs)
        coefs_matrix = dist.MultivariateNormal(self.coef(), self.vcov()).sample(
            torch.Size([n]))
        mu = coefs_matrix[:, :-1].matmul(factors_matrix)
        sigma = coefs_matrix[:, -1:]
        return dist.Normal(mu, sigma).sample()


def HPDI(samples, dim=-1, prob=0.89):
    full_size = samples.size(dim)
    mass = int(prob * full_size)
    sorted_samples = samples.sort(dim)[0]
    intervals = (sorted_samples.index_select(dim, torch.tensor(range(mass, full_size)))
                 - sorted_samples.index_select(dim, torch.tensor(range(full_size - mass))))
    start = intervals.argmin(dim)
    indices = torch.stack([start, start + mass], dim)
    return torch.gather(sorted_samples, dim, indices)


def precis(samples, depth=1, prob=0.89, corr=False, digits=2):
    mean, std_dev, hpd_left, hpd_right = {}, {}, {}, {}
    for i, latent in enumerate(samples):
        mean[latent] = samples[latent].mean().numpy()
        std_dev[latent] = samples[latent].std().numpy()
        hpdi = HPDI(samples[latent], prob)
        hpd_left[latent] = hpdi[0].numpy()
        hpd_right[latent] = hpdi[1].numpy()
    return pd.DataFrame.from_dict({"Mean": mean, "StdDev": std_dev,
        "|{:.2f}".format(prob): hpd_left,
        "{:.2f}|".format(prob): hpd_right}).astype("float").round(digits)
