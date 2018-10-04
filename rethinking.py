import pandas as pd
import torch

import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.contrib.autoguide import AutoDelta
from pyro.infer import SVI, Trace_ELBO


class LinearModel(object):
    
    def __init__(self, y, fit_intercept=True, **kwargs):
        name = kwargs.pop("name", "")
        self.y = y
        self.y_name = pyro.params.param_with_module_name(name, "y")
        self.factors = kwargs
        self.factor_names = {factor: pyro.params.param_with_module_name(name, factor)
                             for factor in kwargs}
        self.fit_intercept = fit_intercept
        if fit_intercept:
            self.intercept_name = pyro.params.param_with_module_name(name, "intercept")
            self.intercept_init = y.mean()
        self.sigma = y.std()
        
    def _get_y_pred(self):
        weights = {factor: pyro.param(self.factor_names[factor], torch.tensor(0.))
                   for factor in self.factors}
        y_pred = sum([weights[factor] * self.factors[factor] for factor in self.factors])
        if self.fit_intercept:
            y_pred += pyro.param(self.intercept_name, self.intercept_init)
        return y_pred
        
    def _model(self):
        y_pred = self._get_y_pred()
        pyro.sample(self.y_name, dist.Normal(y_pred, self.sigma), obs=self.y)
    
    def fit(self, lr=1, num_steps=1000):
        svi = SVI(self._model, lambda: None, optim.Adam({"lr": lr}), Trace_ELBO())
        losses = []
        for _ in range(num_steps):
            losses.append(svi.step())
        return losses
        
    def coef(self):
        weights = {factor: pyro.param(self.factor_names[factor], torch.tensor(0.)).detach()
                   for factor in self.factors}
        if self.fit_intercept:
            weights["intercept"] = pyro.param(self.intercept_name,
                                              self.intercept_init).detach()
        return weights
    
    def resid(self):
        y_pred = self._get_y_pred()
        return (self.y - y_pred).detach()
    

class MAP(object):
    
    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self._guide = AutoDelta(model)

    def fit(self, lr=1, num_steps=1000):
        svi = SVI(self.model, self._guide, optim.Adam({"lr": lr}), Trace_ELBO())
        losses = []
        for _ in range(num_steps):
            losses.append(svi.step(**self.kwargs))
        return losses

    def precis(self, corr=False, digits=2, depth=None):
        mean = self._guide.median()
        mean = {latent: mean[latent].detach() for latent in mean}
        corr = self._guide.covariance(**self.kwargs)
        packed_mean = torch.cat([mean[latent].unsqueeze(0) for latent in mean])
        packed_std_dev = corr.diag().sqrt()
        packed_quantiles = dist.Normal(packed_mean, packed_std_dev).icdf(
            torch.tensor([[0.055], [0.945]]))
        std_dev = {}
        quantile_5_5 = {}
        quantile_94_5 = {}
        for i, latent in enumerate(mean):
            mean[latent] = mean[latent].numpy()
            std_dev[latent] = packed_std_dev[i].numpy()
            quantile_5_5[latent] = packed_quantiles[0, i].numpy()
            quantile_94_5[latent] = packed_quantiles[1, i].numpy()
        
        return pd.DataFrame.from_dict({"Mean": mean, "StdDev": std_dev, "5.5%": quantile_5_5,
                                       "94.5%": quantile_94_5}).round(digits)
        
    def extract_samples(self, n):
        pass


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


def precis(post):
    pass
