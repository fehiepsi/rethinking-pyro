import pandas as pd
import torch
from torch.distributions import transform_to, constraints

import pyro
import pyro.distributions as dist
import pyro.ops.stats as stats
import pyro.poutine as poutine
from pyro.contrib.autoguide import AutoLaplaceApproximation
from pyro.infer import TracePosterior, TracePredictive, Trace_ELBO
from pyro.ops.welford import WelfordCovariance


class MAP(TracePosterior):
    def __init__(self, model, num_samples=10000, start={}):
        super(MAP, self).__init__()
        self.model = model
        self.num_samples = num_samples
        self.start = start

    def _traces(self, *args, **kwargs):
        # find good initial trace
        model_trace = poutine.trace(self.model).get_trace(*args, **kwargs)
        best_log_prob = model_trace.log_prob_sum()
        for i in range(20):
            trace = poutine.trace(self.model).get_trace(*args, **kwargs)
            log_prob = trace.log_prob_sum()
            if log_prob > best_log_prob:
                best_log_prob = log_prob
                model_trace = trace

        # lift model
        prior, unpacked = {}, {}
        param_constraints = pyro.get_param_store().get_state()["constraints"]
        for name, node in model_trace.nodes.items():
            if node["type"] == "param":
                if param_constraints[name] is constraints.positive:
                    prior[name] = dist.HalfCauchy(2)
                else:
                    prior[name] = dist.Normal(0, 10)
                unpacked[name] = pyro.param(name).unconstrained()
            elif name in self.start:
                unpacked[name] = self.start[name]
            elif node["type"] == "sample" and not node["is_observed"]:
                unpacked[name] = transform_to(node["fn"].support).inv(node["value"])
        lifted_model = poutine.lift(self.model, prior)

        # define guide
        packed = torch.cat([v.clone().detach().reshape(-1) for v in unpacked.values()])
        pyro.param("auto_loc", packed)
        delta_guide = AutoLaplaceApproximation(lifted_model)

        # train guide
        optimizer = torch.optim.LBFGS((pyro.param("auto_loc").unconstrained(),), lr=0.1, max_iter=500)
        loss_and_grads = Trace_ELBO().loss_and_grads

        def closure():
            optimizer.zero_grad()
            return loss_and_grads(lifted_model, delta_guide, *args, **kwargs)

        optimizer.step(closure)
        guide = delta_guide.laplace_approximation(*args, **kwargs)

        # get posterior
        for i in range(self.num_samples):
            guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
            model_poutine = poutine.trace(poutine.replay(lifted_model, trace=guide_trace))
            yield model_poutine.get_trace(*args, **kwargs), 1.0

        pyro.clear_param_store()


class LM(MAP):
    def __init__(self, formula, data, num_samples=10000, start={}):
        y_name, predictor_names = formula.split(" ~ ")
        predictor_names = predictor_names.split(" + ")
        self.y = {"name": y_name, "value": torch.tensor(data[y_name].values, dtype=torch.float)}
        self.y["mean"] = self.y["value"].mean()
        self.predictors = {name: torch.tensor(data[name].values, dtype=torch.float) for name in predictor_names}
        self._predictor_means = {name: predictor.mean() for name, predictor in self.predictors.items()}
        super(LM, self).__init__(self._model, num_samples, start)

    def _model(self):
        intercept = pyro.sample("intercept", dist.Normal(self.y["mean"], 10))
        mu = intercept
        for name, predictor in self.predictors.items():
            coef = pyro.sample(name, dist.Normal(0, 10))
            # use "centering trick"
            mu = mu + coef * (predictor - self._predictor_means[name])
        sigma = pyro.sample("sigma", dist.HalfCauchy(2))
        with pyro.plate("plate"):
            return pyro.sample(self.y["name"], dist.Normal(mu, sigma), obs=self.y["value"])

    def _get_centering_constant(self, coefs):
        center = 0
        for name, predictor_mean in self._predictor_means.items():
            center = center + coefs[name] * predictor_mean
        return center


def glimmer(formula, family="normal"):
    print("def model({}):".format(", ".join(kwargs.keys())))
    print("    intercept = pyro.sample('intercept', dist.Normal(0, 10))")
    mu_str = "    mu = intercept + "
    args = list(kwargs)
    for i, latent in enumerate(args[:-1]):
        print("    b_{} = pyro.sample('b_{}', dist.Normal(0, 10))".format(latent, latent))
        mu_str += "b_{} * {}".format(latent, latent)
    print(mu_str)
    print("    sigma = pyro.sample('sigma', dist.HalfCauchy(2))")
    print("    with pyro.plate('plate'):")
    print("        return pyro.sample('{}', dist.Normal(mu, sigma), obs={})"
          .format(args[-1], args[-1]))


def coef(posterior):
    mean = {}
    node_supports = posterior.marginal(posterior.exec_traces[0].stochastic_nodes).support()
    for node, support in node_supports.items():
        mean[node] = support.mean().detach()
    # correct `intercept` due to "centering trick"
    if isinstance(posterior, LM):
        center = posterior._get_centering_constant(mean)
        mean["intercept"] = mean["intercept"] - center
    return mean


def vcov(posterior):
    node_supports = posterior.marginal(posterior.exec_traces[0].stochastic_nodes).support()
    packed_support = torch.cat([support.reshape(support.size(0), -1)
                                for support in node_supports.values()], dim=1)
    cov_scheme = WelfordCovariance(diagonal=False)
    for sample in packed_support:
        cov_scheme.update(sample)
    return cov_scheme.get_covariance().detach()


def precis(posterior, corr=False, digits=2):
    if isinstance(posterior, TracePosterior):
        node_supports = posterior.marginal(posterior.exec_traces[0].stochastic_nodes).support()
    else:
        node_supports = posterior
    mean, std_dev, lower_hpd, upper_hpd = {}, {}, {}, {}
    for node, support in node_supports.items():
        mean[node] = support.mean().item()
        std_dev[node] = support.std().item()
        hpdi = stats.hpdi(support, prob=0.89)
        lower_hpd[node] = hpdi[0].item()
        upper_hpd[node] = hpdi[1].item()
    # correct `intercept` due to "centering trick"
    if isinstance(posterior, LM):
        center = posterior._get_centering_constant(mean)
        mean["intercept"] = mean["intercept"] - center
        lower_hpd["intercept"] = lower_hpd["intercept"] - center
        upper_hpd["intercept"] = upper_hpd["intercept"] - center
    precis = pd.DataFrame.from_dict({"Mean": mean, "StdDev": std_dev, "|0.89": lower_hpd, "0.89|": upper_hpd})

    if corr:
        cov = vcov(posterior)
        corr = cov / cov.diag().ger(cov.diag()).sqrt()
        corr_dict = {}
        pos = 0
        for node in posterior.exec_traces[0].stochastic_nodes:
            corr_dict[node] = corr[:, pos].tolist()
            pos = pos + 1
        precis = precis.assign(**corr_dict)
    return precis.round(digits)


def link(posterior, data=None, n=1000):
    obs_node = posterior.exec_traces[0].observation_nodes[-1]
    mu = []
    if data is None:
        for i in range(n):
            idx = posterior._categorical.sample().item()
            trace = posterior.exec_traces[idx]
            mu.append(trace.nodes[obs_node]["fn"].mean)
    else:
        data[obs_node] = None
        predictive = TracePredictive(poutine.lift(posterior.model, lambda: None), posterior, n).run(**data)
        for trace in predictive.exec_traces:
            mu.append(trace.nodes[obs_node]["fn"].mean)
    return torch.stack(mu).detach()


def sim(posterior, data=None, n=1000):
    obs_node = posterior.exec_traces[0].observation_nodes[-1]
    obs = []
    if data is None:
        for i in range(n):
            idx = posterior._categorical.sample().item()
            trace = posterior.exec_traces[idx]
            obs.append(trace.nodes[obs_node]["fn"].sample())
    else:
        data[obs_node] = None
        predictive = TracePredictive(poutine.lift(posterior.model, lambda: None), posterior, n).run(**data)
        for trace in predictive.exec_traces:
            obs.append(trace.nodes[obs_node]["value"])
    return torch.stack(obs).detach()


def compare(*posteriors):
    WAIC_dict, WAIC_vec = {}, {}
    for m in models:
        WAIC = m.WAIC(pointwise=True)
        WAIC_vec[m.name] = WAIC.pop("WAIC_vec")
        WAIC_dict[m.name] = {w_name: w.item() for w_name, w in WAIC.items()}
    df = pd.DataFrame.from_dict(WAIC_dict).T.sort_values(by="WAIC")
    df["dWAIC"] = df["WAIC"] - df["WAIC"][0]
    weight = (-0.5 * torch.tensor(df["dWAIC"].values, dtype=torch.float)).exp()
    df["weight"] = (weight / weight.sum()).tolist()
    dSE = []
    for i in range(df.shape[0]):
        WAIC1 = WAIC_vec[df.index[0]]
        WAIC2 = WAIC_vec[df.index[i]]
        dSE.append((WAIC1.size(0) * (WAIC2 - WAIC1).var()).sqrt().item())
    df["dSE"] = dSE
    df.rename(columns={"se": "SE"}, inplace=True)
    return df[["WAIC", "pWAIC", "dWAIC", "weight", "SE", "dSE"]]


def compare_plot(waic_df):
    plt.plot(waic_df["WAIC"], waic_df.index, "ko", fillstyle="none")
    for i, model in enumerate(waic_df.index):
        se_df = waic_df.loc[model, :]
        plt.plot([se_df["WAIC"] - se_df["SE"], se_df["WAIC"] + se_df["SE"]], [model, model],
                 c="k")
        if i > 0:
            plt.plot([se_df["WAIC"] - se_df["dSE"], se_df["WAIC"] + se_df["dSE"]],
                     [i - 0.5, i - 0.5], c="gray")
    plt.plot(waic_df["WAIC"] - 2 * waic_df["pWAIC"], waic_df.index, "ko")
    plt.plot(waic_df["WAIC"][1:], [i + 0.5 for i in range(waic_df.shape[0] - 1)], "^",
             c="gray", fillstyle="none")
    plt.gca().invert_yaxis()
    plt.axvline(x=waic_df["WAIC"][0], c="k", alpha=0.2)
    plt.grid(axis="y", linestyle="--")
    plt.xlabel("deviance")
    plt.title("WAIC")


def coeftab(*models, PI=False):
    coef_df = pd.concat([m.precis().stack() for m in models], axis=1, sort=False)
    coef_df.columns = [m.name for m in models]
    mean_df = coef_df.unstack().xs("Mean", axis=1, level=1)
    keys = []
    for m in models:
        keys += list(m._latent_shapes)
    keys = list(dict.fromkeys(keys))
    mean_df = mean_df.loc[keys, :]
    mean_df.loc["nobs"] = [m.sim(n=1).size(-1) for m in models]
    if PI:
        lower_PI = coef_df.unstack().xs("5.5%", axis=1, level=1).loc[keys, :]
        upper_PI = coef_df.unstack().xs("94.5%", axis=1, level=1).loc[keys, :]
        return mean_df, lower_PI, upper_PI
    return mean_df


def ensemble(data, *models):
    n = int(1e3)
    shape = (n, list(data.values())[0].size(0))
    WAIC_df = compare(*models)
    weight = torch.tensor(WAIC_df["weight"].values, dtype=torch.float)
    models_dict = {m.name: m for m in models}
    links = [models_dict[m].link(**data, n=n).reshape(n, -1).expand(shape) for m in WAIC_df.index]
    sims = [models_dict[m].sim(**data, n=n).reshape(n, -1).expand(shape) for m in WAIC_df.index]
    index = torch.cat([torch.tensor([0]),
                       (weight.cumsum(0) * n).round().long().clamp(max=n)])
    weighted_link = torch.cat([l[index[i]:index[i + 1]] for i, l in enumerate(links)])
    weighted_sim = torch.cat([s[index[i]:index[i + 1]] for i, s in enumerate(sims)])
    return {"link": weighted_link, "sim": weighted_sim}


def chainmode(samples, adj=0.5):
    bw_factor = (0.75 * samples.size(0)) ** (-0.2)
    bw = adj * bw_factor * samples.std()
    x_min = samples.min()
    x_max = samples.max()
    x = torch.linspace(x_min, x_max, 1000)
    y = dist.Normal(samples, bw).log_prob(x.unsqueeze(-1)).logsumexp(-1).exp()
    y = y / y.sum() * (x.size(0) / (x_max - x_min))
    if not plot:
        return x, y
    plt.plot(x.tolist(), y.tolist(), c=c, lw=lw)
    plt.xlabel(xlab)
    plt.ylabel("Density")
