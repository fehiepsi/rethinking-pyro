import inspect
import re
import warnings

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
        self._max_tries = 5

    def _traces(self, *args, **kwargs):
        pyro.clear_param_store()

        # find good initial trace
        model_trace = poutine.trace(self.model).get_trace(*args, **kwargs)
        best_log_prob = model_trace.log_prob_sum()
        for i in range(50):
            trace = poutine.trace(self.model).get_trace(*args, **kwargs)
            log_prob = trace.log_prob_sum()
            if log_prob > best_log_prob:
                best_log_prob = log_prob
                model_trace = trace

        # lift model
        model_trace = poutine.util.prune_subsample_sites(model_trace)
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
        loc_param = pyro.param("auto_loc").unconstrained()
        optimizer = torch.optim.LBFGS((loc_param,), lr=0.1, max_iter=500, tolerance_grad=1e-3)
        loss_fn = Trace_ELBO().differentiable_loss

        def closure():
            optimizer.zero_grad()
            loss = loss_fn(lifted_model, delta_guide, *args, **kwargs)
            loss.backward()
            return loss

        optimizer.step(closure)
        guide = delta_guide.laplace_approximation(*args, **kwargs)

        # get posterior
        for i in range(self.num_samples):
            guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
            model_poutine = poutine.trace(poutine.replay(lifted_model, trace=guide_trace))
            yield model_poutine.get_trace(*args, **kwargs), 1.0

    def run(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for i in range(self._max_tries):
                try:
                    return super(MAP, self).run(*args, **kwargs)
                except Exception as e:
                    last_error = e
        raise last_error


def _formula_to_predictors(formula, data):
    y_name, expr_str = formula.split(" ~ ")
    y_node = {"name": y_name, "value": torch.tensor(data[y_name], dtype=torch.float)}
    y_node["mean"] = y_node["value"].mean()

    fit_intercept = True
    predictors = {"Intercept": False}
    col_to_num = dict(zip(data.columns, range(data.shape[1])))
    expr_list = expr_str.split(" + ")
    for expr in expr_list:
        if expr == "0":
            fit_intercept = False
        elif expr.startswith("I"):
            org_expr = expr
            for col in col_to_num:
                expr = expr.replace(col, "c{}".format(col_to_num[col]))
            eval_expr = expr.lstrip("I")
            eval_map = {"c{}".format(i): data.iloc[:, i] for i in range(data.shape[1])}
            predictors[org_expr] = torch.tensor(eval(eval_expr, eval_map), dtype=torch.float)
        elif expr.startswith("C"):
            cat_col = expr[2:-1]
            for cat in data[cat_col].unique():
                predictors["C(d){}".format(cat)] = torch.tensor(data[cat_col] == cat,
                                                                dtype=torch.float)
        elif expr in data.columns:
            predictors[expr] = torch.tensor(data[expr], dtype=torch.float)

    if fit_intercept:
        predictors["Intercept"] = True
    return y_node, predictors


class LM(MAP):
    def __init__(self, formula, data, num_samples=10000, start={}):
        self.formula = formula
        self.y_node, self.predictors = _formula_to_predictors(formula, data)
        self._predictor_means = {name: predictor.mean() for name, predictor
                                 in self.predictors.items() if name != "Intercept"}
        super(LM, self).__init__(self.model, num_samples, start)
        self._max_tries = 1

    def model(self, data=None):
        if data is None:
            y_node, predictors = self.y_node, self.predictors.copy()
        else:
            y_node, predictors = _formula_to_predictors(self.formula, data)
        fit_intercept = predictors.pop("Intercept")

        mu = 0
        if fit_intercept:
            mu = mu + pyro.sample("Intercept", dist.Normal(y_node["mean"], 10))

        for name, predictor in predictors.items():
            coef = pyro.sample(name, dist.Normal(0, 10))
            if fit_intercept:
                # use "centering trick"
                predictor = predictor - self._predictor_means[name]
            mu = mu + coef * predictor
        sigma = pyro.sample("sigma", dist.HalfCauchy(2))
        with pyro.plate("plate"):
            return pyro.sample(y_node["name"], dist.Normal(mu, sigma), obs=y_node["value"])

    def _get_centering_constant(self, coefs):
        center = 0
        for name, predictor_mean in self._predictor_means.items():
            center = center + coefs[name] * predictor_mean
        return center


def glimmer(formula, data):
    y_node, predictors = _formula_to_predictors(formula, data)
    fit_intercept = predictors.pop("Intercept")
    print("def model({}):".format(", ".join(predictors.keys()) + ", {}".format(y_node["name"])))
    mu_str = "    mu = "
    if fit_intercept:
        print("    intercept = pyro.sample('Intercept', dist.Normal(0, 10))")
        mu_str += "intercept + "
    for predictor in predictors:
        coef = predictor.replace("**", "_POW_").replace("*", "_MUL_").replace(" ", "")
        coef = re.sub("\W", "_", coef).strip("_")
        print("    b_{} = pyro.sample('{}', dist.Normal(0, 10))".format(coef, predictor))
        mu_str += "b_{} * {}".format(coef, predictor)
    print(mu_str)
    print("    sigma = pyro.sample('sigma', dist.HalfCauchy(2))")
    print("    with pyro.plate('plate'):")
    print("        return pyro.sample('{}', dist.Normal(mu, sigma), obs={})"
          .format(y_node["name"], y_node["name"]))


def extract_samples(posterior):
    nodes = poutine.util.prune_subsample_sites(posterior.exec_traces[0]).stochastic_nodes
    node_supports = posterior.marginal(nodes).support()
    return {latent: samples.detach() for latent, samples in node_supports.items()}


def coef(posterior):
    mean = {}
    node_supports = extract_samples(posterior)
    for node, support in node_supports.items():
        mean[node] = support.mean()
    # correct `intercept` due to "centering trick"
    if isinstance(posterior, LM) and "Intercept" in mean:
        center = posterior._get_centering_constant(mean)
        mean["Intercept"] = mean["Intercept"] - center
    return mean


def vcov(posterior):
    node_supports = extract_samples(posterior)
    packed_support = torch.cat([support.reshape(support.size(0), -1)
                                for support in node_supports.values()], dim=1)
    cov_scheme = WelfordCovariance(diagonal=False)
    for sample in packed_support:
        cov_scheme.update(sample)
    return cov_scheme.get_covariance()


def precis(posterior, corr=False, digits=2):
    if isinstance(posterior, TracePosterior):
        node_supports = extract_samples(posterior)
    else:
        node_supports = posterior
    df = pd.DataFrame(columns=["Mean", "StdDev", "|0.89", "0.89|"])
    for node, support in node_supports.items():
        if support.dim() == 1:
            hpdi = stats.hpdi(support, prob=0.89)
            df.loc[node] = [support.mean().item(), support.std().item(),
                            hpdi[0].item(), hpdi[1].item()]
        else:
            support = support.reshape(support.size(0), -1)
            mean = support.mean(0)
            std = support.std(0)
            hpdi = stats.hpdi(support, prob=0.89)
            for i in range(mean.size(0)):
                df.loc["{}[{}]".format(node, i)] = [mean[i].item(), std[i].item(),
                                                    hpdi[0, i].item(), hpdi[1, i].item()]
    # correct `intercept` due to "centering trick"
    if isinstance(posterior, LM) and "Intercept" in df.index:
        center = posterior._get_centering_constant(df["Mean"].to_dict()).item()
        df.loc["Intercept", ["Mean", "|0.89", "0.89|"]] -= center

    if corr:
        cov = vcov(posterior)
        corr = cov / cov.diag().ger(cov.diag()).sqrt()
        for i, node in enumerate(df.index):
            df[node] = corr[:, i]
    return df.round(digits)


def link(posterior, data=None, n=1000):
    obs_node = posterior.exec_traces[0].observation_nodes[-1]
    mu = []
    if data is None:
        for i in range(n):
            idx = posterior._categorical.sample().item()
            trace = posterior.exec_traces[idx]
            mu.append(trace.nodes[obs_node]["fn"].mean)
    else:
        obs_keys = inspect.signature(posterior.model).parameters - data.keys()
        obs_dict = {key: None for key in obs_keys}
        predictive = TracePredictive(poutine.lift(posterior.model, lambda: None),
                                     posterior, n).run(**data, **obs_dict)
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
        obs_keys = inspect.signature(posterior.model).parameters - data.keys()
        obs_dict = {key: None for key in obs_keys}
        predictive = TracePredictive(poutine.lift(posterior.model, lambda: None),
                                     posterior, n).run(**data, **obs_dict)
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
        plt.plot([se_df["WAIC"] - se_df["SE"], se_df["WAIC"] + se_df["SE"]],
                 [model, model], c="k")
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
