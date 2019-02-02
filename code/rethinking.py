import inspect
import os
import re
import warnings

import pandas as pd
import seaborn as sns
import torch
import torch.multiprocessing as mp
from torch.distributions import transform_to, constraints

import pyro
import pyro.distributions as dist
import pyro.ops.stats as stats
import pyro.poutine as poutine
from pyro.contrib.autoguide import AutoLaplaceApproximation
from pyro.infer import TracePosterior, TracePredictive, Trace_ELBO
from pyro.infer.mcmc import MCMC
from pyro.ops.welford import WelfordCovariance

os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.simplefilter("ignore", FutureWarning)

mp.set_sharing_strategy("file_system")
sns.set(font_scale=1.25, rc={"figure.figsize": (8, 6)})

pyro.enable_validation()
pyro.set_rng_seed(0)


class MAP(TracePosterior):
    def __init__(self, model, num_samples=10000, start={}):
        super(MAP, self).__init__()
        self.model = model
        self.num_samples = num_samples
        self.start = start

    def _traces(self, *args, **kwargs):
        pyro.clear_param_store()

        # find good initial trace
        model_trace = poutine.trace(self.model).get_trace(*args, **kwargs)
        best_log_prob = model_trace.log_prob_sum()
        for i in range(10):
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
                    prior[name] = dist.HalfCauchy(200)
                else:
                    prior[name] = dist.Normal(0, 1000)
                unpacked[name] = pyro.param(name).unconstrained().clone().detach()
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
            for i in range(10):
                try:
                    return super(MAP, self).run(*args, **kwargs)
                except Exception as e:
                    last_error = e
        raise last_error


def _formula_to_predictors(formula, data):
    dtype = torch.get_default_dtype()
    y_name, expr_str = formula.split(" ~ ")
    y_node = {"name": y_name, "value": torch.tensor(data[y_name], dtype=dtype)}
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
            predictors[org_expr] = torch.tensor(eval(eval_expr, eval_map), dtype=dtype)
        elif expr.startswith("C"):
            cat_col = expr[2:-1]
            for cat in data[cat_col].unique():
                predictors["C(d){}".format(cat)] = torch.tensor(data[cat_col] == cat, dtype=dtype)
        elif expr in data.columns:
            predictors[expr] = torch.tensor(data[expr], dtype=dtype)

    if fit_intercept:
        predictors["Intercept"] = True
    return y_node, predictors


class LM(MAP):
    def __init__(self, formula, data, num_samples=10000, start={}, centering=True):
        self.formula = formula
        self.y_node, self.predictors = _formula_to_predictors(formula, data)
        self._predictor_means = {name: predictor.mean() for name, predictor
                                 in self.predictors.items() if name != "Intercept"}
        self.centering = centering
        super(LM, self).__init__(self.model, num_samples, start)

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
            if fit_intercept and self.centering:
                # use "centering trick"
                predictor = predictor - self._predictor_means[name]
            mu = mu + coef * predictor
        sigma = pyro.sample("sigma", dist.HalfCauchy(2))
        with pyro.plate("plate"):
            return pyro.sample(y_node["name"], dist.Normal(mu, sigma), obs=y_node["value"])

    def _get_centering_constant(self, coefs):
        center = torch.tensor(0.)
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
    node_supports = posterior.marginal(nodes).support(flatten=True)
    return {latent: samples.detach() for latent, samples in node_supports.items()}


def coef(posterior):
    mean = {}
    node_supports = extract_samples(posterior)
    for node, support in node_supports.items():
        mean[node] = support.mean(dim=0)
    # correct `intercept` due to "centering trick"
    if isinstance(posterior, LM) and "Intercept" in mean and posterior.centering:
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
    return cov_scheme.get_covariance(regularize=False)


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
    if isinstance(posterior, LM) and "Intercept" in df.index and posterior.centering:
        center = posterior._get_centering_constant(df["Mean"].to_dict()).item()
        df.loc["Intercept", ["Mean", "|0.89", "0.89|"]] -= center

    if corr:
        cov = vcov(posterior)
        corr = cov / cov.diag().ger(cov.diag()).sqrt()
        for i, node in enumerate(df.index):
            df[node] = corr[:, i]

    if isinstance(posterior, MCMC):
        diagnostics = posterior.marginal(df.index.tolist()).diagnostics()
        df = pd.concat([df, pd.DataFrame(diagnostics).T.astype(float)], axis=1)

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
        data = {name: data[name] if name in data else None
                for name in inspect.signature(posterior.model).parameters}
        predictive = TracePredictive(poutine.lift(posterior.model, dist.Normal(0, 1)),
                                     posterior, n).run(**data)
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
        data = {name: data[name] if name in data else None
                for name in inspect.signature(posterior.model).parameters}
        predictive = TracePredictive(poutine.lift(posterior.model, dist.Normal(0, 1)),
                                     posterior, n).run(**data)
        for trace in predictive.exec_traces:
            obs.append(trace.nodes[obs_node]["value"])
    return torch.stack(obs).detach()


def compare(posteriors):
    post_ics = {}
    with torch.no_grad():
        for name in posteriors:
            post_ics[name] = posteriors[name].information_criterion(pointwise=True)
    n_cases = post_ics[name]["waic"].size(0)
    WAIC = {name: post_ics[name]["waic"].sum() for name in posteriors}
    pWAIC = {name: post_ics[name]["p_waic"].sum() for name in posteriors}
    SE = {name: (n_cases * post_ics[name]["waic"].var()).sqrt() for name in posteriors}
    table = pd.DataFrame({"WAIC": WAIC, "pWAIC": pWAIC}).sort_values(by="WAIC")
    table["dWAIC"] = table["WAIC"] - table.iloc[0, 0]
    table["weight"] = torch.nn.functional.softmax(-1/2 * torch.tensor(table["dWAIC"]), dim=0)
    table["SE"] = pd.Series(SE)
    dSE = []
    for i in range(table.shape[0]):
        WAIC0 = post_ics[table.index[0]]["waic"]
        WAICi = post_ics[table.index[i]]["waic"]
        dSE.append((n_cases * (WAICi - WAIC0).var()).sqrt())
    table["dSE"] = dSE
    return table.astype(float)


def ensemble(posteriors, data):
    weighted_num = (compare(posteriors)["weight"] * 1000).astype(int)
    weighted_num.iloc[-1] -= (sum(weighted_num) - 1000)
    links = []
    sims = []
    for name in weighted_num.index:
        num_samples = weighted_num[name]
        links.append(link(posteriors[name], data, num_samples).reshape(num_samples, -1))
        sims.append(sim(posteriors[name], data, num_samples).reshape(num_samples, -1))
    num_data = max(l.size(1) for l in links)
    links = [l.expand(-1, num_data) for l in links]
    sims = [s.expand(-1, num_data) for s in sims]
    return {"link": torch.cat(links), "sim": torch.cat(sims)}


def _worker(n, fn, fn_args, child_info=None):
    if child_info is not None:
        idx, event, queue = child_info
        pyro.set_rng_seed(idx)
    result = []
    for i in range(n):
        item = fn(*fn_args)
        result.append(item)
        queue.put((idx, item))
        event.wait()
        event.clear()
    return result


def replicate(n, fn, fn_args, mc_cores=None):
    mc_cores = mp.cpu_count() - 1 if mc_cores is None else mc_cores
    queue = mp.Queue()
    events = [mp.Event() for i in range(mc_cores)]
    processes = []
    for i in range(mc_cores):
        n_i = n // mc_cores + (i < n % mc_cores)
        child_info = (i, events[i], queue)
        p = mp.Process(target=_worker, args=(n_i, fn, fn_args, child_info), daemon=True)
        p.start()
        processes.append(p)

    result = []
    for i in range(n):
        idx, item = queue.get()
        result.append(item)
        events[idx].set()

    for i in range(mc_cores):
        processes[i].join()
    return result
