# Statistical Rethinking with PyTorch and Pyro

[Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/) is a great book for applied [Bayesian data analysis](https://en.wikipedia.org/wiki/Bayesian_statistics). The [accompanying code for the book](https://github.com/rmcelreath/rethinking) is written in **R** and **Stan**. It is then [ported to Python language](https://github.com/aloctavodia/Statistical-Rethinking-with-Python-and-PyMC3) using [PyMC3](https://docs.pymc.io/). Recently, [Pyro](http://pyro.ai/) emerges as a scalable and flexible Bayesian modeling tool (check out [Pyro tutorial page](http://pyro.ai/examples/)), so to attract statisticians to this new library, I decided to make a *Pyronic* version for the code in this repository. Inspired by the *PyMC3onic* version, I keep the code in this repository as close as possible to the original code in the book.

To say a bit more about **Pyro**, it is a universal [probabilistic programming language](https://en.wikipedia.org/wiki/Probabilistic_programming_language) which is built on top of [PyTorch](https://pytorch.org/), a very popular platform for deep learning. If you are familiar with [numpy](http://www.numpy.org/), the transition from `numpy.array` to `torch.tensor` is rather straightforward (as demonstrated in [this tutorial](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html)).

## Contents

+ [Preface](http://nbviewer.jupyter.org/github/fehiepsi/rethinking-pyro/blob/master/notebooks/00_preface.ipynb)

+ [Chapter 1. The Golem of Prague](http://nbviewer.jupyter.org/github/fehiepsi/rethinking-pyro/blob/master/notebooks/01_the_golem_of_prague.ipynb)

+ [Chapter 2. Small Worlds and Large Worlds](http://nbviewer.jupyter.org/github/fehiepsi/rethinking-pyro/blob/master/notebooks/02_small_worlds_and_large_worlds.ipynb)

+ [Chapter 3. Sampling the Imaginary](http://nbviewer.jupyter.org/github/fehiepsi/rethinking-pyro/blob/master/notebooks/03_sampling_the_imaginary.ipynb)

+ [Chapter 4. Linear Models](http://nbviewer.jupyter.org/github/fehiepsi/rethinking-pyro/blob/master/notebooks/04_linear_models.ipynb)

+ [Chapter 5. Multivariate Linear Models](http://nbviewer.jupyter.org/github/fehiepsi/rethinking-pyro/blob/master/notebooks/05_multivariate_linear_models.ipynb)

+ [Chapter 6. Overfitting, Regularization, and Information Criteria](http://nbviewer.jupyter.org/github/fehiepsi/rethinking-pyro/blob/master/notebooks/06_overfitting_regularization_and_information_criteria.ipynb)

+ [Chapter 7. Interactions](http://nbviewer.jupyter.org/github/fehiepsi/rethinking-pyro/blob/master/notebooks/07_interactions.ipynb)

+ [Chapter 8. Markov Chain Monte Carlo](http://nbviewer.jupyter.org/github/fehiepsi/rethinking-pyro/blob/master/notebooks/08_markov_chain_monte_carlo.ipynb) (in progress)

+ [Chapter 9. Big Entropy and the Generalized Linear Model](http://nbviewer.jupyter.org/github/fehiepsi/rethinking-pyro/blob/master/notebooks/09_big_entropy_and_the_generalized_linear_model.ipynb)

+ [Chapter 10. Counting and Classification]

+ [Chapter 11. Monsters and Mixtures]

+ [Chapter 12. Multilevel Models]

+ [Chapter 13. Adventures in Covariance]

+ [Chapter 14. Missing Data and Other Opportunities]

+ [Chapter 15. Horoscopes](http://nbviewer.jupyter.org/github/fehiepsi/rethinking-pyro/blob/master/notebooks/15_horoscopes.ipynb)

## Usage

For convenience, I wrap most used classes and functions in the [rethinking.py](./notebooks/rethinking.py) script. The functions `dens` (kernel density estimation), `quantile`, `PI` (percentile interval), `HPDI` (highest posterior density interval), `precis` are used to summarize samples obtained from the modelling. `compare`, `coeftab`, `ensemble` are for comparing and ensembling models. I use PyTorch to get the results for most of these functions.

### Linear model

You can define a linear model of distance over speed in `cars` dataset as follow.

```
# make sure cars_speed and cars_dist are torch.tensor
m = LM("m", mass=mass, brain=brain)

# fit the model
m.fit()
```

Because we have to do inference for many models in one notebook, the first argument of `LM` is its name. It is unique for each model in a notebook and it helps avoid confliction of parameters of different models.

Note that under the hood, I use the **centering** trick (as discussed in the Section 7.3 of the book). In my opinion, this is one of the best tricks ever in doing statistics! For example, without this trick, it is impossible for me to fit a linear regression model on the following brain~mass dataset:

```
# try it yourself: build a model to fit the brain over mass :)
mass = torch.tensor([37.0, 35.5, 34.5, 41.5, 55.5, 61.0, 53.5])
brain = torch.tensor([438., 452, 612, 521, 752, 871, 1350])
```

### MAP

To do [maximum a posteriori inference](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation), you can use the class `MAP` as follows.

```
# define a Pyronic model
def model(weight, height):
    a = pyro.sample("a", dist.Normal(178, 100))
    b = pyro.sample("b", dist.Normal(0, 10))
    mu = a + b * weight
    sigma = pyro.sample("sigma", dist.Uniform(0, 50))
    with pyro.plate("plate"):
        pyro.sample("height", dist.Normal(a + b * weight, sigma), obs=height)

# feed the model and data into MAP
m = MAP("m", model, weight=weight, height=height)

# optimize parameters
m.fit()

# summarize the result, use `corr=True` to display the correlation
m.precis(corr=True)
```

There are something shoud be noted about `MAP`. The `with pyro.plate(...):` statement is used to tell Pyro that the variables in its context are conditionally independent (see [pyro.plate docs](http://docs.pyro.ai/en/dev/primitives.html#pyro.plate)). Without this statement, Pyro will throw some error about shape of `log_prob` at the `height` site. When you use Pyro, it is recommended to enable validation (by calling `pyro.enable_validation()`) to detect this kind of error (and other errors such as value of a distribution is outside of its support).

To generate latent samples from quadratic approximation, you can call `m.extract_samples(n=1000)`. Other methods such as `m.link()`, `m.sim()`, `m.WAIC()` are derived from these latent samples. Their functionalities are the same as in the book.

If you take a look at the code, you will notice that there are many `poutine.foo` statements. These effect handlers are quite flexible tools to do inference. You might be confused about how to use them at the first glance but after playing with them for a while, you'll fall in love with them. Please checkout [theirs docs](http://docs.pyro.ai/en/dev/poutine.html) for more information.

### Optimization

In Pyro, we often use stochastic optimizers such as `SGD`, `Adam`,... to optimize parameters. These optimizers are good for models with large dataset. However, for models with small dataset as in the book, I have a hard time to fit them using the above optimizers. Tuning learning rate, number of iterations,... does not help at all (the error `sigma` is usually exploded), so I use [LBFGS](https://pytorch.org/docs/stable/optim.html#torch.optim.LBFGS) instead. It turns out that optimum values are obtained easily with `LBFGS` without having to tuning anything!

### Initialization

In Pyro, initial latent variables are generated randomly from priors. However, I found that such initialization strategy is not good for models in the book. So I follow the Stan way to initialize latent variables randomly from the interval $\pm 2$ around the median point in the unconstrained space (see [Stan reference manual](https://github.com/stan-dev/stan/releases/download/v2.18.0/reference-manual-2.18.0.pdf)). It helps for many cases!

If you want to feed start values for latent variables, just simply put them into the `start` argument in the definition of MAP. For example, you can do `m = MAP("m", model, start={"a": torch.tensor(0.)}, ...)`.

### HMC

in progress...

## Setup

You might use `pip install -r requirements.txt` (not recommended for now) to install the following packages:
+ `jupyter` for displaying code, visualizations, text in one place,
+ `pandas` for reading data,
+ `matplotlib` for plotting,
+ `torch` for scientific computing,
+ `pyro-ppl` for probabilistic modeling.

Or you can use [conda](https://conda.io/miniconda.html) to create a `rethinking` environment from the `environment.yml` file:
```sh
conda env create -f environment.yml
```
