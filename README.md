# Statistical Rethinking with PyTorch and Pyro

[Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/) is a great book for applied [Bayesian data analysis](https://en.wikipedia.org/wiki/Bayesian_statistics). The [accompanying code for the book](https://github.com/rmcelreath/rethinking) is written in **R** and **Stan**. It is then [ported to Python language](https://github.com/aloctavodia/Statistical-Rethinking-with-Python-and-PyMC3) using [PyMC3](https://docs.pymc.io/). Recently, [Pyro](http://pyro.ai/) emerges as a universal and flexible Bayesian modeling tool (check out [Pyro tutorial page](http://pyro.ai/examples/)), so to attract statisticians to this new library, I decided to make a *Pyronic* version for the code in this repository. Inspired by the *PyMC3onic* version, I keep the code in this repository as close as possible to the original code in the book.

To say a bit more about **Pyro**, it is a universal [probabilistic programming language](https://en.wikipedia.org/wiki/Probabilistic_programming_language) which is built on top of [PyTorch](https://pytorch.org/), a very popular platform for deep learning. If you are familiar with [numpy](http://www.numpy.org/), the transition from `numpy.array` to `torch.tensor` is rather straightforward (as demonstrated in [this tutorial](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html)).

## Contents

+ [Preface](http://nbviewer.jupyter.org/github/fehiepsi/rethinking-pyro/blob/master/00_preface.ipynb)

+ [Chapter 1. The Golem of Prague](http://nbviewer.jupyter.org/fehiepsi/fehiepsi/rethinking-pyro/blob/master/01_the_golem_of_prague.ipynb)

+ [Chapter 2. Small Worlds and Large Worlds](http://nbviewer.jupyter.org/github/fehiepsi/rethinking-pyro/blob/master/02_small_worlds_and_large_worlds.ipynb)

+ [Chapter 3. Sampling the Imaginary]

+ [Chapter 4. Linear Models]

+ [Chapter 5. Multivariate Linear Models]

+ [Chapter 6. Overfitting, Regularization, and Information Criteria]

+ [Chapter 7. Interactions]

+ [Chapter 8. Markov Chain Monte Carlo]

+ [Chapter 9. Big Entropy and the Generalized Linear Model]

+ [Chapter 10. Counting and Classification]

+ [Chapter 11. Monsters and Mixtures]

+ [Chapter 12. Multilevel Models]

+ [Chapter 13. Adventures in Covariance]

+ [Chapter 14. Missing Data and Other Opportunities]

+ [Chapter 15. Horoscopes](http://nbviewer.jupyter.org/github/fehiepsi/rethinking-pyro/blob/master/15_horoscopes.ipynb)

## Setup

You might use `pip` to install the following packages:
+ `jupyter` for displaying code, visualizations, text in one place,
+ `pandas` for reading data,
+ `matplotlib` for plotting,
+ `pytorch` for scientific computing,
+ `pyro-ppl` for probabilistic modeling.

Or you can install [conda](https://conda.io/miniconda.html) and create a `rethinking` environment from the `environment.yml` file:
```sh
conda env create -f environment.yml
```
