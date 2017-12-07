# Statistical Rethinking with Python and Pyro

[Statistical Rethinking](http://xcelab.net/rm/statistical-rethinking/) is an incredible good introductory book to Bayesian Statistics, its follows a _Jaynesian_ and practical approach with very good examples and clear explanations.

In this repository we ported the codes ([originally in R and Stan](https://github.com/rmcelreath/rethinking)) in the book to Pyro. We are trying to keep the examples as close as possible to those in the book, while at the same time trying to express them in the most _Pythonic_ and _Pyronic_ way we can.

## Note

This is a [Pyro](http://pyro.ai/) fork from the repository [Statistical Rethinking with Python and PyMC3](https://github.com/aloctavodia/Statistical-Rethinking-with-Python-and-PyMC3). We recommend the users go over Pyro's [examples](http://pyro.ai/examples/) to be able to appreciate this new deep probabilistic programming language. Pyro is currently in alpha stage so we expect many changes will occur. We will try to update the code of this fork for each new breaking version of Pyro.

## Display notebooks

[View Jupyter notebooks in nbviewer](http://nbviewer.jupyter.org/github/fehiepsi/Statistical-Rethinking-with-Python-and-Pyro/tree/pyro/)

## Contributing

All contributions are welcome!

Feel free to send PR to fix errors, improve the code or made comments that could help the user of this repository and readers of the book.

## Installing the dependencies

To install the dependencies to run these notebooks, you can use
[Anaconda](https://www.continuum.io/downloads). Once you have installed
Anaconda, run:

    conda env create -f environment.yml

to install all the dependencies into an isolated environment. You can switch to
this environment by running:

    source activate stat-rethink-pyro


---

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br /><span>Statistical Rethinking with Python and Pyro</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="https://github.com/fehiepsi/Statistical-Rethinking-with-Python-and-Pyro/graphs/contributors" property="cc:attributionName" rel="cc:attributionURL">All Contributors</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
