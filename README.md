<div align="center">
  <img src="logo-v2.png" alt="Logo" width="450">
</div>

---

[//]: # ([![PyPI Version]&#40;https://img.shields.io/pypi/v/your-package-name.svg&#41;]&#40;https://pypi.org/project/your-package-name/&#41;)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

[//]: # ([![Python Versions]&#40;https://img.shields.io/pypi/pyversions/your-package-name.svg&#41;]&#40;https://pypi.org/project/your-package-name/&#41;)
[//]: # ([![Coverage]&#40;https://img.shields.io/codecov/c/gh/your-username/your-package-name&#41;]&#40;https://codecov.io/gh/your-username/your-package-name&#41;)
[![GitHub Stars](https://img.shields.io/github/stars/GFaure9/raop)](https://github.com/your-username/your-package-name)
[![GitHub Forks](https://img.shields.io/github/forks/GFaure9/raop)](https://github.com/your-username/your-package-name)
[![Last Commit](https://img.shields.io/github/last-commit/GFaure9/raop)](https://github.com/your-username/your-package-name)


[//]: # (<div class="warning" style="background-color: #FFE8B2; border-left: 6px solid #FFC857; padding: 10px; margin-bottom: 20px; font-size: 16px;">)

[//]: # (  <p>⚠ WARNING ⚠ This package is in alpha stage of development. Use at your own risk.</p>)

[//]: # (</div>)

> :error: ** WARNING **: This package is in alpha stage of development. Use at your own risk.


# raop 
__r__(isk)-__a__(ware ) __o__(ption) __p__(ricing) Library

_raop_ is a Python package providing several tools to analyse various types of 
financial options while considering risk factors.

---

# Table of Contents 

1. [Key Features](#feats)
2. [Installation Instructions](#install)
3. [Usage Examples](#usage)
4. [Documentation](#documentation)

---

## 🌟 Key Features <a id="feats"></a>

* __Option Pricing__ and __Greeks Computation__ through multiple possible approaches
(Black-Scholes model, Binomial tree, Monte-Carlo simulations). Available types of options are:
  * European;
  * American;


Next options to be implemented will be: Asian, Barrier, Look-Back, Binary, Spread, Exchange, Chooser and Quanto.


* __Graph Generation__ to facilitate characterizing an option with sensitivity analyses
on its parameters.


* __Stochastic Processes Simulation__ with different possible processes. Available processes are currently:
  * Geometric Brownian Motion;
  * Arithmetic Brownian Motion;
  * Merton Jump-Diffusion;
  * Ornstein-Uhlenbeck process;
  * Cox-Ingersoll-Ross process;
  * Heston model;
  * Variance-Gamma model.


* Incoming features: 
  * __VaR Computation__
  * __Implied Volatility Computation__


## ⬇️ Installation Instructions <a id="install"></a>

To install the package, you can either use the quick method by using ``pip``, writing the following
command in your terminal:

```bash
pip install raop
```

Or you can clone the GitHub project and, in the root directory of the project (i.e. the one containing the
``setup.py`` file), use ``pip`` to install it. To do so, open a terminal and in the folder where you want
to install the package, run the following commands:

```bash
git clone https://github.com/GFaure9/raop.git
cd raop
pip install --upgrade setuptools  # making sure that an up-to-date version of setuptools is installed
pip install -e .
```

That's it! You can now start using the package 😊!

## 🚀 Usage Examples <a id="usage"></a>

Let's see how the package can be used through some examples.

We will first see how to instantiate a European call option and estimate its price with
the Black-Scholes model, with the Binomial model and with Monte-Carlo simulations.

We will then see how we can also compute the greeks of this option.

Finally, we will learn how to plot some graphs to understand how the option's price
and greeks depend on its parameters.

### 1. Creating an ``Option`` object

To define the option, we must provide its different parameters to the instance
of the ``Option`` class that will be created.

```py
from raop.options import Option

call_euro = Option(
  name="european",
  option_type="call",
  underlying_price=36,
  strike_price=40,
  time_to_maturity=1,
  risk_free_rate=0.06,
  volatility=0.2
)

print("The Option instance 'call_euro' has the following attributes:\n", call_euro.to_dict())
```

The script above will return the following result:

```
The Option instance 'call_euro' has the following attributes:
 {'name': 'european', 'option_type': 'call', 's': 36, 'k': 40, 'r': 0.06, 'time_to_maturity': 1, 'sigma': 0.2}
```

### 2. Option's price computation

To determine the valuation of the generated option, utilize the ``compute_price`` method while 
specifying the desired pricing model as an argument. Distinct pricing models can be imported from
the ``raop.pricing_models`` module. It may be needed to provide additional arguments depending on the
chosen pricing model (refer to the [Documentation](#documentation) for specific mandatory arguments).

#### a. With ``BlackScholes`` model

For instance, you can use the Black–Scholes–Merton model as follows:

```py
from raop.pricing_models import BlackScholes

bs_price = call_euro.compute_price(BlackScholes)

print(f"'call_euro' price estimated by Black-Scholes method is: {bs_price}")
```

The script above will return the following result:

```
'call_euro' price estimated by Black-Scholes method is: 2.1737264482268905
```

#### b. With ``Binomial`` model

You can also employ the Cox-Ross-Rubinstein Binomial model to calculate option price. 
Simply choose the corresponding model and specify the number of layers for the generated binomial tree as follows:

```py
from raop.pricing_models import Binomial

bino_price = call_euro.compute_price(Binomial, n_layers=20)

print(f"'call_euro' price estimated by Binomial method is: {bino_price}")
```

The script above will return the following result:

```
'call_euro' price estimated by Binomial method is: 2.168536860102567
```

#### c. With ``MonteCarlo`` model

To use the Monte-Carlo method for option price computation, you will need first to define a stochastic process
to model the evolution of the underlying asset's price over time. Different processes are already implemented in
``raop.stochastic_processes``.

<div align="center">
  <img src="outputs/tests_outputs/test_gbm.png" alt="GBM" width="400">
</div>

In the following example, we instantiate a Geometric Brownian Motion
``gbm`` (see the [Documentation](#documentation) of the StochasticProcess class to access 
the available models and their abbreviations). It is also necessary to enter the desired 
number of simulated  processes as well as the number of time steps for the processes' discretization
(the time step will be $\Delta t = \frac{T}{n_t}$ with $T$ the time to maturity).

```py
from raop.stochastic_processes import StochasticProcess
from raop.pricing_models import MonteCarlo


gbm = StochasticProcess(x0=call_euro.s, model_name="gbm", mu=call_euro.r, sigma=call_euro.sigma)

mc_price = call_euro.compute_price(MonteCarlo, stochastic_process=gbm, n_processes=10000, n_t=50)

print(f"'call_euro' price estimated by Monte-Carlo method is: {mc_price}")
```

The script above returned the following result (note that, as long as the Monte-Carlo method is stochastic,
this result can vary from an evaluation to another and depending on chosen the number of simulations):

```
'call_euro' price estimated by Monte-Carlo method is: 2.176008642538358
```

### 3. Option's greeks computation


## 📖 Documentation <a id="documentation"></a>

Detailed package documentation can be found at ???




[//]: # (https://github.com/banesullivan/README/blob/main/README.md?plain=1)



