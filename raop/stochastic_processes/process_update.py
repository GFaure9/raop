import numpy as np

from typing import Union, Tuple


def gbm(s_t: Union[float, np.ndarray], dt: float, mu: float, sigma: float) -> np.ndarray:
    """
    Compute `s_t` at time step `t + dt` considering that it follows a Geometric Brownian Motion
    characterized by the SDE:
    $$dS = \\mu S_t dt + \sigma S_t dW_t$$

    Where:
    $$dW_t \sim \mathcal{N}(0, dt)$$

    Args:
        s_t (Union[float, np.ndarray]): random variable(s) value(s) at current time.
        dt (float): time step increment.
        mu (float): drift of the process.
        sigma (float): volatility of the process.

    Returns:
        np.ndarray: value(s) of random variable(s) at next time (current time incremented by `dt`).
    """
    if type(s_t) != np.ndarray:
        s_t = np.array([s_t])
    dw = np.random.normal(0, np.sqrt(dt), len(s_t))
    ds = mu * s_t * dt + sigma * s_t * dw
    return s_t + ds


def abm(s_t: Union[float, np.ndarray], dt: float, mu: float, sigma: float) -> np.ndarray:
    """
    Compute `s_t` at time step `t + dt` considering that it follows a Arithmetic Brownian Motion
    characterized by the SDE:
    $$dS = \\mu dt + \sigma dW_t$$

    Where:
    $$dW_t \sim \mathcal{N}(0, dt)$$

    Args:
        s_t (Union[float, np.ndarray]): random variable(s) value(s) at current time.
        dt (float): time step increment.
        mu (float): drift of the process.
        sigma (float): volatility of the process.

    Returns:
        np.ndarray: value(s) of random variable(s) at next time (current time incremented by `dt`).
    """
    if type(s_t) != np.ndarray:
        s_t = np.array([s_t])
    dw = np.random.normal(0, np.sqrt(dt), len(s_t))
    ds = mu * dt + sigma * dw
    return s_t + ds


def orn_uhl(
        s_t: Union[float, np.ndarray],
        dt: float,
        theta: float,
        sigma: float,
        mu: float = 0,
) -> np.ndarray:
    """
    Compute `s_t` at time step `t + dt` considering that it follows an Ornstein–Uhlenbeck process (a Mean-Reverting process)
    characterized by the SDE:
    $$dS = \\theta (\\mu - S_t) dt + \sigma dW_t$$

    Where:
    $$dW_t \sim \mathcal{N}(0, dt)$$

    Args:
        s_t (Union[float, np.ndarray]): random variable(s) value(s) at current time.
        dt (float): time step increment.
        mu (float): drift of the process.
        sigma (float): volatility of the process.
        theta (float): a positive parameter.

    Returns:
        np.ndarray: value(s) of random variable(s) at next time (current time incremented by `dt`).
    """
    if type(s_t) != np.ndarray:
        s_t = np.array([s_t])
    dw = np.random.normal(0, np.sqrt(dt), len(s_t))
    ds = theta * (mu - s_t) * dt + sigma * dw
    return s_t + ds


def merton_jd(
        s_t: Union[float, np.ndarray],
        dt: float,
        mu: float,
        sigma: float,
        p: float,
) -> np.ndarray:
    """
    Compute `s_t` at time step `t + dt` considering that it follows a Merton Jump Diffusion process
    characterized by the SDE:
    $$dS = \\mu S_t dt + \\sigma S_t dW_t + S_t dJ_t$$

    Where:
    $$dW_t \sim \mathcal{N}(0, dt)$$

    and for sufficiently small `dt`
    $$dJ_t = Z_{N_t}dN_t \quad \\text{with} \quad Z_{N_t} \sim \mathcal{N}(0, 1) \quad \\text{and} \quad dN_t \sim Bernoulli(p \\times dt)$$

    For further details and theory about this process, see for example
    [N. Privault's course on _Stochastic Calculus for Jump Processes_ at Nanyang University](https://personal.ntu.edu.sg/nprivault/MA5182/stochastic-calculus-jump-processes.pdf).

    Args:
        s_t (Union[float, np.ndarray]): random variable(s) value(s) at current time.
        dt (float): time step increment.
        mu (float): drift of the process.
        sigma (float): volatility of the process.
        p (float): positive parameter (Bernoulli law's probability intervening in process' SDE is `p * dt`).

    Returns:
        np.ndarray: value(s) of random variable(s) at next time (current time incremented by `dt`).
    """
    if type(s_t) != np.ndarray:
        s_t = np.array([s_t])
    size = len(s_t)
    dw = np.random.normal(0, np.sqrt(dt), size)
    dn = np.random.binomial(1, p * dt, size)
    z = np.random.normal(size)
    ds = mu * s_t * dt + sigma * s_t * dw + s_t * z * dn
    return s_t + ds


def cir(
        s_t: Union[float, np.ndarray],
        dt: float,
        theta: float,
        sigma: float,
        mu: float = 0,
) -> np.ndarray:
    """
    Compute `s_t` at time step `t + dt` considering that it follows a Cox-Ingersoll-Ross process (a Mean-Reverting process)
    characterized by the SDE:
    $$dS = \\theta (\\mu - S_t) dt + \\sigma \sqrt{S_t} * dW_t$$

    Where:
    $$dW_t \sim \mathcal{N}(0, dt)$$

    Args:
        s_t (Union[float, np.ndarray]): random variable(s) value(s) at current time.
        dt (float): time step increment.
        mu (float): drift of the process.
        sigma (float): volatility of the process.
        theta (float): a positive parameter.

    Returns:
        np.ndarray: value(s) of random variable(s) at next time (current time incremented by `dt`).
    """
    if type(s_t) != np.ndarray:
        s_t = np.array([s_t])
    dw = np.random.normal(0, np.sqrt(dt), len(s_t))
    ds = theta * (mu - s_t) * dt + sigma * np.sqrt(s_t) * dw
    return s_t + ds


def heston(
        s_t: Union[float, np.ndarray],
        nu_t: Union[float, np.ndarray],
        dt: float,
        mu: float,
        kappa: float,
        theta: float,
        xi: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute `s_t` at time step `t + dt` considering that it follows a Heston model
    characterized by the following SDEs:
    $$dS = \\mu S_t dt + \sqrt{\\nu_t} S_t dW_t^{S}$$
    $$d\\nu = \\kappa (\\theta - \\nu_t) dt + \\xi \sqrt{\\nu_t} dW_t^{\\nu}$$

    With:
    $$dW_t^{S} \sim \mathcal{N}(0, dt)$$
    $$dW_t^{\\nu} \sim \mathcal{N}(0, dt)$$

    For further details, see for example [the Heston model Wikipedia page](https://en.wikipedia.org/wiki/Heston_model).

    Args:
        s_t (Union[float, np.ndarray]): random variable(s) value(s) at current time (typically underlying price).
        nu_t (Union[float, np.ndarray]): the instantaneous variance(s) of the previous random variable(s) at current time (typically volatility).
        dt (float): time step increment.
        mu (float): drift of the process.
        kappa (float): the rate at which `nu_t` reverts to `theta`.
        theta (float): the long variance, or long-run average variance of the price.
        xi (float): the volatility of the volatility ("vol of vol").

    Returns:
        Tuple[np.ndarray, np.ndarray]: value(s) of random variable(s) and its(their) variance(s) at next time (current time incremented by `dt`).
    """
    if type(s_t) != np.ndarray:
        s_t = np.array([s_t])
    if type(nu_t) != np.ndarray:
        nu_t = np.array([nu_t])
    assert len(s_t) == len(nu_t)
    dw_nu = np.random.normal(0, np.sqrt(dt), len(nu_t))
    dnu = kappa * (theta - nu_t) * dt + xi * np.sqrt(nu_t) * dw_nu
    dw_s = np.random.normal(0, np.sqrt(dt), len(s_t))
    ds = mu * s_t * dt + np.sqrt(nu_t) * s_t * dw_s
    return s_t + ds, nu_t + dnu


def vg(
        s_t: Union[float, np.ndarray],
        dt: float,
        theta: float,
        sigma: float,
        nu: float,
) -> np.ndarray:
    """
    Compute `s_t` at time step `t + dt` considering that it follows a Variance-Gamma model
    characterized by the SDE:
    $$dS = \\theta dG + \\sigma sqrt{dG} Z$$

    With:
    $$dG \sim \Gamma(\\frac{dt}{\\nu}, \\nu)$$
    $$Z \sim \mathcal{N}(0, 1)$$

    For further details, see for example [the Variance-Gamma model Wikipedia page](https://en.wikipedia.org/wiki/Variance_gamma_process).

    Args:
        s_t (Union[float, np.ndarray]): random variable(s) value(s) at current time.
        dt (float): time step increment.
        theta (float): positive parameter.
        sigma (float): positive parameter.
        nu (float): positive parameter.

    Returns:
        np.ndarray: value(s) of random variable(s) at next time (current time incremented by `dt`).
    """
    if type(s_t) != np.ndarray:
        s_t = np.array([s_t])
    dg = np.random.gamma(dt/nu, nu, len(s_t))
    z = np.random.normal(0, 1, len(s_t))
    ds = theta * dg + sigma * np.sqrt(dg) * z
    return s_t + ds
