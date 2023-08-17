import numpy as np

from typing import Union


def gbm(s_t: Union[float, np.ndarray], dt: float, mu: float, sigma: float) -> np.ndarray:
    # In this model, the underlying asset's price is assumed to follow the SDE:
    #                           dS = mu * S_t * dt + sigma * S_t * dW_t
    #                               (with W_t a Wiener process)
    #          (i.e. with dW_t a normal random variables with expected value zero and variance dt)
    # In the last equation, mu represents the drift and sigma the volatility.
    if type(s_t) != np.ndarray:
        s_t = np.array([s_t])
    dw = np.random.normal(0, np.sqrt(dt), len(s_t))
    ds = mu * s_t * dt + sigma * s_t * dw
    return s_t + ds


def abm(s_t: Union[float, np.ndarray], dt: float, mu: float, sigma: float) -> np.ndarray:
    # In this model, the underlying asset's price is assumed to follow the SDE:
    #                           dS = mu * dt + sigma * dW_t
    #                               (with W_t a Wiener process)
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
    # In this model, the underlying asset's price is assumed to follow the SDE:
    #                           dS = theta * (mu - S_t) * dt + sigma * dW_t
    #                               (with W_t a Wiener process)
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
    # In this model, the underlying asset's price is assumed to follow the SDE:
    #                           dS = mu * S_t * dt + sigma * S_t * dW_t + S_t * dJ_t
    #                      (with W_t a Wiener process and J_t a Compound Poisson Process)
    # According to N. Privault's "Stochastic Calculus for Jump Processes":
    #   dJ_t = Z_Nt * dNt with Z_Nt ~ N(1) and for sufficiently small dt dNt ~ Bernoulli(p * dt)
    if type(s_t) != np.ndarray:
        s_t = np.array([s_t])
    size = len(s_t)
    dw = np.random.normal(0, np.sqrt(dt), size)
    dn = np.random.binomial(1, p * dt, size)
    z = np.random.normal(size)
    ds = mu * s_t * dt + sigma * s_t * dw + s_t * z * dn
    return s_t + ds


# todo: define different stochastic processes computation strategies
