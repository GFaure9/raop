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


def cir(
        s_t: Union[float, np.ndarray],
        dt: float,
        theta: float,
        sigma: float,
        mu: float = 0,
) -> np.ndarray:
    # In this model, the underlying asset's price is assumed to follow the SDE:
    #                           dS = theta * (mu - S_t) * dt + sigma * sqrt(S_t) * dW_t
    #                               (with W_t a Wiener process)
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
        ksi: float,
) -> np.ndarray:
    # In this model, the underlying asset's price is assumed to follow the SDE:
    #                           dS = mu * S_t * dt + sqrt(nu_t) * S_t * dW_t^{S}
    #                           dnu = kappa * (theta - nu_t) * dt + ksi * sqrt(nu_t) * dW_t^{nu}
    #                               (with W_t a Wiener process)
    # Cf. Wiki https://en.wikipedia.org/wiki/Heston_model
    if type(s_t) != np.ndarray:
        s_t = np.array([s_t])
    if type(nu_t) != np.ndarray:
        nu_t = np.array([nu_t])
    assert len(s_t) == len(nu_t)
    dw_nu = np.random.normal(0, np.sqrt(dt), len(nu_t))
    dnu = kappa * (theta - nu_t) * dt + ksi * np.sqrt(nu_t) * dw_nu
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
    # In this model, the underlying asset's price is assumed to follow the SDE:
    #                           dS = theta * dG + sigma * sqrt(dG) * Z
    #                               ( with dG ~ Gamma(dt/nu, nu) and Z ~ Normal(0, 1) )
    # see Wiki https://en.wikipedia.org/wiki/Variance_gamma_process (Simulation)
    if type(s_t) != np.ndarray:
        s_t = np.array([s_t])
    dg = np.random.gamma(dt/nu, nu, len(s_t))
    z = np.random.normal(0, 1, len(s_t))
    ds = theta * dg + sigma * np.sqrt(dg) * z
    return s_t + ds
