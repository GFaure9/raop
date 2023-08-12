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


def abm():
    pass

# todo: define different stochastic processes computation strategies
