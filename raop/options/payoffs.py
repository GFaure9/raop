import numpy as np

from typing import Union


def european(
        s: Union[float, np.ndarray],
        k: float,
        option_type: str,
        s_0_to_t: np.ndarray = None,
        **kwargs
) -> Union[float, np.ndarray]:
    """
    Implement computation of instantaneous european option's payoff: $$Call_t(S_t, K_t) = max(S_t - K_t, 0)$$ $$Put_t(S_t, K_t) = max(K_t - S_t, 0)$$

    Args:
        s (Union[float, np.ndarray]): underlying price of the option. Also accepts an array of underlying prices.
        k (float): strike price of the option.
        option_type: type of the option. Either "put" or "call".
        s_0_to_t (np.ndarray): underlying prices from time 0 to t. Default value is None. If not, `s` will be set equal to array's last row (prices at time t).
        **kwargs:

    Returns:
        Union[float, np.ndarray]: if `s` is a float, the function returns a float corresponding to option's payoff. If `s` is an array, returns an array with option's payoffs corresponding to each price.
    """

    if s_0_to_t is not None:
        # this is to homogenize usage of payoff functions in OptionPricingModel subclasses
        s = s_0_to_t[-1, :]

    zero = 0
    if type(s) == np.ndarray:
        zero = np.zeros_like(s)

    pay_offs = None
    if option_type == "call":
        pay_offs = np.amax([zero, s - k], axis=0)  # N.B: np.amax() returns a element-wise maximums
    elif option_type == "put":
        pay_offs = np.amax([zero, k - s], axis=0)

    return pay_offs


def american(s_0_to_t: np.ndarray = None, **kwargs) -> Union[float, np.ndarray]:
    """
    Implement computation of instantaneous american option's payoff.

    Args:
        s_0_to_t (np.ndarray): underlying prices from time 0 to t. Default value is None. If not, `s` will be set equal to the array.
        **kwargs: same arguments as those of `european` function.

    Returns:
        Union[float, np.ndarray]: if `s` is a float, the function returns a float corresponding to option's payoff. If `s` is an array, returns an array with option's payoffs corresponding to each price.
    """
    if s_0_to_t is not None:
        # used for instance for Monte-Carlo and binomial pricing methods
        kwargs["s"] = s_0_to_t
        pay_offs = european(**kwargs)
        return pay_offs
    else:
        # if we are already at time t the payoff is the same as for a european option
        return european(**kwargs)


def asian():
    pass
