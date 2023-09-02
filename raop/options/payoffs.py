import numpy as np

from typing import Union


def european(
        s: Union[float, np.ndarray],
        k: float,
        option_type: str,
        s_0_to_t: np.ndarray = None,
        **kwargs
) -> Union[float, np.ndarray]:

    if s_0_to_t is not None:
        s = s_0_to_t[-1, :]

    zero = 0
    if type(s) == np.ndarray:
        zero = np.zeros_like(s)

    pay_offs = None
    if option_type == "call":
        pay_offs = np.amax([zero, s - k], axis=0)
    elif option_type == "put":
        pay_offs = np.amax([zero, k - s], axis=0)

    return pay_offs


def american(s_0_to_t: np.ndarray = None, **kwargs):
    if s_0_to_t is not None:
        kwargs["s"] = s_0_to_t
        pay_offs = european(**kwargs)
        return pay_offs
    else:
        return european(**kwargs)


def asian():
    pass
