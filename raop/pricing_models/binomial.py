import numpy as np

from raop.pricing_models.pricing_model import OptionPricingModel
from raop.utils.log import logger
from raop.pricing_models.core.binary_tree import create_binary_tree,\
    compute_last_option_prices,\
    compute_previous_option_prices
from raop.options import payoffs

from typing import Union


class Binomial(OptionPricingModel):
    # todo: if possible accelerate even more by using C# or C++ code (main problem: storing all variables)
    def __init__(self, option, n_layers: int):
        super().__init__(option)

        if self.option.name not in ["european", "american"]:
            error_msg = f"This method is only adapted for European and American options: " \
                        f"option name must be either 'european' or 'american'."
            log.error(error_msg)
            raise ValueError(error_msg)

        self.name = "Binomial [Cox-Ross-Rubinstein]"  # todo: maybe add other sub-models: Jarrow-Rudd ? Leisen-Reimer ?
        self.n_layers = n_layers

    def compute_price(
            self,
            other_s: float = None,
            other_sigma: float = None,
            other_r: float = None,
            other_time_to_maturity: float = None,
    ):
        log.info(f"Started computing option's price with {self.name} method...")
        log.info(f"Number of layers of the tree: {self.n_layers}")

        if self.n_layers > 20:
            log.warning("Number of layers < 21 is recommended to ensure reasonable computing times.")

        opt = self.option
        s, k, opt_name, opt_type = opt.s, opt.k, opt.name, opt.option_type
        if other_s is not None:
            s = other_s

        u = self._u(other_sigma, other_time_to_maturity)
        d = self._d(other_sigma, other_time_to_maturity)
        p = self._p(other_sigma, other_r, other_time_to_maturity)
        discount = self._discount(other_r, other_time_to_maturity)

        american = (opt.name == "american")
        if american:
            log.warning("Reminder: you are pricing an American option.")

        # binary_tree, last_layer = create_binary_tree(self.n_layers, u, d, opt.s) # for old method using anytree
        underlying_prices_tree = create_binary_tree(self.n_layers, u, d, s)
        log.info("Computed underlying prices binary tree!")

        # for old method using anytree
        # compute_last_option_prices(last_layer, getattr(payoffs, opt.name), opt.k, opt.option_type)
        payoffs_func = getattr(payoffs, opt_name)
        payoffs_tree = compute_last_option_prices(underlying_prices_tree, payoffs_func, k, opt_type, american)
        log.info("Computed last layer options' prices in binary tree!")

        # compute_previous_option_prices(binary_tree, p, discount)  # for old method using anytree
        price = compute_previous_option_prices(payoffs_tree, p, discount, american)
        log.info("Computed previous options' prices in binary tree!")

        # price = binary_tree.root.option_price  # for old method using anytree

        log.info(f"Finished computing option's price! [Price = {price}]\n")
        return price

    def compute_greeks(self) -> dict:
        log.info(f"Started computing option's Greeks with {self.name} method...")

        price = self.compute_price()

        greeks = {
            "delta": self._delta(price=price),
            "gamma": self._gamma(price=price),
            "vega": self._vega(price=price),
            "theta": self._theta(price=price),
            "rho": self._rho(price=price),
        }

        log.info(f"Finished computing option's Greeks!\nGreeks: {greeks}\n")
        return greeks

    def _delta(self, perturbation: float = 0.01, price: float = None) -> float:
        log.info(f"Computing Delta...")

        opt = self.option
        s = opt.s
        ds = s * perturbation

        new_price = self.compute_price(s + ds)
        if price is None:
            price = self.compute_price()

        delta = (new_price - price)/ds

        return delta

    def _gamma(self, perturbation: float = 0.1,  price: float = None) -> float:
        log.info(f"Computing Gamma...")

        opt = self.option
        s = opt.s
        ds = s * perturbation

        new_price0 = self.compute_price(other_s=s - ds)
        new_price1 = self.compute_price(other_s=s + ds)
        if price is None:
            price = self.compute_price()

        gamma = (new_price1 - 2*price + new_price0) / (ds**2)

        return gamma

    def _vega(self, perturbation: float = 0.05,  price: float = None) -> float:
        log.info(f"Computing Vega...")

        opt = self.option
        sigma = opt.sigma
        dsigma = sigma * perturbation

        new_price = self.compute_price(other_sigma=sigma + dsigma)
        if price is None:
            price = self.compute_price()

        vega = (new_price - price)/dsigma

        return vega

    def _theta(self, perturbation: float = 0.05,  price: float = None) -> float:
        log.info(f"Computing Theta...")

        opt = self.option
        time_to_maturity = opt.time_to_maturity
        dtime = time_to_maturity * perturbation

        new_price = self.compute_price(other_time_to_maturity=time_to_maturity - dtime)
        if price is None:
            price = self.compute_price()

        theta = (new_price - price)/dtime

        return theta

    def _rho(self, perturbation: float = 0.01,  price: float = None) -> Union[float, None]:
        log.info(f"Computing Rho...")

        opt = self.option
        r = opt.r
        if r == 0:
            return None
        dr = r * perturbation

        new_price = self.compute_price(other_r=r + dr)
        if price is None:
            price = self.compute_price()

        rho = (new_price - price)/dr

        return rho

    def _u(self, other_sigma: float = None, other_time_to_maturity: float = None) -> float:
        dt = self._dt(other_time_to_maturity)
        sigma = self.option.sigma
        if other_sigma is not None:
            sigma = other_sigma
        return np.exp(sigma * np.sqrt(dt))

    def _d(self, other_sigma: float = None, other_time_to_maturity: float = None) -> float:
        dt = self._dt(other_time_to_maturity)
        sigma = self.option.sigma
        if other_sigma is not None:
            sigma = other_sigma
        return np.exp(-sigma * np.sqrt(dt))

    def _p(
            self,
            other_sigma: float = None,
            other_r: float = None,
            other_time_to_maturity: float = None
    ) -> float:
        # todo: add q in some way if possible: (r-q) instead of r
        dt = self._dt(other_time_to_maturity)
        u, d = self._u(other_sigma), self._d(other_sigma)
        r = self.option.r
        if other_r is not None:
            r = other_r
        return (np.exp(r * dt) - d)/(u - d)

    def _discount(self, other_r: float = None, other_time_to_maturity: float = None) -> float:
        dt = self._dt(other_time_to_maturity)
        r = self.option.r
        if other_r is not None:
            r = other_r
        return np.exp(-r * dt)

    def _dt(self, other_time_to_maturity: float):
        dt = self.option.time_to_maturity/self.n_layers
        if other_time_to_maturity:
            dt = other_time_to_maturity/self.n_layers
        return dt


log = logger
