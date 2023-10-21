import numpy as np

from raop.pricing_models.pricing_model import OptionPricingModel
from raop.utils import logger

from scipy.stats import norm


class BlackScholes(OptionPricingModel):
    """
    Subclass of `OptionPricingModel` defining a pricing model that uses the Black-Scholes model.

    Attributes:
        option (collections.namedtuple): its keys are:

                "name"
                "option_type"
                "s"
                "k"
                "r"
                "time_to_maturity"
                "sigma"
        name (str): name of the pricing model. Default is "Black-Scholes".

    Methods:
        **compute_price**: estimate the price of the option described in `option` using Black-Scholes model.

        **compute_greeks**: estimate the greeks of the option described in `option` using Black-Scholes model.
    """
    def __init__(self, option):
        super().__init__(option)

        if self.option.name != "european":
            error_msg = f"This method is only adapted for European options: " \
                        f"option name must be 'european'."
            log.error(error_msg)
            raise ValueError(error_msg)

        self.name = "Black-Scholes"

    def compute_price(self) -> float:
        """
        Compute `self.option` 's price using Black-Scholes model.

        In this model, prices of European call and put options are given by:
        $$Call(S, K) = N(d_1) S - N(d_2) K e^{-r (T - t)}$$
        $$Put(S, K) = -N(-d_1) S + N(-d_2) K e^{-r (T - t)}$$

        With:
        $$N(x) := \mathbb{P}(X \leq x)  \quad   X \sim \mathcal{N}(0, 1)$$
        $$d_1 = ln(\\frac{S}{K}) + \\frac{r + \\frac{\sigma^2}{2}}{\sigma \sqrt{T - t}}$$
        $$d_2 = d_1 - \sigma \sqrt{T - t}$$

        For further details , see for example
        [the Black-Scholes Model Wikipedia page](https://en.wikipedia.org/wiki/Black-Scholes_model).

        Returns:
            float: option's price estimated with Black-Scholes model.
        """
        log.info(f"Started computing option's price with {self.name} method...")

        opt = self.option
        s, k, r, sigma, dt = opt.s, opt.k, opt.r, opt.sigma, opt.time_to_maturity
        price = None

        if opt.option_type == "call":
            n_d1 = norm.cdf(self._d1)
            n_d2 = norm.cdf(self._d2)
            price = n_d1 * s - n_d2 * k * np.exp(-r * dt)

        elif opt.option_type == "put":
            n_minus_d1 = norm.cdf(-self._d1)
            n_minus_d2 = norm.cdf(-self._d2)
            price = n_minus_d2 * k * np.exp(-r * dt) - n_minus_d1 * s

        log.info(f"Finished computing option's price! [Price = {price}]\n")
        return price

    def compute_greeks(self) -> dict:
        """
        Compute `self.option` 's greeks using Black-Scholes model.

        In this model, greeks of European options are given by:
        $$\delta^{Call} = N(d_1) \quad \\text{and} \quad \delta^{Put} = N(d_1) - 1$$
        $$\gamma = \\frac{N'(d_1)}{S \sigma \sqrt{T - t}}$$
        $$\\nu = S N'(d_1) \sqrt{T - t}$$
        $$\\theta^{Call} = \\theta_0 - r K e^{-r (T - t)} N(d_2) \quad \\text{and} \quad \\theta^{Put} = \\theta_0 + r K e^{-r (T - t)} N(-d_2)$$
        $$\\rho^{Call} = \\rho_0 N(d_2) \quad \\text{and} \quad \\rho^{Put} = -\\rho_0 N(-d_2)$$

        With:
        $$\\theta_0 = \\frac{S N'(d_1) \sigma}{2 \sqrt{T - t}}$$
        $$\\rho_0 = K (T - t) e^{-r (T - t)}$$

        See `BlackScholes.compute_price` method for other variables definitions.

        Returns:
            dict: dictionary containing values of `self.option` 's greeks estimated with Black-Scholes model. Its keys are:

                    "delta"
                    "gamma"
                    "vega"
                    "theta"
                    "rho"
        """
        log.info(f"Started computing option's Greeks with {self.name} method...")

        greeks = {
            "delta": self._delta,
            "gamma": self._gamma,
            "vega": self._vega,
            "theta": self._theta,
            "rho": self._rho,
        }

        log.info(f"Finished computing option's Greeks!\nGreeks: {greeks}\n")
        return greeks

    @property
    def _delta(self) -> float:
        log.info(f"Computing Delta...")

        opt = self.option

        if opt.option_type == "call":
            return norm.cdf(self._d1)
        elif opt.option_type == "put":
            return norm.cdf(self._d1) - 1

    @property
    def _gamma(self) -> float:
        log.info(f"Computing Gamma...")

        opt = self.option
        s, sigma, dt = opt.s, opt.sigma, opt.time_to_maturity

        return self._norm_cdf_derivative(self._d1)/(s * sigma * np.sqrt(dt))

    @property
    def _vega(self) -> float:
        log.info(f"Computing Vega...")

        opt = self.option
        s, dt = opt.s, opt.time_to_maturity

        return s * self._norm_cdf_derivative(self._d1) * np.sqrt(dt)

    @property
    def _theta(self) -> float:
        log.info(f"Computing Theta...")

        opt = self.option
        s, k, r, sigma, dt = opt.s, opt.k, opt.r, opt.sigma, opt.time_to_maturity

        theta0 = - (s * self._norm_cdf_derivative(self._d1) * sigma)/(2 * np.sqrt(dt))

        if opt.option_type == "call":
            return theta0 - r * k * np.exp(-r * dt) * norm.cdf(self._d2)
        elif opt.option_type == "put":
            return theta0 + r * k * np.exp(-r * dt) * norm.cdf(-self._d2)

    @property
    def _rho(self) -> float:
        log.info(f"Computing Rho...")

        opt = self.option
        s, k, r, sigma, dt = opt.s, opt.k, opt.r, opt.sigma, opt.time_to_maturity

        rho0 = k * dt * np.exp(-r * dt)

        if opt.option_type == "call":
            return rho0 * norm.cdf(self._d2)
        elif opt.option_type == "put":
            return -rho0 * norm.cdf(-self._d2)

    @staticmethod
    def _norm_cdf_derivative(x):
        return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

    @property
    def _d1(self) -> float:
        opt = self.option
        s, k, r, sigma, dt = opt.s, opt.k, opt.r, opt.sigma, opt.time_to_maturity

        return (np.log(s/k) + (r + sigma**2 / 2) * dt) / (sigma * np.sqrt(dt))

    @property
    def _d2(self) -> float:
        opt = self.option
        sigma, dt = opt.sigma, opt.time_to_maturity

        return self._d1 - sigma * np.sqrt(dt)


log = logger
