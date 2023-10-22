import numpy as np

from raop.pricing_models.pricing_model import OptionPricingModel
from raop.stochastic_processes import StochasticProcess
from raop.utils import logger
from raop.options import payoffs
from raop.pricing_models.core import monte_carlo


class MonteCarlo(OptionPricingModel):
    """
    Subclass of `OptionPricingModel` defining a pricing model that uses the Monte-Carlo method.

    Attributes:
        option (collections.namedtuple): its keys are:

                "name"
                "option_type"
                "s"
                "k"
                "r"
                "time_to_maturity"
                "sigma"
        name (str): name of the pricing model. Default is "Monte-Carlo".
        stochastic_process (raop.stochastic_processes.stochastic_process.StochasticProcess): object defining the stochastic process that models the evolution of underlying asset's price.
        n_processes (int): number of simulated processes in the method.
        basis_functions (str): type of basis functions used when pricing American options using the Longstaff-Schwarz method (LSM). Default value is "laguerre". Possibilities are:

                "laguerre"
                "hermite"
                "legendre"
                "jacobi"
                "chebyt"  # Chebyshev polynomials
                "gegenbauer"
        number_of_functions (int): number of basis functions used to approximate continuation values in LSM. Default value is 20.

    Methods:
        **compute_price**: estimate the price of the option described in `option` using Monte-Carlo method.

        **compute_greeks**: estimate the greeks of the option described in `option` using Monte-Carlo method.
    """
    def __init__(self, option,
                 stochastic_process: StochasticProcess, n_processes: int, n_t: int,
                 basis_functions: str = None, number_of_functions: int = None):
        super().__init__(option)
        self.name = "Monte-Carlo"
        self.stochastic_process = stochastic_process
        self.n_proc = n_processes
        self.n_t = n_t

        self.basis_functions = "laguerre"
        self.number_of_functions = 20  # this could be changed -> investigations should be led to find what is optimal
        if basis_functions is not None:
            self.basis_functions = basis_functions
        if number_of_functions is not None:
            self.number_of_functions = number_of_functions

    def compute_price(self):
        """
        Compute `self.option` 's price using the Monte-Carlo method.

        For European options, it consists in simulating `self.n_processes` possible evolutions
        of underlying price up to maturity date
        (using `raop.stochastic_processes.stochastic_process.StochasticProcess`),
        and averaging the discounted final payoffs of all cases.
        $$V = \\frac{1}{N} \sum_{i=1}^{N}{e^{-r (T - t)} Payoff(S_i)}$$

        For American options, Longstaff-Schwarz method is used: see for example
        [this Oxford University's presentation](https://people.maths.ox.ac.uk/gilesm/mc/module_6/american.pdf)
        for a detailed description of the method.

        Returns:
            float: option's price estimated with the Monte-Carlo method.
        """
        log.info(f"Started computing option's price with {self.name} method...")

        opt = self.option
        sto_pro = self.stochastic_process
        n_t, n_var = self.n_t, self.n_proc
        bas_funcs, n_funcs = self.basis_functions, self.number_of_functions

        price = monte_carlo(sto_pro=sto_pro, n_t=n_t, n_var=n_var, option=opt,
                            basis_functions=bas_funcs, number_of_functions=n_funcs)

        log.info(f"Finished computing option's price! [Price = {price}]\n")
        return price

    def compute_greeks(self) -> dict:
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
        pass

    @property
    def _gamma(self) -> float:
        log.info(f"Computing Gamma...")
        pass

    @property
    def _vega(self) -> float:
        log.info(f"Computing Vega...")
        pass

    @property
    def _theta(self) -> float:
        log.info(f"Computing Theta...")
        pass

    @property
    def _rho(self) -> float:
        log.info(f"Computing Rho...")
        pass


log = logger
