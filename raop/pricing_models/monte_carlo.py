import numpy as np

from raop.pricing_models.pricing_model import OptionPricingModel
from raop.stochastic_processes.stochastic_process import StochasticProcess
from raop.utils.log import logger
from raop.options import payoffs
from raop.pricing_models.core.mc_method import monte_carlo


class MonteCarlo(OptionPricingModel):
    def __init__(self, option,
                 stochastic_process: StochasticProcess, n_processes: int, n_t: int,
                 basis_functions: str = None, number_of_functions: int = None):
        # todo: add class/functions descriptions
        super().__init__(option)
        self.name = "Monte-Carlo"
        self.stochastic_process = stochastic_process
        self.n_proc = n_processes
        self.n_t = n_t

        self.basis_functions = "laguerre"
        self.number_of_functions = 20
        if basis_functions is not None:
            self.basis_functions = basis_functions
        if number_of_functions is not None:
            self.number_of_functions = number_of_functions

    def compute_price(self):
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
