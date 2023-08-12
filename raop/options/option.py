import inspect
import numpy as np
import pandas as pd

from typing import Type, Union, Tuple, List
from itertools import product
from raop.pricing_models.pricing_model import OptionPricingModel
from raop.utils.log import logger

from raop.options import payoffs


def check_option_pricing_kwargs(model: Type[OptionPricingModel]):
    init_arguments = inspect.signature(model.__init__).parameters
    init_arguments_names = [param for param in init_arguments]
    init_arguments_names.remove("self")
    init_arguments_names.remove("option")
    if len(init_arguments_names) > 0:
        log.warning(f"The model {model}"
                    f" requires to pass the following additional arguments to the function:"
                    f"\n{init_arguments_names}")


class Option:
    """
    Main class to model an option among the following types:
    - european
    - american
    ...
    """
    args_to_attr = {
        "name": "name",
        "option_type": "option_type",
        "underlying_price": "s",
        "strike_price": "k",
        "risk_free_rate": "r",
        "time_to_maturity": "time_to_maturity",
        "volatility": "sigma",
    }

    def __init__(
            self,
            name: str,
            option_type: str,
            underlying_price: float,
            strike_price: float,
            risk_free_rate: float,
            time_to_maturity: float,
            volatility: float,
    ):
        # todo: maybe add 'q' to the arguments, the dividend yield of the underlying
        # todo: complete/adapt the framework for asian options
        # todo: scripts to estimate drift ? implied volatility ?
        self.name = name

        if option_type not in ["call", "put"]:
            error_msg = f"'option_type' must be either 'put' or 'call'."  # please correct your input
            log.error(error_msg)
            raise ValueError(error_msg)
        self.option_type = option_type  # "call" or "put"

        self.s = underlying_price
        self.k = strike_price
        self.r = risk_free_rate
        self.time_to_maturity = time_to_maturity
        log.warning("'time_to_maturity' must be provided in years.")

        self.sigma = volatility
        log.info(f"Option instance created with attributes:\n{self.to_dict()}\n")

    def to_dict(self) -> dict:
        # ################# Old Method ################
        # return {
        #     "name": self.name,
        #     "option_type": self.option_type,
        #     "s": self.s,
        #     "k": self.k,
        #     "r": self.r,
        #     "time_to_maturity": self.time_to_maturity,
        #     "sigma": self.sigma,
        # }
        # #############################################
        return vars(self)

    def payoff(self, **kwargs) -> float:
        log.info(f"Started computing {self.name} option's pay-off...")

        payoffs_func = getattr(payoffs, self.name)
        pay_off = payoffs_func(**self.to_dict(), **kwargs)

        log.info(f"Finished computing option's pay-off![Pay-Off = {pay_off}]\n")
        return pay_off

    def compute_price(self, model: Type[OptionPricingModel], **kwargs) -> float:
        # Check if necessary additional key-word arguments to be passed in the function
        check_option_pricing_kwargs(model)
        return model(self.to_dict(), **kwargs).compute_price()

    def compute_greeks(self, model: Type[OptionPricingModel], **kwargs) -> dict:
        check_option_pricing_kwargs(model)
        return model(self.to_dict(), **kwargs).compute_greeks()

    def sensitivity(
            self,
            output: str,
            variable: Union[str, List[str]],
            variations: Union[Tuple, List[Tuple]],
            model: Type[OptionPricingModel],
            num: int = 20,
            **kwargs,
    ) -> np.ndarray:

        # Adjust type of arguments if necessary
        if type(variable) == str:
            variable = [variable]
        if type(variations) == tuple:
            variations = [variations]

        # Construct vectors of values to compute output with
        values = {}
        for var, bnd in zip(variable, variations):
            attribute = self.args_to_attr[var]
            nominal_value = getattr(self, attribute)
            b_low, b_upp = bnd
            start = nominal_value * (1 + b_low/100)
            stop = nominal_value * (1 + b_upp/100)
            values[attribute] = np.linspace(start, stop, num)
        combinations = list(product(*values.values()))

        # Compute output
        res = []
        if output == "option_price":
            for vec in combinations:
                for att, val in zip(values.keys(), vec):
                    setattr(self, att, val)
                res.append(self.compute_price(model, **kwargs))

        elif output in ["delta", "gamma", "vega", "theta", "rho"]:
            for vec in combinations:
                for att, val in zip(values.keys(), vec):
                    setattr(self, att, val)
                res.append(self.compute_greeks(model, **kwargs)[output])

        # Construct DataFrame
        data = np.column_stack((np.array(combinations), np.array(res)))
        columns_names = variable + [output]
        df = pd.DataFrame(data=data, columns=columns_names)

        return df

# possible graphs:
#       - greeks vs underlying asset's price
#       - option price vs strike
#       - option price vs maturity
#       - payoff vs underlying asset's price
#       - implied volatility surface (x=money-ness(S/K), y=time to maturity, z=implied volatility)


log = logger

if __name__ == "__main__":
    # log.setLevel("ERROR")

    option_euro = Option(
        name="european",
        option_type="put",
        underlying_price=36,
        strike_price=40,
        time_to_maturity=1,
        risk_free_rate=0.06,
        volatility=0.2
    )
    # option_euro = Option(
    #     name="european",
    #     option_type="put",
    #     underlying_price=30,
    #     strike_price=50,
    #     time_to_maturity=7,
    #     risk_free_rate=0.05,
    #     volatility=0.5
    # )
    option_amer = Option(
        name="american",
        option_type="put",
        underlying_price=36,
        strike_price=40,
        time_to_maturity=1,
        risk_free_rate=0.06,
        volatility=0.2
    )
    # option_amer = Option(
    #     name="american",
    #     option_type="put",
    #     underlying_price=30,
    #     strike_price=50,
    #     time_to_maturity=7,
    #     risk_free_rate=0.05,
    #     volatility=0.5
    # )

    option_euro.payoff()
    option_amer.payoff()

    from raop.pricing_models.black_scholes import BlackScholes
    from raop.pricing_models.binomial import Binomial
    from raop.pricing_models.monte_carlo import MonteCarlo
    from raop.stochastic_processes.stochastic_process import StochasticProcess

    # option_euro.compute_greeks(BlackScholes)
    # option_euro.compute_greeks(Binomial, n_layers=18)
    # option_amer.compute_greeks(Binomial, n_layers=18)

    GBM_euro = StochasticProcess(x0=option_euro.s, model_name="gbm", mu=option_euro.r, sigma=option_euro.sigma)
    option_euro.compute_price(MonteCarlo, stochastic_process=GBM_euro, n_processes=10000, n_t=100)
    # option_euro.compute_price(BlackScholes)
    # option_euro.compute_price(Binomial, n_layers=15)

    GBM_amer = StochasticProcess(x0=option_amer.s, model_name="gbm", mu=option_amer.r, sigma=option_amer.sigma)
    option_amer.compute_price(MonteCarlo, stochastic_process=GBM_amer, n_processes=10000, n_t=50,
                              basis_functions="hermite", number_of_functions=20)
    # option_amer.compute_price(Binomial, n_layers=20)

# todo: add standard error in MonteCarlo
# todo: create a function StochasticProcess.from_option(Option, model_name)
# todo: test uploading as a package in TestPyPi (with setup)
# todo: github repo + first push
# todo: add class/functions descriptions

