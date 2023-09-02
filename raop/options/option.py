import inspect
import numpy as np
import pandas as pd

from typing import Type, Union, Tuple, List
from itertools import product
from raop.pricing_models.pricing_model import OptionPricingModel
from raop.utils import logger

# from raop.options import payoffs


class Option:
    """
    Main class to model a stock option among the following types:

    - european
    - american

    Attributes:
        name (str): name of the option (its category): "european" or "american".
        option_type (str): type of the option. Either "put" or "call".
        underlying_price (float): price of the underlying asset at t=0 (when option is priced).
        strike_price (float): strike price of the option.
        risk_free_rate (float): risk-free rate of the option.
        time_to_maturity (float): time when the option expires (expressed in years).
        volatility (float): volatility of the underlying asset price at t=0.

    Methods:
        **compute_price** (`model`): return the price of the option estimated with a
        `model` selected among subclasses of `OptionPricingModel`.

        **compute_greeks** (`model`): return the greeks of the option estimated with a `model`
        selected among subclasses of `OptionPricingModel`.

        **sensitivity** (`output`, `variable`, `variations`, `model`, `num`): return a
        `pd.DataFrame` containing the estimated values of `output` (a string) when varying the
        values of the variables listed in `variable` by a certain percentage around the option's
        baseline values. This percentages of variations to be applied are listed in a list of tuples
        `variations`. `num` is the number of values subdividing the defined ranges of variations.
        `model` is the subclass of `OptionPricingModel` that will be used to compute the output's
        values.

        **to_dict** : return a dictionary containing the attributes' values of the
        defined `Option` object.
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
        """
        Returns:
            dict: a dictionary with the values of the different attributes.
        """
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

    # def payoff(self, **kwargs) -> float:
    #     log.info(f"Started computing {self.name} option's pay-off...")
    #
    #     payoffs_func = getattr(payoffs, self.name)
    #     pay_off = payoffs_func(**self.to_dict(), **kwargs)
    #
    #     log.info(f"Finished computing option's pay-off![Pay-Off = {pay_off}]\n")
    #     return pay_off

    @staticmethod
    def check_option_pricing_kwargs(model: Type[OptionPricingModel]):
        """
        Display log with potential additional key-word arguments that need to be specified
        when using a certain `model`.
        Args:
            model (Type[OptionPricingModel]): subclass of `OptionPricingModel`.

        Returns:
            None: will display a warning message if extra arguments are required and log level is set to DEBUG.
        """
        init_arguments = inspect.signature(model.__init__).parameters
        init_arguments_names = [param for param in init_arguments]
        init_arguments_names.remove("self")
        init_arguments_names.remove("option")
        if len(init_arguments_names) > 0:
            log.warning(f"The model {model}"
                        f" requires to pass the following additional arguments to the function:"
                        f"\n{init_arguments_names}")

    def compute_price(self, model: Type[OptionPricingModel], **kwargs) -> float:
        """
        Estimate the price of the option using the `model` approach.
        Args:
            model (Type[OptionPricingModel]): subclass of `OptionPricingModel` that will be used to compute option's price.
            **kwargs: potentially required extra arguments depending on the chosen `OptionPricingModel` subclass.

        Returns:
            float: computed option's price.
        """
        self.check_option_pricing_kwargs(model)
        return model(self.to_dict(), **kwargs).compute_price()

    def compute_greeks(self, model: Type[OptionPricingModel], **kwargs) -> dict:
        """
        Estimate the greeks of the option using the `model` approach.
        Args:
            model (Type[OptionPricingModel]): subclass of `OptionPricingModel` that will be used to compute option's greeks.
            **kwargs: potentially required extra arguments depending on the chosen `OptionPricingModel` subclass.

        Returns:
            dict: computed option's greeks.
        """
        self.check_option_pricing_kwargs(model)
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
        """
        Perform sensitivity analysis of variable(s) in `variable` on `output` (option's price or specific
        greeks...).
        Args:
            output (str): name of the output. Possible outputs are:
                            "option_price",
                            "delta",
                            "gamma",
                            "theta",
                            "vega",
                            "rho",
            variable (Union[str, List[str]]): name or list of names of varying parameters for the sensitivity study.
            variations (Union[Tuple, List[Tuple]]): i-th tuple contains the +/- percentages of variation to apply to the i-th parameter in `variable`.
            model (Type[OptionPricingModel]): subclass of `OptionPricingModel` that will be used to compute the output.
            num (int): number of subdivisions of the parameters' variations ranges.
            **kwargs: potentially required extra arguments depending on the chosen `OptionPricingModel` subclass.

        Returns:
            pd.DataFrame: pandas DataFrame object containing parameters and corresponding output values.

        For instance, the following script will compute option's price for all combinations volatilities
        and strike prices taken respectively in the ranges [vol - 50%, vol + 1000%] and
        [strike price - 50%, strike price + 20%].

        >>> call_euro = Option(name="european", option_type="call", underlying_price=36, strike_price=40, time_to_maturity=1, risk_free_rate=0.06, volatility=0.2)
        >>> df_sens = call_euro.sensitivity(output="option_price", variable=["volatility", "strike_price"], variations=[(-50, 1000), (-50, 20)], num=20, model=Binomial, n_layers=10)
        >>> print(df_sens)
                              volatility  strike_price  option_price
                        0           0.1     20.000000      0.000000
                        1           0.1     21.473684      0.000000
                        2           0.1     22.947368      0.000000
                        3           0.1     24.421053      0.000000
                        4           0.1     25.894737      0.000000
                        ..          ...           ...           ...
                        395         2.2     42.105263     27.032943
                        396         2.2     43.578947     28.115194
                        397         2.2     45.052632     29.197446
                        398         2.2     46.526316     30.279697
                        399         2.2     48.000000     31.361948
                        [400 rows x 3 columns]
        """

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


log = logger
