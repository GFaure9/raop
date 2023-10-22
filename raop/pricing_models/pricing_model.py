from collections import namedtuple


class OptionPricingModel:
    # remark: it could be possible to make this an abstract class using abc package (might be better...)
    """
    Generic class for defining an option pricing model.

    Attributes:
        option (collections.namedtuple): its keys are:

                "name"
                "option_type"
                "s"
                "k"
                "r"
                "time_to_maturity"
                "sigma"

    Methods:
        **compute_price**: return an estimation of the price of the option defined in `option`.

        **compute_greeks**: return an estimation of the greeks of the option defined in `option`.
    """
    def __init__(self, option: dict, **kwargs):
        # using namedtuple is a way to be able to access to dict keys' values as class attributes
        # this is to facilitate its usage/readability later in the code
        Option = namedtuple("Option", [key for key in option.keys()])
        self.option = Option(*option.values())

    def compute_price(self):
        """
        Compute `self.option` 's price.
        """
        return

    def compute_greeks(self):
        """
        Compute `self.option` 's greeks.
        """
        return
