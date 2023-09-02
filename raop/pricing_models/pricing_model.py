from collections import namedtuple


class OptionPricingModel:
    def __init__(self, option: dict, **kwargs):
        Option = namedtuple("Option", [key for key in option.keys()])
        self.option = Option(*option.values())

    def compute_price(self):
        return

    def compute_greeks(self):
        return
