import numpy as np
import matplotlib.pyplot as plt

from typing import Union, Tuple
from collections import namedtuple
from raop.stochastic_processes import process_update
from raop.utils import logger


process_update_models = {
    "gbm": ["mu", "sigma"],  # Geometric Brownian Motion
    "abm": ["mu", "sigma"],  # Arithmetic Brownian Motion
    "merton_jd": ["mu", "sigma", "p"],  # Merton Jump Diffusion process
    "orn_uhl": ["theta", "sigma", "mu"],  # Mean-Reverting Processes: Ornstein-Uhlenbeck process
    "cir": [],  # Mean-Reverting Processes: Cox-Ingersoll-Ross process
    "heston": [],  # Heston Model
    "vg": [],  # Variance-Gamma (VG) Model
    "levy": [],  # Levy Processes
    "fbm": [],  # Fractional Brownian Motion
}


class StochasticProcess:
    def __init__(self, x0: float, model_name: str, **kwargs):
        if model_name not in process_update_models.keys():
            error_msg = f"'model_name' must be chosen among the following names:\n" \
                        f"{list(process_update_models.keys())}"
            log.error(error_msg)
            raise ValueError(error_msg)

        n_kwargs = len(kwargs)
        in_model_args = True
        if n_kwargs == 0:
            in_model_args = False
        if n_kwargs > 0:
            for key in kwargs.keys():
                if key not in process_update_models[model_name]:
                    in_model_args = False
        # todo: include the case where a model arg is missing
        if not in_model_args:
            error_msg = f"To instantiate a '{model_name}' StochasticProcess, " \
                        f"the following additional arguments must be specified:\n" \
                        f"{process_update_models[model_name]}"
            log.error(error_msg)
            raise ValueError(error_msg)

        self.x0 = x0
        self.model_name = model_name
        Params = namedtuple("Params", kwargs.keys())
        self.model_params = Params(*kwargs.values())

    def compute_xt(self, t: float, n_t: int, n_var: int = 1) -> Tuple[Union[float, np.ndarray], np.ndarray]:
        p = self.model_params
        xt = self.x0
        x_0_to_t = [[xt]]
        if n_var > 1:
            xt = np.ones(n_var) * xt
            x_0_to_t = [xt.tolist()]
        dt = t / n_t
        process_update_func = getattr(process_update, self.model_name)
        for k in range(n_t):
            xt = process_update_func(s_t=xt, dt=dt, **p._asdict())
            x_0_to_t.append(xt.tolist())
        return xt, np.array(x_0_to_t)

    def plot(self, t: float, n_t: int, n_var: int = 1, save_path: str = None):
        processes = self.compute_xt(t, n_t, n_var)[1].T
        dt = t/n_t
        time = np.arange(0, t + dt, dt)

        fig, ax = plt.subplots()
        plt.style.use("ggplot")
        ax.set_title(f"{self.model_name} stochastic process(es) from t=0 to t={t}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Stochastic process(es)")
        ax.grid(True, alpha=0.5, ls="--")
        for process in processes:
            ax.plot(time, process, linestyle="-")
        ax.legend()
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()


log = logger


if __name__ == "__main__":
    from raop.options.option import Option
    option_euro = Option(
        name="european",
        option_type="put",
        underlying_price=30,
        strike_price=120,
        time_to_maturity=10,
        risk_free_rate=0.05,
        volatility=0.5
    )

    GBM_euro = StochasticProcess(x0=option_euro.s, model_name="gbm", mu=option_euro.r, sigma=option_euro.sigma)
    ABM_euro = StochasticProcess(x0=option_euro.s, model_name="abm", mu=option_euro.r, sigma=option_euro.sigma)
    OrnUhl_euro = StochasticProcess(x0=option_euro.s, model_name="orn_uhl",
                                    theta=0.05, mu=option_euro.r, sigma=option_euro.sigma)
    MertonJD_euro = StochasticProcess(x0=option_euro.s, model_name="merton_jd",
                                      p=0.001, mu=option_euro.r, sigma=option_euro.sigma)

    from raop.utils import logger
    logger.setLevel("ERROR")
    # GBM_euro.plot(t=5, n_t=1000, n_var=100)
    # ABM_euro.plot(t=5, n_t=1000, n_var=100)
    # OrnUhl_euro.plot(t=5, n_t=1000, n_var=100)
    # MertonJD_euro.plot(t=5, n_t=1000, n_var=100)

