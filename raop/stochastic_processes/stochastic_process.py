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
    "cir": ["theta", "sigma", "mu"],  # Mean-Reverting Processes: Cox-Ingersoll-Ross process
    "heston": ["mu", "kappa", "theta", "xi"],  # Heston Model
    "vg": ["theta", "sigma", "nu"],  # Variance-Gamma (VG) Model
}


class StochasticProcess:
    """
    Main class to model a stochastic process (for simulating underlying asset's price evolution through time).

    Attributes:
         x0 (float): initial value of the random variable (classically underlying asset's price at t=0).
         model_name (str): name of the chosen stochastic process to make evolve the random variable. See `raop.stochastic_processes.process_update` for further details on available update models. Possibilities are:

                    "gbm"  # Geometric Brownian Motion
                    "abm"  # Arithmetic Brownian Motion
                    "merton_jd"  # Merton Jump Diffusion process
                    "orn_uhl"  # Mean-Reverting Processes: Ornstein-Uhlenbeck process
                    "cir"  # Mean-Reverting Processes: Cox-Ingersoll-Ross process
                    "heston"  # Heston Model
                    "vg"  # Variance-Gamma (VG) Model
         nu0 (float): initial value of a second random variable on which could depend the first random variable (classically volatility). Default value is np.NaN.
         **kwargs: potentially needed key-words arguments for process update computation, depending on chosen model. See `raop.stochastic_processes.process_update` for further details on update models arguments. Mandatory arguments for each model are:

                    "gbm" -> "mu", "sigma"
                    "abm" -> "mu", "sigma"
                    "merton_jd" -> "mu", "sigma", "p"
                    "orn_uhl" -> "theta", "sigma", "mu"
                    "cir" -> "theta", "sigma", "mu"
                    "heston" -> "mu", "kappa", "theta", "xi"
                    "vg" -> "theta", "sigma", "nu"
         model_params (namedtuple): contains all the parameters previously given as kwargs that are necessary to update the random variables with the chosen model. Automatically created when instantiating a `StochasticProcess`.

    Methods:
        **compute_xt** (`t`, `n_t`, `n_var`): return `n_var` simulated random variables values at time `t` (starting with initial value `x0`) and all computed values between 0 and t with time step $$\Delta t = \\frac{t}{n_t}$$

        **plot** (`t`, `n_t`, `n_var`, `save_path`): plot `n_var` simulated random variables values between 0 and `t` (starting at `x0`) with `n_t` steps. If `save_path` is None, display the plot. Else save it at the given path.
    """
    def __init__(self, x0: float, model_name: str, nu0: float = np.NaN, **kwargs):
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
            for arg in process_update_models[model_name]:
                if arg not in kwargs.keys():
                    in_model_args = False
        if not in_model_args:
            error_msg = f"To instantiate a '{model_name}' StochasticProcess, " \
                        f"the following additional arguments must be specified:\n" \
                        f"{process_update_models[model_name]}"
            log.error(error_msg)
            raise ValueError(error_msg)

        if model_name == "heston" and nu0 == np.NaN:
            error_msg = f"To instantiate a '{model_name}' StochasticProcess 'nu0' must be specified"
            log.error(error_msg)
            raise ValueError(error_msg)

        self.x0 = x0
        self.nu0 = nu0
        self.model_name = model_name
        Params = namedtuple("Params", kwargs.keys())
        self.model_params = Params(*kwargs.values())

    def compute_xt(self, t: float, n_t: int, n_var: int = 1) -> Tuple[Union[float, np.ndarray], np.ndarray]:
        """
        Simulate `n_var` random variables updated at each time step with `self.model_name` stochastic process
        over a duration `t` discretized into `n_t` steps, starting with value `self.x0`.

        Args:
            t (float): stochastic process duration.
            n_t (int): number of time steps to subdivide duration `t`.
            n_var (int): number of simulated stochastic processes.

        Returns:
            Tuple[Union[float, np.ndarray], np.ndarray]: random variables values at time `t` and a np.ndarray with all random variable values between 0 and t (included).
        """
        p = self.model_params
        xt = self.x0
        nut = self.nu0
        x_0_to_t = [[xt]]
        if n_var > 1:
            xt = np.ones(n_var) * xt
            nut = np.ones(n_var) * nut
            x_0_to_t = [xt.tolist()]
        dt = t / n_t
        process_update_func = getattr(process_update, self.model_name)
        for k in range(n_t):
            if self.model_name == "heston":
                xt, nut = process_update_func(s_t=xt, nu_t=nut, dt=dt, **p._asdict())
            else:
                xt = process_update_func(s_t=xt, dt=dt, **p._asdict())
            x_0_to_t.append(xt.tolist())
        return xt, np.array(x_0_to_t)

    def plot(self, t: float, n_t: int, n_var: int = 1, save_path: str = None):
        """
        Plot `n_var` random variables updated at each time step with `self.model_name` stochastic process
        over a duration `t` discretized into `n_t` steps, starting with value `self.x0`.

        x-axis is for time and y-axis is for random variables values.

        Args:
            t (float): stochastic process duration.
            n_t (int): number of time steps to subdivide duration `t`.
            n_var (int): number of simulated stochastic processes.
            save_path (str): path where to save the plot. Default is None: in that case plot will be displayed. If not, it will be saved at the given file path.
        """
        processes = self.compute_xt(t, n_t, n_var)[1].T
        dt = t/n_t
        time = np.arange(0, t + dt, dt)

        plt.style.use("ggplot")
        fig, ax = plt.subplots()
        ax.set_title(f"{self.model_name} stochastic process(es) from t=0 to t={t}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Stochastic process(es) values")
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

    # option_euro = Option(
    #     name="european",
    #     option_type="put",
    #     underlying_price=30,
    #     strike_price=120,
    #     time_to_maturity=10,
    #     risk_free_rate=0.05,
    #     volatility=0.5
    # )

    option_euro = Option(
        name="european",
        option_type="call",
        underlying_price=36,
        strike_price=40,
        time_to_maturity=1,
        risk_free_rate=0.06,
        volatility=0.2
    )

    mu = option_euro.r
    sigma = option_euro.sigma
    s0 = option_euro.s

    GBM_euro = StochasticProcess(x0=s0, model_name="gbm", mu=mu, sigma=sigma)
    ABM_euro = StochasticProcess(x0=s0, model_name="abm", mu=mu, sigma=sigma)
    OrnUhl_euro = StochasticProcess(x0=s0, model_name="orn_uhl", theta=0.05, mu=mu, sigma=sigma)
    MertonJD_euro = StochasticProcess(x0=s0, model_name="merton_jd", p=0.001, mu=mu, sigma=sigma)
    CIR_euro = StochasticProcess(x0=s0, model_name="cir", theta=0.05, mu=mu, sigma=sigma)
    Heston_euro = StochasticProcess(x0=s0, model_name="heston", theta=0.05, mu=mu, nu0=sigma, kappa=0.3, ksi=0.005)
    VG_euro = StochasticProcess(x0=s0, model_name="vg", theta=0.05, nu=0.01, sigma=sigma)

    from raop.utils import logger
    logger.setLevel("ERROR")
    GBM_euro.plot(t=1, n_t=1000, n_var=100, save_path="../../outputs/tests_outputs/test_gbm")
    # ABM_euro.plot(t=5, n_t=1000, n_var=100)
    # OrnUhl_euro.plot(t=5, n_t=1000, n_var=100)
    # MertonJD_euro.plot(t=1, n_t=1000, n_var=20)
    # CIR_euro.plot(t=5, n_t=1000, n_var=500)
    # Heston_euro.plot(t=5, n_t=1000, n_var=100)
    # VG_euro.plot(t=5, n_t=1000, n_var=1000)

