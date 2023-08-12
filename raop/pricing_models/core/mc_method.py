import numpy as np
import scipy.special as spe

from raop.stochastic_processes.stochastic_process import StochasticProcess
from raop.options import payoffs
from raop.utils.log import logger

available_functions = [
    "laguerre",
    "hermite",
    "legendre",
    "jacobi",
    "chebyt",  # Chebyshev
    "gegenbauer",
]


# General implementation of the Monte-Carlo method
def monte_carlo(
        sto_pro: StochasticProcess,
        n_t: int,
        n_var: int,
        option: any,
        basis_functions: str = "laguerre",
        number_of_functions: int = 20,
) -> float:

    t = option.time_to_maturity
    underlying_prices_0_to_t = sto_pro.compute_xt(t=t, n_t=n_t, n_var=n_var)[1]
    payoffs_func = getattr(payoffs, option.name)
    pay_offs = payoffs_func(s_0_to_t=underlying_prices_0_to_t, **option._asdict())
    r = option.r

    if option.name in ["european"]:
        # Naive method for Bermudan options
        discounted_payoffs = np.exp(- r * t) * pay_offs
        return np.mean(discounted_payoffs)

    elif option.name in ["american"]:
        # Longstaff-Schwarz American pricing method
        # (for details cf. https://people.maths.ox.ac.uk/gilesm/mc/module_6/american.pdf)
        # Possible basis functions for continuation value LSM:
        # Laguerre, Hermite, Legendre, Chebyshev, Gegenbauer, Jacobi polynomials...

        if basis_functions not in available_functions:
            error_msg = f"'basis_functions' must be in:\n{available_functions}"  # please correct your input
            log.error(error_msg)
            raise ValueError(error_msg)

        log.info(f"Using Longstaff-Schwartz method with"
                 f" {number_of_functions} {basis_functions} basis functions...")

        # Compute continuation values
        k = len(underlying_prices_0_to_t) - 1
        continuation_values = np.empty_like(underlying_prices_0_to_t)
        continuation_values[-1] = pay_offs[-1]

        while k > 0:
            s_previous = underlying_prices_0_to_t[k - 1, :]

            basis_func = getattr(spe, basis_functions)
            basis_fs = np.array([basis_func(i)(s_previous) for i in range(number_of_functions)])

            maxis_ck_hk = np.maximum(continuation_values[k], pay_offs[k])

            # ############# Old Method (details linear regression) ###############
            # b_v_psi = np.zeros(number_of_functions)
            # b_psi_psi = np.empty((number_of_functions, number_of_functions))
            #
            # for row in range(number_of_functions):
            #     v_psi = maxis_ck_hk * basis_fs[row, :]
            #     b_v_psi[row] = np.mean(v_psi, axis=0)
            #
            #     for col in range(number_of_functions):
            #         psi_psi = basis_fs[row, :] * basis_fs[col, :]
            #         b_psi_psi[row, col] = np.mean(psi_psi, axis=0)
            #
            # b_psi_psi_inv = np.linalg.inv(b_psi_psi)
            # beta = np.dot(b_psi_psi_inv, b_v_psi)
            # continuation_values[k - 1] = np.dot(basis_fs.T, beta)
            # ####################################################################

            # Perform linear regression
            x = basis_fs.T
            y = maxis_ck_hk
            beta = np.linalg.lstsq(x, y, rcond=None)[0]

            # Update continuation values
            continuation_values[k - 1] = np.dot(basis_fs.T, beta)

            k -= 1

        # Compute optimal times and payoffs
        where_h_greater_c = np.argwhere(pay_offs[1:, :] > continuation_values[1:, :])
        where_h_greater_c[:, 0] += 1
        n = pay_offs.shape[1]
        taus_payoffs = np.column_stack((np.ones(n) * t, pay_offs[-1]))
        taus = where_h_greater_c[np.argsort(where_h_greater_c[:, 1])]
        taus = taus[np.unique(taus[:, 1], return_index=True)[1]]
        dt = t / n_t
        taus_payoffs[taus[:, 1], 0] = taus[:, 0] * dt
        taus_payoffs[taus[:, 1], 1] = pay_offs[taus[:, 0], taus[:, 1]]

        # Compute discounted payoffs
        discounted_payoffs = np.exp(- r * taus_payoffs[:, 0]) * taus_payoffs[:, 1]

        return np.mean(discounted_payoffs)


log = logger
