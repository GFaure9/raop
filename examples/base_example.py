from raop.options import Option
from raop.pricing_models import BlackScholes, Binomial, MonteCarlo
from raop.stochastic_processes import StochasticProcess
from raop.graphs import Graph

# ####################### Creating a European call option ########################
call_euro = Option(
  name="european",
  option_type="call",
  underlying_price=36,
  strike_price=40,
  time_to_maturity=1,
  risk_free_rate=0.06,
  volatility=0.2
)

# ############## Computing the option's price with different models ##############
bs_price = call_euro.compute_price(BlackScholes)
bino_price = call_euro.compute_price(Binomial, n_layers=20)
gbm = StochasticProcess(x0=call_euro.s, model_name="gbm", mu=call_euro.r, sigma=call_euro.sigma)
mc_price = call_euro.compute_price(MonteCarlo, stochastic_process=gbm, n_processes=10000, n_t=50)

# ########### Computing greeks of the option with Black-Scholes model ############
bs_greeks = call_euro.compute_greeks(BlackScholes)

# ######################## Sensitivity analyses and plots ########################
# Gamma VS Underlying Price
df_curve = call_euro.sensitivity(
    output="gamma",
    variable="underlying_price",
    variations=(-50, 50),  # +/- 50% variations of option's underlying price
    num=100,
    model=BlackScholes,  # using Black-Scholes model
)

# Plot from the computed data
graph_curve = Graph(df_curve)
graph_curve.plot_curve()

# Option's Price VS Volatilty and Strike Price
df_surf = call_euro.sensitivity(
    output="option_price",
    variable=["volatility", "strike_price"],
    variations=[(-50, 1000), (-50, 20)],
    num=20,
    model=Binomial,  # using Binomial model with 10 layers
    n_layers=10,
)

# Plot from the computed data
graph_surf = Graph(df_surf)
graph_surf.plot_surface()

# ############################## Printing results ################################
print("The Option instance 'call_euro' has the following attributes:\n", call_euro.to_dict())
print(f"'call_euro' price estimated by Black-Scholes method is: {bs_price}")
print(f"'call_euro' price estimated by Binomial method is: {bino_price}")
print(f"'call_euro' price estimated by Monte-Carlo method is: {mc_price}")
print(f"'call_euro' greeks estimated by Black-Scholes method are:\n{bs_greeks}")
print(f"Option's Price VS Volatilty and Strike Price pd.DataFrame:\n{df_surf}")
