# todo: add standard error in MonteCarlo

# todo: create a function StochasticProcess.from_option(Option, model_name)

# todo: test uploading as a package in TestPyPi (with setup)

# todo: add class/functions descriptions (DOCUMENTATION)

# todo: generate automatic doc html formatter using adapted tool

# todo: maybe add 'q' to the arguments, the dividend yield of the underlying

# todo: complete/adapt the framework for asian options

# todo: scripts to estimate drift? implied volatility?

# todo: should "payoffs" be kept in methods?

# todo: if possible accelerate even more Binomial or MonteCarlo by
#  using C# or C++ code (main problem: storing all variables)

# todo: maybe add other sub-models in Binomial (than CRR): Jarrow-Rudd? Leisen-Reimer?

# todo: in _p() of Binomial, add q in some way if possible: (r-q) instead of r

# todo: in BlackScholes _theta() method check if there is an error... (with multiple tests)

# possible graphs:
#       - greeks vs underlying asset's price
#       - option price vs strike
#       - option price vs maturity
#       - payoff vs underlying asset's price
#       - implied volatility surface (x=money-ness(S/K), y=time to maturity, z=implied volatility)
