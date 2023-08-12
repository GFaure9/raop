import unittest
import logging

import pandas as pd
import numpy as np

from raop.options import Option
from raop.pricing_models import Binomial

from raop.utils import find_project_root, days_between_dates, logger

logger.setLevel(logging.ERROR)


class TestBinomial(unittest.TestCase):
	# todo: optimize the functions/class (to limit repetitions from function to another)
	root_path = find_project_root()
	data_option_prices_eu = pd.read_csv(f"{root_path}/data/tests_datasets/option_prices_eu.csv")
	data_greeks_eu = pd.read_csv(f"{root_path}/data/tests_datasets/option_greeks_eu.csv")
	data_option_prices_usa = pd.read_csv(f"{root_path}/data/tests_datasets/option_prices_usa.csv")
	data_greeks_usa = pd.read_csv(f"{root_path}/data/tests_datasets/option_greeks_usa.csv")

	def test_european_option_prices(self, precision: float = 1):
		df_prices = self.data_option_prices_eu
		df_greeks = self.data_greeks_eu

		test_ids = df_prices.index[df_prices["settlement"] != 0.0]

		for i in test_ids:
			row = df_prices.iloc[i]

			option_type = "call" if row["call_put"] == "C" else "put"
			underlying_price = row["futures_close"]
			strike_price = row["strike"]
			volatility = df_greeks.iloc[i]["iv_interpolated"]
			time_to_maturity = days_between_dates(row["date"], row["options_expiration_date"])/365

			real_price = row["settlement"]

			option = Option(
				name="european",
				option_type=option_type,
				underlying_price=underlying_price,
				strike_price=strike_price,
				volatility=volatility,
				risk_free_rate=0.0,
				time_to_maturity=time_to_maturity,
			)

			computed_price = option.compute_price(Binomial, n_layers=10)

			error = abs(computed_price - real_price)

			# print(f"Results for row {i}:")
			# print("Real Price: ", real_price)
			# print("Computed Price: ", computed_price)
			# print(f"Error: {np.round(error, 2)}")
			# print("\n")

			self.assertLessEqual(error, precision)

	def test_american_option_prices(self, precision: float = 2):
		df_prices = self.data_option_prices_usa
		df_greeks = self.data_greeks_usa

		test_ids = df_prices.index[df_prices["settlement"] != 0.0]

		for i in test_ids:
			row = df_prices.iloc[i]

			option_type = "call" if row["call_put"] == "C" else "put"
			underlying_price = row["futures_close"]
			strike_price = row["strike"]
			volatility = df_greeks.iloc[i]["iv_interpolated"]
			time_to_maturity = days_between_dates(row["date"], row["options_expiration_date"])/365

			real_price = row["settlement"]

			option = Option(
				name="american",
				option_type=option_type,
				underlying_price=underlying_price,
				strike_price=strike_price,
				volatility=volatility,
				risk_free_rate=0.0,
				time_to_maturity=time_to_maturity,
			)

			computed_price = option.compute_price(Binomial, n_layers=10)

			error = abs(computed_price - real_price)

			# print(f"Results for row {i}:")
			# print("Real Price: ", real_price)
			# print("Computed Price: ", computed_price)
			# print(f"Error: {np.round(error, 2)}")
			# print("\n")

			self.assertLessEqual(error, precision)

	def test_european_option_greeks(self, precision: float = 0.5):
		# todo: actual function fails the test (sign) -> maybe find from where it comes from
		#  or find another dataset for tests ?
		df_prices = self.data_option_prices_eu
		df_greeks = self.data_greeks_eu

		test_ids = df_prices.index[df_prices["settlement"] != 0.0]

		greeks = ["delta", "gamma", "vega", "theta"]

		for i in test_ids:
			row = df_greeks.iloc[i]

			call_put = "call_put" if "call_put" in df_greeks.columns else "call/put"
			option_type = "call" if row[call_put] == "C" else "put"
			underlying_price = row["futures_close"]
			strike_price = row["strike"]
			volatility = df_greeks.iloc[i]["iv_interpolated"]
			time_to_maturity = days_between_dates(row["date"], row["options_expiration_date"]) / 365

			real_greeks = row[greeks]

			option = Option(
				name="european",
				option_type=option_type,
				underlying_price=underlying_price,
				strike_price=strike_price,
				volatility=volatility,
				risk_free_rate=0.0,
				time_to_maturity=time_to_maturity,
			)

			computed_greeks = option.compute_greeks(Binomial, n_layers=10)
			computed_greeks["vega"] /= 100
			computed_greeks["theta"] /= 365

			errors_greeks = {
				g: abs(computed_greeks[g] - real_greeks[g]) for g in greeks
			}
			same_sign_greeks = {
				g: (computed_greeks[g] * real_greeks[g] >= 0) or (abs(computed_greeks[g] * real_greeks[g]) <= 1e-6)
				for g in greeks
			}

			print(f"Results for row {i}:")
			for greek in greeks:
				print(f"Real {greek}: ", real_greeks[greek])
				print(f"Computed {greek}: ", computed_greeks[greek])
				print(f"Error: {np.round(errors_greeks[greek], 4)}")
				error = errors_greeks[greek]
				same_sign = same_sign_greeks[greek]
				self.assertTrue(same_sign)
				self.assertLessEqual(error, precision)
			print("\n")

	def test_american_option_greeks(self, precision: float = 0.5):
		# todo: actual function fails the test (sign and precision) -> maybe find from where it comes from
		#  or find another dataset for tests ?
		df_prices = self.data_option_prices_usa
		df_greeks = self.data_greeks_usa

		test_ids = df_prices.index[df_prices["settlement"] != 0.0]

		greeks = ["delta", "gamma", "vega", "theta"]

		for i in test_ids:
			row = df_greeks.iloc[i]

			call_put = "call_put" if "call_put" in df_greeks.columns else "call/put"
			option_type = "call" if row[call_put] == "C" else "put"
			underlying_price = row["futures_close"]
			strike_price = row["strike"]
			volatility = df_greeks.iloc[i]["iv_interpolated"]
			time_to_maturity = days_between_dates(row["date"], row["options_expiration_date"]) / 365

			real_greeks = row[greeks]

			option = Option(
				name="american",
				option_type=option_type,
				underlying_price=underlying_price,
				strike_price=strike_price,
				volatility=volatility,
				risk_free_rate=0.0,
				time_to_maturity=time_to_maturity,
			)

			computed_greeks = option.compute_greeks(Binomial, n_layers=10)
			computed_greeks["vega"] /= 100
			computed_greeks["theta"] /= 365

			errors_greeks = {
				g: abs(computed_greeks[g] - real_greeks[g]) for g in greeks
			}
			same_sign_greeks = {
				g: (computed_greeks[g] * real_greeks[g] >= 0) or (abs(computed_greeks[g] * real_greeks[g]) <= 1e-6)
				for g in greeks
			}

			print(f"Results for row {i}:")
			for greek in greeks:
				print(f"Real {greek}: ", real_greeks[greek])
				print(f"Computed {greek}: ", computed_greeks[greek])
				print(f"Error: {np.round(errors_greeks[greek], 4)}")
				error = errors_greeks[greek]
				same_sign = same_sign_greeks[greek]
				self.assertTrue(same_sign)
				self.assertLessEqual(error, precision)
			print("\n")


if __name__ == "__main__":
	unittest.main()
