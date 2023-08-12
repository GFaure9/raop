from datetime import datetime


def days_between_dates(date1: str, date2: str):
    # Convert the date strings to datetime objects
    date_format = "%m/%d/%Y"
    date1_obj = datetime.strptime(date1, date_format)
    date2_obj = datetime.strptime(date2, date_format)

    # Calculate the difference between the two dates
    delta = date2_obj - date1_obj

    # Return the number of days as an integer
    return abs(delta.days)


if __name__ == "__main__":
    # Example usage:
    d1 = "10/19/2021"
    d2 = "10/24/2021"
    print(days_between_dates(d1, d2))  # Output: 5
