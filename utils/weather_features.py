import meteostat as ms
from datetime import date
import pandas as pd
import numpy as np
import holidays

def get_weather_data(start_date: date, end_date: date, locations: int = 4) -> pd.DataFrame:
    """
    Fetches weather data for a given location and date range.

    Parameters:
    start_date (date): The start date for the weather data.
    end_date (date): The end date for the weather data.
    locations (int): The number of nearby weather stations to consider.

    Returns:
    pd.DataFrame: A DataFrame containing the weather data.
    """
    # Specify locations and time range
    CITIES = {
        "Frankfurt": ms.Point(50.1155, 8.6842, 113),
        "Berlin": ms.Point(52.5200, 13.4050, 34),
        "Munich": ms.Point(48.1351, 11.5820, 519),
        "Cologne": ms.Point(50.9375, 6.9603, 37),
        "Hamburg": ms.Point(53.5511, 9.9937, 8),
    }

    # Fetch daily average, minimum and maximum temperature for each city
    city_weather = {}
    for city_name, point in list(CITIES.items())[:locations]:
        stations = ms.stations.nearby(point, limit=3)
        ts = ms.daily(stations, start_date, end_date)
        city_df = ms.interpolate(ts, point).fetch()
        city_weather[city_name] = city_df[[ms.Parameter.TEMP, ms.Parameter.TMIN, ms.Parameter.TMAX, ms.Parameter.WSPD, ms.Parameter.TSUN, ms.Parameter.CLDC]].rename(
            columns={
                ms.Parameter.TEMP: "Temp",
                ms.Parameter.TMIN: "Min Temp",
                ms.Parameter.TMAX: "Max Temp",
                ms.Parameter.WSPD: "Wind Speed",
                ms.Parameter.TSUN: "Sunshine Duration",
                ms.Parameter.CLDC: "Cloud Cover",
            }
        )

    # Build per-metric dataframes across all cities
    temp_df = pd.DataFrame({city: metrics["Temp"] for city, metrics in city_weather.items()})
    tmin_df = pd.DataFrame({city: metrics["Min Temp"] for city, metrics in city_weather.items()})
    tmax_df = pd.DataFrame({city: metrics["Max Temp"] for city, metrics in city_weather.items()})
    wspd_df = pd.DataFrame({city: metrics["Wind Speed"] for city, metrics in city_weather.items()})
    sshn_df = pd.DataFrame({city: metrics["Sunshine Duration"] for city, metrics in city_weather.items()})
    cldc_df = pd.DataFrame({city: metrics["Cloud Cover"] for city, metrics in city_weather.items()})
    
    weather_df = pd.DataFrame(
        {
            "Temp": temp_df.mean(axis=1),               # °C
            "Min Temp": tmin_df.mean(axis=1),           # °C
            "Max Temp": tmax_df.mean(axis=1),           # °C
            "Wind Speed": wspd_df.mean(axis=1),         # km/h
            "Sunshine Duration": sshn_df.mean(axis=1),  # minutes
            "Cloud Cover": cldc_df.mean(axis=1),
        }
    )

    datetime_index = pd.DatetimeIndex(pd.to_datetime(weather_df.index))
    weather_df.index = datetime_index

    weather_df["dayofweek"] = datetime_index.dayofweek
    weather_df["month"] = datetime_index.month
    weather_df["is_weekend"] = (weather_df["dayofweek"] >= 5).astype(int)

    de_holidays = holidays.Germany(prov="NW")
    weather_df["is_holiday"] = datetime_index.normalize().isin(de_holidays).astype(int)

    weather_df["dow_sin"] = np.sin(2 * np.pi * weather_df["dayofweek"] / 7)
    weather_df["dow_cos"] = np.cos(2 * np.pi * weather_df["dayofweek"] / 7)
    weather_df["month_sin"] = np.sin(2 * np.pi * weather_df["month"] / 12)
    weather_df["month_cos"] = np.cos(2 * np.pi * weather_df["month"] / 12)
    
    weather_df = weather_df.drop(columns=["dayofweek", "month"])

    return weather_df


if __name__ == "__main__":
    weather_df = get_weather_data(date(2026, 2, 25), date(2026, 2, 28), locations=3)
    print(weather_df.head())