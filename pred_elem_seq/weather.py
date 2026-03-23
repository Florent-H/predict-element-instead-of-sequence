import pandas as pd


def get_weather(weather_path, sim_time_idx, columns=None):

    # # if the sim_time_idx is in 15 minutes time steps
    # if len(sim_time_idx) in (35040, 35040 + 24 * 4):
    #     weather = pd.read_csv(
    #         weather_path.parents[0] / (str(weather_path.stem) + "_QH.csv"),
    #         index_col=0,
    #         parse_dates=True,
    #     )
    # # use hourly weather file for all other time step resolutions (i.e., H, D, M)
    # else:
    # get weather dataframe
    skip_rows = 8
    weather_all = pd.read_csv(
        weather_path,
        delimiter=",",
        skiprows=skip_rows,
        header=None,
        index_col=False,
    )
    # assign all columns of weather_all dataframe
    all_columns = [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "data_source_unct",
        "temp_air",
        "temp_dew",
        "relative_humidity",
        "atmospheric_pressure",
        "etr",
        "etrn",
        "ghi_infrared",
        "ghi",
        "dni",
        "dhi",
        "global_hor_illum",
        "direct_normal_illum",
        "diffuse_horizontal_illum",
        "zenith_luminance",
        "wind_direction",
        "wind_speed",
        "total_sky_cover",
        "opaque_sky_cover",
        "visibility",
        "ceiling_height",
        "present_weather_observation",
        "present_weather_codes",
        "precipitable_water",
        "aerosol_optical_depth",
        "snow_depth",
        "days_since_last_snowfall",
        "albedo",
        "liquid_precipitation_depth",
        "liquid_precipitation_quantity",
    ]
    weather_all.columns = all_columns

    if columns:
        # filter requested weather columns
        weather = weather_all[columns]
    else:
        weather = weather_all[
            [
                "dni",
                "dhi",
                "temp_air",
                "temp_dew",
                "relative_humidity",
                "atmospheric_pressure",
                "wind_speed",
                "wind_direction",
            ]
        ]
        weather.columns = [
            "DNI",
            "DHI",
            "TOut",
            "dewPoint",
            "relHum",
            "press",
            "uWind",
            "dirWind",
        ]

    # cast to float32
    weather = weather.astype("float32")

    # apply hourly time index to weather dataframe
    weather.index = sim_time_idx

    return weather
