import pandas as pd
import numpy as np

def load_rt_csv(year):
    prices = pd.read_csv(f"../dat/OASIS_Real_Time_Dispatch_Zonal_LBMP_{year}.csv")

    prices["time"] = pd.to_datetime(prices.loc[:, "RTD End Time Stamp"])

    prices = pd.Series(data=prices["RTD Zonal LBMP"].values, index=prices["time"])
    prices.index.name = "time"

    ns5min = 5 * 60 * 1000000000   # 5 minutes in nanoseconds
    prices.index = pd.to_datetime(((prices.index.astype(np.int64) // ns5min) * ns5min))
    return prices


def load_nyiso_rtd():
    prices_2019 = load_rt_csv(2019)
    prices_2020 = load_rt_csv(2020)
    prices_2021 = load_rt_csv(2021)
    prices_2022 = load_rt_csv(2022)

    return pd.concat([prices_2019, prices_2020, prices_2021, prices_2022])