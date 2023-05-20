import pandas as pd
import numpy as np
from numba import jit

def load_rt_csv(year):
    prices = pd.read_csv(f"../dat/OASIS_Real_Time_Dispatch_Zonal_LBMP_{year}.csv")

    prices["time"] = pd.to_datetime(prices.loc[:, "RTD End Time Stamp"])

    prices = pd.Series(data=prices["RTD Zonal LBMP"].values, index=prices["time"])
    prices.index.name = "time"

    # deals with daylight savings time
    prices = prices.tz_localize("EST", ambiguous="infer")

    # Sometimes, the last price is corrected before the next interval starts.
    # We need to clean this up.
    ns5min = 5 * 60 * 1000000000   # 5 minutes in nanoseconds
    rounded = pd.to_datetime(((prices.index.astype(np.int64) // ns5min) * ns5min))
    index = prices.index.copy()
    dropthis = []

    for ind, val in enumerate(rounded):
        if ind + 1 < len(rounded) and rounded[ind + 1] == val:
            ts = index[ind]
            nxt_ts = index[ind + 1]
            
            prices[ts] = prices[nxt_ts]
            dropthis += [nxt_ts]
    
    prices = prices.drop(dropthis)
    return prices

def load_nyiso_rtd():
    prices_2019 = load_rt_csv(2019)
    prices_2020 = load_rt_csv(2020)
    prices_2021 = load_rt_csv(2021)
    prices_2022 = load_rt_csv(2022)

    return pd.concat([prices_2019, prices_2020, prices_2021, prices_2022])


@jit(nopython=True)
def _find_bid_ind(bid, B):
    # naive implementation, but what the hell...
    ind = 0
    for i in range(len(B)):
        if np.array_equal(bid, B[i]):
            return ind
        ind += 1
    return -1