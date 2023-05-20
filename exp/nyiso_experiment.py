import numpy as np
from numba import jit
import jump_diffusion_process as jdp
import markov_decision_process as mdp
import util
import datetime
from timeit import default_timer as timer

STAGES = 24
SETTLEMENTS = 12

# number of scenarios
W_TRAIN = 1000
W_TEST = 1000
# size of scenario lattice
k = 50

# Storage unit efficiencies
# like Cheng, Powell (https://ieeexplore.ieee.org/abstract/document/7558191)
EFF_CHARGE = .9
EFF_DISCHARGE = .9

R_MIN = 0
R_MAX = 60 
R_STEP = 1
R = np.arange(R_MIN, R_MAX + R_STEP, R_STEP)

# Sample sizes chosen after sampling experiment.
B_NUM = 30
P_NUM = 1


def main():
    res_dir = 'results_nyiso_experiment/'

    # Load price data, fit Poisson spike process.
    prices = util.load_nyiso_rtd()

    P_MIN, P_MAX = np.quantile(prices, [.025, .975])
    print(f"95% of prices lie within the interval [{P_MIN}, {P_MAX}].")
    print(f"We use this as lower and upper limits for our price and bid space samples.")

    # last price at time 22:50, s.t. test_paths starts at 22:55
    p0 = prices[-15]
    p0_ts = prices.index[-15]

    print("Calibrating Poisson Spike Process...")
    jdp_ = jdp.PoissonSpikeProcess(scale=30).fit(prices)
    # extract seasonality for optimization horizon
    S = jdp_.get_S_before_sim(p0_ts, (STAGES + 1) * SETTLEMENTS, freq="5T")
    jdp_params = jdp_._params

    _, B, P = mdp.create_sampled_spaces(P_MIN, P_MAX, B_NUM, P_NUM)

    print(f"Running NYISO real-time market experiment using BADP-lattice with T={STAGES}, |B|={B_NUM}, |P|={P_NUM}...")
    start = timer()
    print(f"Start of execution: {datetime.datetime.now()}")

    V, pol = mdp.badp_lattice(P, B, S, jdp_params, W=W_TRAIN, k=k)

    np.save(res_dir + f"V_STAGES_{STAGES}_W_{W_TRAIN}_k{k}_R_{len(R)}_P_{P_NUM}_B_{B_NUM}", V)
    np.save(res_dir + f"pol_STAGES_{STAGES}_W_{W_TRAIN}_k{k}_R_{len(R)}_P_{P_NUM}_B_{B_NUM}", pol)

    elapsed = (timer() - start)
    print(f"Time elapsed: {datetime.timedelta(seconds=elapsed)}")


if __name__ == '__main__':
    main()