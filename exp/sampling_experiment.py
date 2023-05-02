import numpy as np
from numba import jit
import jump_diffusion_process as jdp
import markov_decision_process as mdp
import util
import datetime
import time
from timeit import default_timer as timer
import logging

B_NUMS = [10]
P_NUMS = [5, 10, 15, 30, 45, 60, 120]  # 120

B_P_COMBOS = [(b, p) for b in B_NUMS for p in P_NUMS]

STAGES = 24
SETTLEMENTS = 12

# number of scenarios
W_TRAIN = 1000
# size of scenario lattice
k = 50

# Storage unit efficiencies
# like Cheng, Powell (https://ieeexplore.ieee.org/abstract/document/7558191)
EFF_CHARGE = .9
EFF_DISCHARGE = .9

# Inventory discretization (sample rate not changed)
R_MIN = 0
R_MAX = 60  # 6 (500 KWh), 60 (5 MWh), 72 (6 MWh)
R_STEP = 1
R = np.arange(R_MIN, R_MAX + R_STEP, R_STEP)


def main():
    log_name = f'experiment_{time.strftime("%Y%m%d-%H%M%S")}.log'
    logging.basicConfig(encoding='utf-8',
                        level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[
                            logging.FileHandler(log_name),
                            logging.StreamHandler()])

    logging.info(f"Executing B_P_COMBOS={B_P_COMBOS}.")

    # Load price data, fit Poisson spike process.
    prices = util.load_nyiso_rtd()

    P_MIN, P_MAX = np.quantile(prices, [.025, .975])
    logging.info(f"95% of prices lie within the interval [{P_MIN}, {P_MAX}].")
    logging.info(f"We use this as lower and upper limits for our price and bid space samples.")

    # last price at time 22:50, s.t. test_paths starts at 22:55
    p0 = prices[-15]
    p0_ts = prices.index[-15]

    jdp_ = jdp.PoissonSpikeProcess(scale=30, thresh=50).fit(prices)
    S = jdp_.get_S_before_sim(p0_ts, (STAGES + 1) * SETTLEMENTS + 1, freq="5T")
    jdp_params = jdp_._params
    
    for (ind, (b_num, p_num)) in enumerate(B_P_COMBOS):
        _, B, P = mdp.create_sampled_spaces(P_MIN, P_MAX, b_num, p_num)
        
        logging.info("#" * 20)
        logging.info(f"Running combination {ind + 1}/{len(B_P_COMBOS)} with (b_num, p_num)=({b_num}, {p_num}).")

        state_space_size = len(R) * len(P) * len(B)
        logging.info(f"Size of state space={state_space_size}")

        # Solve MDP
        start = timer()
        logging.info(f"Start of execution: {datetime.datetime.now()}")

        V, bid = mdp.lattice_badp(P, B, S, jdp_params, W=W_TRAIN, k=k)

        elapsed = (timer() - start)
        logging.info(f"Time elapsed: {datetime.timedelta(seconds=elapsed)}")
        
        np.save(f"V_STAGES_{STAGES}_W_{W_TRAIN}_k{k}_R_{len(R)}_P_{p_num}_B_{b_num}", V)
        np.save(f"bid_STAGES_{STAGES}_W_{W_TRAIN}_k{k}_R_{len(R)}_P_{p_num}_B_{b_num}", bid)

if __name__ == '__main__':
    main()