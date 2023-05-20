import numpy as np
from numba import jit
import jump_diffusion_process as jdp
import markov_decision_process as mdp
import util
import datetime
import time
from timeit import default_timer as timer
import logging

B_NUMS = [2, 5, 10, 15, 30, 45, 60]
TEST_B_FIX_P = 1

P_NUMS = [1, 5, 10, 15, 30, 45, 60]
TEST_P_FIX_B = 10

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


def main():
    res_dir = 'results_sampling_experiment/'
    log_name = res_dir + f'sampling_experiment_{time.strftime("%Y%m%d-%H%M%S")}.log'
    logging.basicConfig(encoding='utf-8',
                        level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[
                            logging.FileHandler(log_name),
                            logging.StreamHandler()])

    # Load price data, fit Poisson spike process.
    prices = util.load_nyiso_rtd()

    P_MIN, P_MAX = np.quantile(prices, [.025, .975])
    logging.info(f"95% of prices lie within the interval [{P_MIN}, {P_MAX}].")
    logging.info(f"We use this as lower and upper limits for our price and bid space samples.")

    # last price at time 22:50, s.t. test_paths starts at 22:55
    p0 = prices[-15]
    p0_ts = prices.index[-15]

    logging.info("Calibrating Poisson Spike Process...")
    jdp_ = jdp.PoissonSpikeProcess(scale=30).fit(prices)
    S = jdp_.get_S_before_sim(p0_ts, (STAGES + 1) * SETTLEMENTS + 1, freq="5T")
    jdp_params = jdp_._params

    test_paths = jdp_.simulate(W_TEST, (STAGES + 1) * SETTLEMENTS, p0, freq="5T", last_ts=p0_ts)
    test_paths.to_csv(res_dir + "sampling_experiment_test_paths.csv")

    # |B|-effect
    logging.info("#" * 5 + " TESTING EFFECT OF |B| under perfect foresight " + "#" * 5)
    for (ind, b_num) in enumerate(B_NUMS):
        _, B, _ = mdp.create_sampled_spaces(P_MIN, P_MAX, b_num, 1)
        
        logging.info("#" * 20)
        logging.info(f"Running combination {ind + 1}/{len(B_NUMS)} with {b_num=}.")

        # Solve determinstic dynamic program
        start = timer()
        logging.info(f"Start of execution: {datetime.datetime.now()}")

        pol = mdp.perfect_foresight(test_paths, B)

        elapsed = (timer() - start)
        logging.info(f"Time elapsed: {datetime.timedelta(seconds=elapsed)}")
        
        np.save(res_dir + f"perfect_foresight_len_B_{b_num}.npy", pol)

    # |P|-effect
    logging.info("#" * 5 + " TESTING EFFECT OF |P| " + "#" * 5)

    for (ind, p_num) in enumerate(P_NUMS):
        b_num = TEST_P_FIX_B
        _, B, P = mdp.create_sampled_spaces(P_MIN, P_MAX, b_num, p_num)
        
        logging.info("#" * 20)
        logging.info(f"Running combination {ind + 1}/{len(P_NUMS)} with (b_num, p_num)=({b_num}, {p_num}).")

        state_space_size = len(R) * len(P) * len(B)
        logging.info(f"Size of state space={state_space_size}, size of action space={len(B)}")

        # Solve MDP
        start = timer()
        logging.info(f"Start of execution: {datetime.datetime.now()}")

        V, pol = mdp.badp_lattice(P, B, S, jdp_params, W=W_TRAIN, k=k)

        elapsed = (timer() - start)
        logging.info(f"Time elapsed: {datetime.timedelta(seconds=elapsed)}")
        
        np.save(res_dir + f"V_STAGES_{STAGES}_W_{W_TRAIN}_k{k}_R_{len(R)}_P_{p_num}_B_{b_num}", V)
        np.save(res_dir + f"pol_STAGES_{STAGES}_W_{W_TRAIN}_k{k}_R_{len(R)}_P_{p_num}_B_{b_num}", pol)

    # |B|-effect
    logging.info("#" * 5 + " TESTING EFFECT OF |B| " + "#" * 5)

    for (ind, b_num) in enumerate(B_NUMS):
        _, B, P = mdp.create_sampled_spaces(P_MIN, P_MAX, b_num, 1)
        
        logging.info("#" * 20)
        logging.info(f"Running combination {ind + 1}/{len(B_NUMS)} with (b_num, p_num)=({b_num}, 1).")

        state_space_size = len(R) * len(P) * len(B)
        logging.info(f"Size of state space={state_space_size}, size of action space={len(B)}")

        # Solve MDP
        start = timer()
        logging.info(f"Start of execution: {datetime.datetime.now()}")

        V, pol = mdp.badp_lattice(P, B, S, jdp_params, W=W_TRAIN, k=k)

        elapsed = (timer() - start)
        logging.info(f"Time elapsed: {datetime.timedelta(seconds=elapsed)}")
        
        np.save(res_dir + f"V_STAGES_{STAGES}_W_{W_TRAIN}_k{k}_R_{len(R)}_P_{1}_B_{b_num}", V)
        np.save(res_dir + f"pol_STAGES_{STAGES}_W_{W_TRAIN}_k{k}_R_{len(R)}_P_{1}_B_{b_num}", pol)

if __name__ == '__main__':
    main()