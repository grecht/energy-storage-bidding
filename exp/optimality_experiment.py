from numba import jit
from scipy.stats import norm
import numpy as np
import kmeans as km
import datetime
import time
import timeit
from timeit import default_timer as timer
import logging
import util

STAGES = 24
# Jiang et al also ignore efficiency for this benchmark.
EFF_DISCHARGE = 1
EFF_CHARGE = 1

W_TRAIN = 1000
W_TEST = 1000

BIDS = np.linspace(15, 85, num=30)
B = [(buy, sell) for buy in BIDS for sell in BIDS if buy <= sell]             
# Include possibility to place no sell bid,
# but always buy at or below zero.
no_bid = (0, np.inf)
B += [no_bid]
B = np.array(B)


@jit(nopython=True)
def finite_support_badp_lattice(realizations, 
                                pmf,
                                R_MIN=0,
                                R_MAX=18,
                                W=W_TRAIN,
                                k=50,
                                B=B):
    # include dummy final stage.
    T = STAGES + 1
    R = np.arange(R_MIN, R_MAX + 1)
    
    V = np.full((T, len(R), len(B)), np.nan, dtype=np.float64)
    V[-1] = 0  # terminal reward
    pol = np.full((STAGES, len(R), len(B), 2), np.nan, dtype=np.float64)
    
    # At stage t, we decide on bid for stage (t+1).
    for t in range(STAGES - 1, -1, -1):  # (STAGES - 1), ..., 0
        # one period for determining inventory,
        # one for the expected reward.
        p_w = finite_support_price_process(W,
                                           2,
                                           realizations,
                                           pmf,
                                           t0=t+1)
        # create scenario lattice
        labels, centroids = km.kmeans(p_w, k)
        p_lat = centroids.T
        pr = km.det_prob(labels, k)

        for (r_ind, r_t) in enumerate(R):  # resource state
            for (b_tminus_1_ind, (b_buy, b_sell)) in enumerate(B):  # bids made for this stage at (t-1)
                opt_buy = np.nan
                opt_sell = np.nan
                opt_exp_rew = 0
                
                # possible bids for (t+1, t+2], i.e. the actions!
                for (nxt_bid_ind, (buy, sell)) in enumerate(B):
                    exp_rew = 0

                    # iterate over lattices
                    for l in range(k):
                        r_nxt = r_t

                        if t > 0:  # no bidding before t=0
                            if p_lat[0, l] > b_sell and r_nxt - 1 >= R_MIN:
                                r_nxt -= 1
                            elif p_lat[0, l] < b_buy and r_nxt + 1 <= R_MAX:
                                r_nxt += 1
                        trade = 0

                        if p_lat[1, l] > sell and r_nxt - 1 >= R_MIN:
                            trade = 1 * EFF_DISCHARGE
                        elif p_lat[1, l] < buy and r_nxt + 1 <= R_MAX:
                            trade = -1 / EFF_CHARGE
                        elif p_lat[1, l] > sell and r_nxt - 1 < R_MIN:
                            trade = -1  # penalty

                        lat_rew = trade * p_lat[1, l]
                        nxt_rew = V[t + 1, r_nxt, nxt_bid_ind]
                        
                        exp_rew += pr[l] * (lat_rew + nxt_rew)

                    if exp_rew >= opt_exp_rew:
                        opt_exp_rew = exp_rew
                        opt_buy = buy
                        opt_sell = sell

                V[t, r_ind, b_tminus_1_ind] = opt_exp_rew
                pol[t, r_ind, b_tminus_1_ind, 0] = opt_buy
                pol[t, r_ind, b_tminus_1_ind, 1] = opt_sell
                        
    return V, pol


@jit(nopython=True)
def finite_support_optimal_solution(realizations,
                                    pmf,
                                    R_MIN=0,
                                    R_MAX=18,
                                    B=B):
    # include dummy final stage.
    T = STAGES + 1
    R = np.arange(R_MIN, R_MAX + 1)
    
    V = np.full((T, len(R), len(B)), np.nan, dtype=np.float64)
    V[-1] = 0  # terminal reward
    pol = np.full((STAGES, len(R), len(B), 2), np.nan, dtype=np.float64)
    
    # At stage t, we decide on bid for stage (t+1).
    for t in range(STAGES - 1, -1, -1):  # (STAGES - 1), ..., 0
        for (r_ind, r_t) in enumerate(R):  # resource state
            for (b_tminus_1_ind, (b_buy, b_sell)) in enumerate(B):  # bids made for this stage at (t-1)
                opt_buy = np.nan
                opt_sell = np.nan
                opt_exp_rew = 0
                
                # possible bids for (t+1, t+2], i.e. the actions!
                for (nxt_bid_ind, (buy, sell)) in enumerate(B):
                    exp_rew = 0

                    for eps_1_ind, eps_1 in enumerate(realizations):
                        rew = 0

                        p_1 = deterministic_part(t+1) + eps_1
                        r_nxt = r_t
                        
                        if t > 0:  # no bidding before t=0
                            if p_1 > b_sell and r_nxt - 1 >= R_MIN:
                                r_nxt -= 1
                            elif p_1 < b_buy and r_nxt + 1 <= R_MAX:
                                r_nxt += 1
                        
                        for eps_2_ind, eps_2 in enumerate(realizations):
                            p_2 = deterministic_part(t+2) + eps_2
                            trade = 0

                            if p_2 > sell and r_nxt - 1 >= R_MIN:
                                trade = 1 * EFF_DISCHARGE
                            elif p_2 < buy and r_nxt + 1 <= R_MAX:
                                trade = -1 / EFF_CHARGE
                            elif p_2 > sell and r_nxt - 1 < R_MIN:
                                trade = -1  # penalty
                            rew += pmf[eps_2_ind] * trade * p_2

                        nxt_rew = V[t + 1, r_nxt, nxt_bid_ind]
                        exp_rew += pmf[eps_1_ind] * (rew + nxt_rew)
                    
                    if exp_rew >= opt_exp_rew:
                        opt_exp_rew = exp_rew
                        opt_buy = buy
                        opt_sell = sell
                
                V[t, r_ind, b_tminus_1_ind] = opt_exp_rew
                pol[t, r_ind, b_tminus_1_ind, 0] = opt_buy
                pol[t, r_ind, b_tminus_1_ind, 1] = opt_sell
    return V, pol


def simulate_finite_support_policy(pol, price_paths, R_MIN=0, R_MAX=18, B=B):
    _, W = price_paths.shape
    R = np.arange(R_MIN, R_MAX + 1)
    
    T = STAGES + 1

    rew = np.zeros((T, W))
    act = np.zeros((T, W))
    r = np.zeros((T + 1, W), dtype=int)

    dec = np.zeros((T, W, 2))
    prv_bid_ind = 0

    # start at first stage where rewards are evaluated
    for t in range(1, T):
        stg_bid = pol[t - 1, r[t-1], prv_bid_ind]
        
        buy, sell = np.hsplit(stg_bid, 2)
        buy = buy.reshape(-1)
        sell = sell.reshape(-1)

        dec[t, :] = stg_bid

        trade = np.zeros(W)

        exec_sell_bid = (price_paths[t, :] > sell) & (r[t, :] - 1 >= R_MIN)    
        exec_buy_bid = (price_paths[t, :] < buy) & (r[t, :] + 1 <= R_MAX)
        undersupply = (price_paths[t, :] > sell) & (r[t, :] - 1 < R_MIN)            

        act[t, exec_sell_bid] = -1
        act[t, exec_buy_bid] = 1
        
        trade[exec_sell_bid] = 1 * EFF_DISCHARGE
        trade[exec_buy_bid] = -1 / EFF_CHARGE
        trade[undersupply] = -1

        rew[t, :] = trade * price_paths[t, :]
        r[t + 1, :] = r[t, :] + act[t, :]

        prv_bid_ind = util._find_bid_ind(stg_bid, B)
    dec = dec[1:]
    act = act[1:]
    r = r[1:]
    rew = rew[1:]
    
    return dec, act, r, rew


def pseudonormal_pmf(realizations=np.arange(-20,21), mean=0, std=7):
    pmf = np.zeros(len(realizations))
    norm_scale = sum([norm.pdf(x, loc=mean, scale=std) for x in realizations])
    for ind, x in enumerate(realizations):
        pmf[ind] = norm.pdf(x, loc=mean, scale=std) / norm_scale
    return pmf, realizations

def uniform_pmf(realizations=np.arange(-20,21)):
    return np.array([1/len(realizations)] * len(realizations)), realizations


@jit(nopython=True)
def finite_support_price_process(W, T, realizations, pmf, t0=0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    paths = np.zeros((W, T), dtype=np.float64)
    for w in range(W):
        for t in range(T):
            S_t = deterministic_part(t + t0)
            eps = realizations[rand_ind_choice(pmf)]
            paths[w, t] = S_t + eps
    return paths


@jit(nopython=True)
def deterministic_part(t):
    return 15 * np.sin(3 * np.pi * t / 24) + 50


@jit(nopython=True)
def rand_ind_choice(pmf):
    # https://github.com/numba/numba/issues/2539#issuecomment-507306369
    return np.searchsorted(np.cumsum(pmf), np.random.random(), side="right")


def main():
    # test running time
    log_name = f'results_optimality_experiment/optimality_experiment_{time.strftime("%Y%m%d-%H%M%S")}.log'
    logging.basicConfig(encoding='utf-8',
                        level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[
                            logging.FileHandler(log_name),
                            logging.StreamHandler()])
    Rmax = [18, 47, 140]
    pmf, realizations = pseudonormal_pmf()

    for rmax in Rmax:
        R_MIN = 0
        R_MAX = rmax
        R = np.arange(R_MIN, R_MAX + 1)

        logging.info("#" * 20)
        logging.info(f"Running {rmax=}.")
        
        # Solve MDP
        logging.info(f"Executing BDP...")
        start = timer()

        finite_support_optimal_solution(realizations, pmf, R_MIN, R_MAX)

        elapsed = (timer() - start)
        logging.info(f"Time elapsed: {datetime.timedelta(seconds=elapsed)}")

        logging.info(f"Executing BADP-LATTICE...")
        start = timer()

        finite_support_badp_lattice(realizations, pmf, R_MIN, R_MAX)

        elapsed = (timer() - start)
        logging.info(f"Time elapsed: {datetime.timedelta(seconds=elapsed)}")

        # _, _ = finite_support_lattice_badp(realizations, pmf, R_MIN, R_MAX)

if __name__ == '__main__':
    main()