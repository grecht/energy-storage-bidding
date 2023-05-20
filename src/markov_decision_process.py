import numpy as np
import pandas as pd
from numba import jit, prange
import kmeans as km
from jump_diffusion_process import PoissonSpikeProcess as psp
fast_simulate_T = psp.fast_simulate_T
import util

STAGES = 24
SETTLEMENTS = 12

# number of scenarios
W_TRAIN = 1000
W_TEST = 1000

# Storage unit efficiencies
# like Cheng, Powell (https://ieeexplore.ieee.org/abstract/document/7558191)
EFF_CHARGE = .9
EFF_DISCHARGE = .9

# Resource states
R_MIN = 0
R_MAX = 60  # 6 (500 KWh), 60 (5 MWh), 72 (6 MWh)
R_STEP = 1
R = np.arange(R_MIN, R_MAX + R_STEP, R_STEP)


def create_sampled_spaces(P_MIN, P_MAX, b_num=10, p_num=20):
    BIDS = np.linspace(0, P_MAX, num=b_num)
    B = [(buy, sell) for buy in BIDS for sell in BIDS if buy <= sell]             
    # Include possibility to place no sell bid,
    # but always buy at or below zero.
    no_bid = (0, np.inf)
    B += [no_bid]
    B = np.array(B)
    
    P = None
    if p_num == 1:
        P = np.array([26])
    else:
        P = np.linspace(P_MIN, P_MAX, num=p_num)
    return BIDS, B, P


@jit(nopython=True)
def badp_lattice(P, B, S, jdp_params, W=W_TRAIN, k=50):
    # include dummy final stage.
    T = STAGES + 1
    
    V = np.full((T, len(P), len(R), len(B)), np.nan, dtype=np.float64)
    V[-1] = 0  # terminal reward
    pol = np.full((STAGES, len(P), len(R), len(B), 2), np.nan, dtype=np.float64)

    nxt_stg = SETTLEMENTS  # TODO: -1??
    
    # At stage t, we decide on bid for stage (t+1).
    for t in range(STAGES - 1, -1, -1):  # (STAGES - 1), ..., 0
        for (p_ind, p_t) in enumerate(P):  # last known price at time of decision
            # one period for determining inventory,
            # one for the expected reward.
            # p_w = psp.fast_simulate_T(W,
            p_w = fast_simulate_T(W,
                                  t * SETTLEMENTS, 
                                  (t + 2) * SETTLEMENTS,
                                  p_t,
                                  S,
                                  jdp_params)
            # create scenario lattice
            labels, centroids = km.kmeans(p_w, k)
            p_lat = centroids.T
            pr = km.det_prob(labels, k)

            # for nxt_stg, we must interpolate later
            hi = searchsorted_parallel(P, p_lat[nxt_stg, :])
            lo = hi - 1

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
                            lat_rew = 0
                            r_lat = r_t
                            r_lat_nxt_stg = np.nan

                            if t == 0:
                                # No bidding
                                r_lat_nxt_stg = r_t
                            else:
                                # apply bids and thereby determine resource state.
                                for s in range(SETTLEMENTS):
                                    action = 0

                                    if p_lat[s, l] > b_sell and r_lat - 1 >= R_MIN:
                                        action = 1
                                    elif p_lat[s, l] < b_buy and r_lat + 1 <= R_MAX:
                                        action = -1
                                    r_lat = r_lat - action
                                r_lat_nxt_stg = r_lat

                            # determine lattice reward 
                            for s in range(SETTLEMENTS, 2 * SETTLEMENTS):
                                action = 0
                                trade = 0

                                if p_lat[s, l] > sell and r_lat - 1 >= R_MIN:
                                    action = 1
                                    trade = (1 / SETTLEMENTS) * 1 * EFF_DISCHARGE
                                elif p_lat[s, l] < buy and r_lat + 1 <= R_MAX:
                                    action = -1
                                    trade = (1 / SETTLEMENTS) * (-1) / EFF_CHARGE
                                elif p_lat[s, l] > sell and r_lat - 1 < R_MIN:
                                    trade = -1 / SETTLEMENTS  # penalty

                                lat_rew += trade * p_lat[s, l]
                                r_lat = r_lat - action

                            # linear interpolation of rewards in next stage
                            # due to prices outside of sampled state space.
                            nxt_rew = np.nan
                            if hi[l] == 0:
                                nxt_rew = V[t + 1, 0, r_lat_nxt_stg, nxt_bid_ind]
                            elif hi[l] == P.size:
                                nxt_rew = V[t + 1, -1, r_lat_nxt_stg, nxt_bid_ind]
                            else:
                                nxt_rew = np.interp(p_lat[nxt_stg, l],
                                                    (P[lo[l]], P[hi[l]]),
                                                    (V[t + 1, lo[l], r_lat_nxt_stg, nxt_bid_ind],
                                                     V[t + 1, hi[l], r_lat_nxt_stg, nxt_bid_ind]))

                            exp_rew += pr[l] * (lat_rew + nxt_rew)

                        if exp_rew >= opt_exp_rew:
                            opt_exp_rew = exp_rew
                            opt_buy = buy
                            opt_sell = sell

                    V[t, p_ind, r_ind, b_tminus_1_ind] = opt_exp_rew
                    pol[t, p_ind, r_ind, b_tminus_1_ind, 0] = opt_buy
                    pol[t, p_ind, r_ind, b_tminus_1_ind, 1] = opt_sell
                        
    return V, pol


# See https://stackoverflow.com/a/71751542
@jit(nopython=True, parallel=True)
def searchsorted_parallel(a, b):
    res = np.empty(len(b), np.intp)
    for i in prange(len(b)):
        res[i] = np.searchsorted(a, b[i])
    return res


def perfect_foresight(price_paths, B):
    price_paths_ = price_paths.copy()
    if not isinstance(price_paths, np.ndarray):
        # TODO: check indices
        price_paths_ = price_paths_.iloc[SETTLEMENTS:].values  # do not need stage 0
    else:
        price_paths_ = price_paths[SETTLEMENTS:]
    W = price_paths_.shape[1]
    stages = (int) (len(price_paths_) / SETTLEMENTS)

    # already sets terminal reward V[T, :] = 0
    V_pf = np.zeros((stages + 1, len(R), W))

    # 0 = buy, 1 = sell
    pol_pf = np.full((stages, len(R), 2, W), np.nan)

    for t in range(stages - 1, -1, -1):
        stg_idx = t * SETTLEMENTS
        
        opt_b_buy = np.full((len(R), W), np.nan)
        opt_b_sell = np.full((len(R), W), np.nan)
        opt_rew = np.zeros((len(R), W))

        for buy, sell in B:  # maximize over B
            rew = np.zeros((len(R), W))
            tmp_i = np.tile(R, (W, 1)).T

            for s in range(SETTLEMENTS):
                ind = stg_idx + s

                exec_sell_bid = (price_paths_[ind, :] > sell) & (tmp_i - 1 >= R_MIN)
                exec_buy_bid = (price_paths_[ind, :] < buy) & (tmp_i + 1 <= R_MAX)
                
                # penalty for undersupply
                penalty = (price_paths_[ind, :] > sell) & (tmp_i - 1 < R_MIN)

                action = np.zeros((len(R), W), dtype=int)
                action[exec_sell_bid] = 1
                action[exec_buy_bid] = -1

                trade = np.zeros((len(R), W))
                trade[exec_sell_bid] = (1 / SETTLEMENTS) * 1 * EFF_DISCHARGE
                trade[exec_buy_bid] = (1 / SETTLEMENTS) * (-1) * EFF_DISCHARGE
                trade[penalty] = -1 / SETTLEMENTS

                tmp_i -= action
                rew += trade * price_paths_[ind, :]
            rew += V_pf[t + 1, tmp_i, np.arange(W)]
            
            leq = rew >= opt_rew
            opt_rew[leq] = rew[leq]
            opt_b_buy[leq] = buy
            opt_b_sell[leq] = sell
            
        V_pf[t] = opt_rew
        pol_pf[t, :, 0] = opt_b_buy
        pol_pf[t, :, 1] = opt_b_sell
            
    return pol_pf


def simulate_pf_policy(pol, price_paths, i0=0):
    if not isinstance(price_paths, np.ndarray):
        price_paths_ = price_paths.iloc[SETTLEMENTS:].values
    else:
        price_paths_ = price_paths[SETTLEMENTS:]
    W = price_paths_.shape[1]
    stages = (int) (len(price_paths_) / SETTLEMENTS)

    act_pf = np.zeros((stages * SETTLEMENTS, W), dtype=int)
    rew_pf = np.zeros((stages * SETTLEMENTS, W))
    inv_pf = np.zeros((stages * SETTLEMENTS + 1, W), dtype=int)
    dec_pf = np.zeros((stages, W, 2))

    inv_pf[0] = i0

    for t in range(stages):
        stg_idx = t * SETTLEMENTS
        
        stg_bid = pol[t, inv_pf[stg_idx, :], :, np.arange(W)]
        
        buy, sell = np.hsplit(stg_bid, 2)
        buy = buy.reshape(-1)
        sell = sell.reshape(-1)
        
        dec_pf[t] = stg_bid
        
        for s in range(SETTLEMENTS):
            ind = stg_idx + s
            trade = np.zeros(W)
            
            exec_sell_bid = (price_paths_[ind, :] > sell) & (inv_pf[ind, :] - 1 >= R_MIN)
            exec_buy_bid = (price_paths_[ind, :] < buy) & (inv_pf[ind, :] + 1 <= R_MAX)
            penalty = (price_paths_[ind, :] > sell) & (inv_pf[ind, :] - 1 < R_MIN)
            
            act_pf[ind, exec_sell_bid] = -1
            act_pf[ind, exec_buy_bid] = 1
            inv_pf[ind + 1, :] = inv_pf[ind, :] + act_pf[ind, :]
            
            trade[exec_sell_bid] = (1 / SETTLEMENTS) * 1 * EFF_DISCHARGE
            trade[exec_buy_bid] = (1 / SETTLEMENTS) * (-1) / EFF_DISCHARGE
            trade[penalty] = -1 / SETTLEMENTS
                        
            rew_pf[ind, :] = trade * price_paths_[ind, :]
    return dec_pf, act_pf, inv_pf, rew_pf   


def simulate_policy(pol,
                    price_paths,
                    p0,
                    P,
                    B):
    if isinstance(price_paths, pd.Series) or isinstance(price_paths, pd.DataFrame):
        price_paths_ = price_paths.values
    else:
        price_paths_ = price_paths

    W = price_paths_.shape[1]

    P_bins = (P[1:] + P[:-1]) / 2

    # find closest value in P
    prv_p = np.searchsorted(P_bins, p0)

    T = STAGES + 1
    
    rew = np.zeros((T * SETTLEMENTS, W))
    act = np.zeros((T * SETTLEMENTS, W))
    inv = np.zeros((T * SETTLEMENTS + 1, W), dtype=int)

    dec = np.zeros((T, W, 2))
    
    # they are all the same - no rewards are applied at t=0.
    prv_bid_ind = 0

    # start at first stage where rewards are evaluated
    for t in range(1, T):
        prv_stg_idx = (t - 1) * SETTLEMENTS
        stg_idx = t * SETTLEMENTS

        stg_bid = pol[t - 1, prv_p, inv[prv_stg_idx], prv_bid_ind]
        
        buy, sell = np.hsplit(stg_bid, 2)
        buy = buy.reshape(-1)
        sell = sell.reshape(-1)

        dec[t, :] = stg_bid

        for s in range(SETTLEMENTS):
            ind = stg_idx + s
            trade = np.zeros(W)

            exec_sell_bid = (price_paths_[ind, :] > sell) & (inv[ind, :] - 1 >= R_MIN)    
            exec_buy_bid = (price_paths_[ind, :] < buy) & (inv[ind, :] + 1 <= R_MAX)
            undersupply = (price_paths_[ind, :] > sell) & (inv[ind, :] - 1 < R_MIN)            

            act[ind, exec_sell_bid] = -1
            act[ind, exec_buy_bid] = 1
            
            trade[exec_sell_bid] = (1 / SETTLEMENTS) * 1 * EFF_DISCHARGE
            trade[exec_buy_bid] = (1 / SETTLEMENTS) * (-1) / EFF_CHARGE
            trade[undersupply] = -1 / SETTLEMENTS

            rew[ind, :] = trade * price_paths_[ind, :]
            inv[ind + 1, :] = inv[ind, :] + act[ind, :]

        # observe current price (and determine its closest value in P)
        prv_p = np.searchsorted(P_bins, price_paths_[stg_idx, :])
        prv_bid_ind = util._find_bid_ind(stg_bid, B)

    dec = dec[1:]
    act = act[SETTLEMENTS:]
    inv = inv[SETTLEMENTS:]
    rew = rew[SETTLEMENTS:]
    
    return dec, act, inv, rew