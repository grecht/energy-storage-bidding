"""TODO: describe module."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.stats import linregress, norm
import pyomo.environ as pyo
from scipy.stats import skew, kurtosis

from numba import jit, prange


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


@dataclass
class JumpDiffParams:
    """Encapsulates the parameters of the jump diffusion process."""

    kappa: float
    mu: float
    sigma: float
    mu_j: float
    sigma_j: float
    p_j: float


class PoissonJumpProcess:
    """TODO: describe"""

    def __init__(self, lambd, solver='ipopt'):
        self._lambd = lambd
        self._solver = solver

    def fit_simulate(self, P, W, T, p1, seed=None):
        self.fit(P)
        return self.simulate(W, T, p1, seed=seed)

    def fit(self,
            P,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True):
        """TODO: document me"""

        # infer frequency of price time series
        self._freq = pd.infer_freq(P.index)
        self._last_ts = P.index[-1]  # + pd.Timedelta(1, self._freq)

        self._tf = AsinhTransformer(P, xi=0, lambd=self._lambd)
        X = self._tf.transform(P)


        self._ads = AdditiveDeseasonalizer(daily_seasonality=daily_seasonality,
                                           weekly_seasonality=weekly_seasonality,
                                           yearly_seasonality=yearly_seasonality).fit(X)
        X = X - self._ads.get_seasonality(X)

        self._dt = 1 / len(X)
        self._jdparams = self._estimate_ou_with_jumps(X)

        return self

    def _estimate_ou_with_jumps(self,
                                Y,
                                solver_name='ipopt',
                                kappa_0=None,
                                mu_0=None,
                                sigma_sq_0=None,
                                mu_j_0=None,
                                sigma_sq_j_0=None,
                                p_j_0=None):
        """
        Estimate the parameters of an Ornstein-Uhlenbeck process with Poisson jumps.

        We use a maximum-likelihood estimator as in Escribano, Pena, Villaplana
        (https://doi.org/10.1111/j.1468-0084.2011.00632.x).
        """
        m = pyo.ConcreteModel()

        dt = self._dt

        m.kappa = pyo.Var(domain=pyo.Reals, bounds=(0, None))
        m.mu = pyo.Var(domain=pyo.Reals)
        m.sigma_sq = pyo.Var(domain=pyo.Reals, bounds=(0, None))
        m.mu_j = pyo.Var(domain=pyo.Reals)
        m.sigma_sq_j = pyo.Var(domain=pyo.Reals, bounds=(0, None))
        m.p_j = pyo.Var(domain=pyo.Reals)

        def p_j_constr_lo(m):
            return 0 <= m.p_j * dt

        def p_j_constr_hi(m):
            return m.p_j * dt <= 1

        m.p_j_constr_lo = pyo.Constraint(rule=p_j_constr_lo)
        m.p_j_constr_hi = pyo.Constraint(rule=p_j_constr_hi)
        
        # initial values
        m.kappa = kappa_0 if kappa_0 else 10
        m.mu = mu_0 if mu_0 else np.median(Y)
        m.sigma_sq = sigma_sq_0 if sigma_sq_0 else np.var(Y)
        m.mu_j = mu_j_0 if mu_j_0 else np.median(Y)
        m.sigma_sq_j = sigma_sq_j_0 if sigma_sq_j_0 else np.var(Y)
        m.p_j = p_j_0 / dt if p_j_0 else .5 / dt

        # minimize negative log-likelihood
        m.obj = pyo.Objective(
            expr=- sum(
                [pyo.log(
                    m.p_j * dt
                    * (1 / (2 * np.pi * (m.sigma_sq + m.sigma_sq_j)) ** .5)
                    * (pyo.exp((- (Y[t+1] - (Y[t] + m.kappa * (m.mu - Y[t]) * dt + m.mu_j)) ** 2)
                            / (2 * (m.sigma_sq + m.sigma_sq_j))))
                    + (1 - m.p_j * dt)
                    * (1 / (2 * np.pi * m.sigma_sq) ** .5)
                    * (pyo.exp((- (Y[t+1] - (Y[t] + m.kappa * (m.mu - Y[t]) * dt)) ** 2)
                            / (2 * m.sigma_sq)))
                )
                for t in range(len(Y) - 1)]
            ),
            sense=pyo.minimize
        )

        solver = pyo.SolverFactory(solver_name)
        solver.solve(m)

        kappa = pyo.value(m.kappa)
        mu = pyo.value(m.mu)
        sigma = np.sqrt(pyo.value(m.sigma_sq))
        mu_j = pyo.value(m.mu_j)
        sigma_j = np.sqrt(pyo.value(m.sigma_sq_j))
        p_j = pyo.value(m.p_j)

        return JumpDiffParams(kappa, mu, sigma, mu_j, sigma_j, p_j)

    def simulate(self,
                 W,
                 T,
                 p1,
                 last_ts=None,
                 freq=None,
                 seed=None,
                 round_to_cents=True):
        """
        Simulate W paths with T increments and starting price p1.

        Uses the Euler-Maruyama method.
        """
        rng = default_rng(seed=seed)

        # create index for simulation
        # Start from first increment after fitting data by default.
        if not last_ts:
            last_ts = self._last_ts
        if not freq:
            freq = self._freq
        sim_index = pd.date_range(start=last_ts, periods=T+1, freq=freq)

        # transform starting price
        y0 = self._tf.transform(p1)
        y0 = y0 - self._ads.get_seasonality(pd.Series(index=[sim_index[0]],
                                                      data=y0))

        xi = rng.normal(size=(T, W))
        xi_j = rng.normal(size=(T, W))
        b = rng.binomial(1, self._jdparams.p_j * self._dt, size=(T, W))

        y = np.zeros((T + 1, W))
        y[0, :] = y0

        # this is the standard deviation of a Wiener process increment.
        sqrt_dt = np.sqrt(self._dt)

        kappa = self._jdparams.kappa
        mu = self._jdparams.mu
        sigma = self._jdparams.sigma
        mu_j = self._jdparams.mu_j
        sigma_j = self._jdparams.sigma_j

        for t in range(T):
            y[t + 1, :] = (y[t, :]
                           + kappa * (mu - y[t, :]) * self._dt
                           + sigma * sqrt_dt * xi[t, :]
                           + b[t, :] * (mu_j + sigma_j * xi_j[t, :]))

        sim = pd.DataFrame(data=y, index=sim_index)
        sim = sim.iloc[1:, :]  # remove starting price

        # transform back by first adding seasonality and then
        # applying inverse variance-stabilizing transformation.
        sim = (sim.T + self._ads.get_seasonality(sim.iloc[:, 0])).T
        sim = self._tf.inv_transform(sim)

        if round_to_cents:
            return np.round(sim, 2)
        return sim


class PoissonSpikeProcess:
    """
    Detects and removes spikes, models the difference with a standard
    Ornstein-Uhlenbeck process, then adds back spikes following their
    empirical probability and size distribution.

    Based on Zhou et al (https://dx.doi.org/10.2139/ssrn.1962414).
    """

    def __init__(self, scale=30):
        self._scale = scale

    def fit(self,
            p,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            ret_despiked_p=False):
        self._freq = pd.infer_freq(p.index)
        self._last_ts = p.index[-1]
        self._daily_seasonality = daily_seasonality
        self._weekly_seasonality = weekly_seasonality
        self._yearly_seasonality = yearly_seasonality

        self._tf = AsinhTransformer(p, lambd=self._scale, normalize=False)

        self._params, p_hat = self._estimate_parameters(p)

        if ret_despiked_p:
            return self, p_hat
        return self

    def simulate(self,
                 W,
                 T,
                 p0,
                 last_ts=None,
                 freq=None,
                 seed=None,
                 round_to_cents=True):
        """
        Simulate W paths with T increments and starting price p0.
        """
        rng = default_rng(seed=seed)

        # create index for simulation
        # Start from first increment after fitting data by default.
        if not last_ts:
            last_ts = self._last_ts
        if not freq:
            freq = self._freq
        sim_index = pd.date_range(start=last_ts, periods=T+1, freq=freq)[1:]

        # transform starting price
        x0 = self._tf.transform(p0)
        x0 = x0 - self._ads.get_seasonality(pd.Series(index=[last_ts], data=x0))

        kappa = self._params[0]
        mu = self._params[1]
        sigma = self._params[2]
        emp_jump_distr = self._params[3]
        emp_jump_prob = self._params[4]

        xi = rng.normal(size=(T, W))

        x = np.zeros((T + 1, W))
        x[0, :] = x0

        # simulate despiked difference
        for t in range(T):
            x[t + 1, :] = x[t, :] + kappa * (mu - x[t, :]) + sigma * xi[t, :]

        x = x[1:, :]
        paths = pd.DataFrame(index=sim_index)
        s = self._ads.get_seasonality(paths)

        # nan values in seasonality occur due to missing values in original data.
        s = s.fillna(s.median())

        x = (x.T + s.values).T
        x = self._tf.inv_transform(x)

        # add jumps
        b = rng.binomial(1, emp_jump_prob, size=(T, W))
        xi_j = rng.choice(emp_jump_distr, size=(T, W))
        x += b * xi_j

        paths = pd.DataFrame(data=x, index=sim_index)

        if round_to_cents:
            return np.round(paths, 2)
        return paths

    def get_S_before_sim(self, t0_ts, T, freq=None):
        # transform starting price
        if not t0_ts:
            t0_ts = self._last_ts
        if not freq:
            freq = self._freq
        paths_index = pd.date_range(start=t0_ts, periods=T+1, freq=freq)
        paths = pd.DataFrame(index=paths_index)
        S = self._ads.get_seasonality(paths)
        return S.values

    @staticmethod
    @jit(nopython=True, parallel=True)
    def fast_simulate(W,
                      start_t,
                      end_t,
                      p0,
                      S,
                      params,
                      scale=30,
                      offset=0):
        """
        Simulate W paths with T increments and starting price p1.
        """
        T = end_t - start_t

        x0 = np.arcsinh((p0 - offset) / scale)
        x0 = x0 - S[0]

        kappa = params[0]
        mu = params[1]
        sigma = params[2]
        emp_jump_distr = params[3]
        emp_jump_prob = params[4]

        x = np.zeros((T + 1, W), dtype=np.float64)
        x[0, :] = x0

        xi_j = np.zeros((T, W), dtype=np.float64)

        for t in range(T):
            for w in prange(W):
                # simulate despiked difference
                x[t + 1, w] = x[t, w] + kappa * (mu - x[t, w]) + sigma * np.random.normal()

                # sample jumps
                if np.random.binomial(1, emp_jump_prob) == 1:
                    xi_j[t, w] = np.random.choice(emp_jump_distr)

        x = x[1:, :]
        s = S[start_t:end_t]

        x = (x.T + s).T
        x = np.sinh(x) * scale + offset

        # add jumps
        x += xi_j
        return x

    @staticmethod
    @jit(nopython=True, parallel=True)
    def fast_simulate_T(W,
                        start_t,
                        end_t,
                        p0,
                        S,
                        params,
                        scale=30,
                        offset=0):
        """
        Simulate W paths with T increments and starting price p1.
        """
        T = end_t - start_t

        x0 = np.arcsinh((p0 - offset) / scale)
        x0 = x0 - S[0]

        kappa = params[0]
        mu = params[1]
        sigma = params[2]
        emp_jump_distr = params[3]
        emp_jump_prob = params[4]

        x = np.zeros((W, T + 1), dtype=np.float64)
        x[0, :] = x0

        xi_j = np.zeros((W, T), dtype=np.float64)

        for w in prange(W):
            for t in range(T):
                # simulate despiked difference
                x[w, t + 1] = x[w, t] + kappa * (mu - x[w, t]) + sigma * np.random.normal()

                # sample jumps
                if np.random.binomial(1, emp_jump_prob) == 1:
                    xi_j[w, t] = np.random.choice(emp_jump_distr)

        x = x[:, 1:]
        x = x + S[start_t:end_t]
        x = np.sinh(x) * scale + offset

        # add jumps
        x += xi_j
        return x

    def _estimate_parameters(self, p):
        # despiked prices, spikes
        p_hat, j_hat = self._detect_spikes(p)

        # emp_jump_distr = j_hat[j_hat != 0].values
        emp_jump_distr = j_hat[j_hat != 0]
        emp_jump_prob = len(j_hat[j_hat != 0]) / len(j_hat)

        # estimate ou process parameters on despiked and deseasonalized prices.
        p_tr = self._tf.transform(p_hat)

        # estimate seasonality function on despiked, transformed prices
        self._ads = AdditiveDeseasonalizer(daily_seasonality=self._daily_seasonality,
                                           weekly_seasonality=self._weekly_seasonality,
                                           yearly_seasonality=self._yearly_seasonality).fit(p_tr)
        x = p_tr - self._ads.get_seasonality(p_tr)

        kappa, mu, sigma = self._estimate_ou_process(x)

        return (kappa, mu, sigma, emp_jump_distr, emp_jump_prob), p_hat

    def _detect_spikes(self, p, lo_q=.01, hi_q=.96):
        p_hat = p.copy()
        j_hat = np.zeros(p_hat.size)
        spikes = np.zeros(p_hat.size)

        ads = AdditiveDeseasonalizer(daily_seasonality=False, weekly_seasonality=False)
        yearly_seasonality = ads.fit_get_seasonality(p_hat).values

        lo, hi = np.quantile(p_hat, [lo_q, hi_q])
        spikes = (p_hat < lo) | (p_hat > hi)

        j_hat[spikes] = p_hat[spikes] - yearly_seasonality[spikes]
        p_hat[spikes] = yearly_seasonality[spikes]

        return p_hat, j_hat

    def _estimate_ou_process(self, p):
        # https://link.springer.com/article/10.1023/A:1013846631785

        # P_{t+1} = α + ɸ * P_t + eps
        y = p[1:].values
        X = p[:-1].values

        reg = linregress(X, y)

        phi = reg.slope.squeeze()
        alpha = reg.intercept

        # determine parameters of stochastic process
        kappa = 1 - phi
        mu = alpha / (1 - phi)

        # volatility = standard deviation of residuals
        y_pred = phi * X + alpha
        sigma = np.std(y - y_pred)

        return kappa, mu, sigma

class AdditiveDeseasonalizer:
    def __init__(self,
                 daily_seasonality=True,
                 weekly_seasonality=True,
                 yearly_seasonality=True):
        self._daily_seasonality = daily_seasonality
        self._weekly_seasonality = weekly_seasonality
        self._yearly_seasonality = yearly_seasonality

        self._daily_pattern = None
        self._weekly_pattern = None
        self._yearly_pattern = None

    def fit_get_seasonality(self, ts):
        self.fit(ts)
        return self.get_seasonality(ts)

    def fit(self, ts):
        # remove Feb 29th
        ts = ts.loc[~((ts.index.day == 29) & (ts.index.month == 2))]
        if self._daily_seasonality:
            self._daily_pattern, s_daily = self._det_daily_seasonality(ts)
            ts = ts - s_daily
        if self._weekly_seasonality:
            self._weekly_pattern, s_weekly = self._det_weekly_seasonality(ts)
            ts = ts - s_weekly
        if self._yearly_seasonality:
            self._yearly_pattern, s_yearly = self._det_yearly_seasonality(ts)
            ts = ts - s_yearly

        return self

    def get_seasonality(self, ts):
        """
        Match seasonality to DatetimeIndex of 'ts'.
        """
        s = np.zeros(len(ts))

        if self._daily_seasonality:
            s += self._match_pattern_to_ts(ts, self._daily_pattern, "day")
        if self._weekly_seasonality:
            s += self._match_pattern_to_ts(ts, self._weekly_pattern, "week")
        if self._yearly_seasonality:
            s += self._match_pattern_to_ts(ts, self._yearly_pattern, "year")

        return s

    def _det_daily_seasonality(self, X):
        """
        Determine daily seasonality based on hourly medians.

        X must have a DatetimeIndex.
        """
        medians = X.groupby([X.index.hour, X.index.minute]).median()
        medians.index = medians.index.rename(["hour", "minute"])
        medians.name = "pattern"

        daily_seasonality = self._match_pattern_to_ts(X, medians, length="day")

        return medians, daily_seasonality

    def _det_weekly_seasonality(self, X):
        """
        Determine weekly seasonality based on hourly medians.

        X must have a DatetimeIndex.
        """
        medians = X.groupby([X.index.weekday,
                             X.index.hour,
                             X.index.minute]).median()
        medians.index = medians.index.rename(["weekday", "hour", "minute"])
        medians.name = "pattern"

        weekly_seasonality = self._match_pattern_to_ts(X,
                                                       medians,
                                                       length="week")

        return medians, weekly_seasonality

    def _det_yearly_seasonality(self, X):
        """Determine yearly seasonality.

        See Geman, Roncoroni (https://doi.org/10.1086/500675).
        """
        median_year = X.groupby([X.index.month,
                                 X.index.day,
                                 X.index.hour,
                                 X.index.minute]).median()
        median_year.index = median_year.index.rename(["month",
                                                      "day",
                                                      "hour",
                                                      "minute"])

        inc_fracs = np.array([(i+1) / len(median_year)
                              for i in range(len(median_year))])
        season_matrix = np.array([np.sin(2 * np.pi * inc_fracs),
                                  np.cos(2 * np.pi * inc_fracs),
                                  np.sin(4 * np.pi * inc_fracs),
                                  np.cos(4 * np.pi * inc_fracs),
                                  inc_fracs,
                                  np.ones(len(inc_fracs))]).T

        season_params, _, _, _ = np.linalg.lstsq(season_matrix,
                                                 median_year,
                                                 rcond=None)
        pattern = season_matrix @ season_params

        pattern = pd.DataFrame(index=median_year.index, data=pattern)
        pattern.name = "pattern"
        pattern = pattern.rename(columns={0: "pattern"})

        yearly_seasonality = self._match_pattern_to_ts(X,
                                                       pattern,
                                                       length="year")
        return pattern, yearly_seasonality

    @staticmethod
    def _match_pattern_to_ts(ts, pattern, length):
        """
        Extrapolates pattern to length of ts.

        length in ["day", "week", "year"]
        """
        pattern_extr = pd.DataFrame(data=np.zeros(len(ts)),
                                    index=ts.index,
                                    columns=["value"])

        pattern_extr["hour"] = pattern_extr.index.hour
        pattern_extr["minute"] = pattern_extr.index.minute

        if length == "day":
            pattern_extr = pattern_extr.join(pattern, on=["hour", "minute"])
        elif length == "week":
            pattern_extr["weekday"] = pattern_extr.index.weekday
            pattern_extr = pattern_extr.join(pattern,
                                             on=["weekday", "hour", "minute"])
        elif length == "year":
            pattern_extr["month"] = pattern_extr.index.month
            pattern_extr["day"] = pattern_extr.index.day
            pattern_extr = pattern_extr.join(pattern,
                                             on=["month",
                                                 "day",
                                                 "hour",
                                                 "minute"])
            # Linearly interpolate in case of Feb 29th.
            pattern_extr = pattern_extr.interpolate()

        pattern_extr["value"] = pattern_extr["pattern"]
        pattern_extr = pattern_extr["value"]
        return pattern_extr



@jit(nopython=True)
def seed(a):
    np.random.seed(a)

class AsinhTransformer:
    """
    Transforms data using the area hyperbolic sine transform.

    See Schneider (http://doi.org/10.21314/JEM.2011.079). This is a
    variance-stabilizing transformation suited for negative electricity prices,
    for which the historically used log-transform is undefined. It aims
    to dampen the characteristic spikes to which the calibration of forecasting
    models is very sensitive. An Ornstein-Uhlenbeck process in Asinh-space
    still has a closed-form solution given by the Johnson distribution (as a
    counterpart to the lognormal distribution in log-space).
    More variance-stabilizing transformations are proposed by Uniejewski et al
    (https://doi.org/10.1109/TPWRS.2017.2734563), who also suggest using a
    robust normalization beforehand as we do here.
    """

    def __init__(self, p, xi=0, lambd=1, normalize=True):
        """
        Initialize.

        The normalization parameters are determined using p.
        """
        self.xi = xi
        self.lambd = lambd

        self.normalize = normalize
        if normalize:
            # normalization parameters
            self.a = np.median(p)
            # median absolute deviation adjusted by a factor for asymptotically
            # normal consistency to the standard deviation
            self.b = np.median(np.abs(p - np.median(p))) * (1 / norm.ppf(.75))

    def transform(self, x):
        """Normalize x, then apply the Asinh-transform."""
        if self.normalize:
            x_norm = self._normalize(x)
            return self._asinh_transform(x_norm)
        return self._asinh_transform(x)

    def inv_transform(self, x):
        """Undo the transformation based on p's original values."""
        if self.normalize:
            x_norm = self._inv_asinh_transform(x)
            return self._inv_normalize(x_norm)
        return self._inv_asinh_transform(x)

    def _normalize(self, x):
        """Normalize 'p' with shift 'a' and scale 'b'."""
        return 1 / self.b * (x - self.a)

    def _inv_normalize(self, x):
        """Undo the normalization."""
        return self.b * x + self.a

    def _asinh_transform(self, x):
        return np.arcsinh((x - self.xi) / self.lambd)

    def _inv_asinh_transform(self, x):
        return np.sinh(x) * self.lambd + self.xi


def main():
    pass


if __name__ == "__main__":
    main()
