import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq, minimize
from datetime import datetime

# ── Black-Scholes pricer ──────────────────────────────────────────────────────

def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_vol(market_price, S, K, T, r):
    if T <= 0 or market_price <= 0:
        return np.nan
    try:
        return brentq(lambda s: bs_call(S, K, T, r, s) - market_price,
                      1e-6, 10.0, xtol=1e-6)
    except ValueError:
        return np.nan

# ── SVI model ─────────────────────────────────────────────────────────────────

def svi_w(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

def fit_svi(k, iv, T):
    w_market = iv**2 * T

    def loss(params):
        a, b, rho, m, sigma = params
        w_fit = svi_w(k, a, b, rho, m, sigma)
        if np.any(w_fit < 0):
            return 1e10
        return np.sum((w_fit - w_market)**2)

    best = None
    for rho_init in [-0.9, -0.7, -0.5]:
        p0     = [0.02, 0.3, rho_init, -0.05, 0.1]
        bounds = [(1e-5, 0.5), (1e-4, 2), (-0.999, 0.999), (-0.5, 0.5), (1e-5, 0.5)]
        result = minimize(loss, p0, bounds=bounds, method='L-BFGS-B')
        if best is None or result.fun < best.fun:
            best = result
    return best.x

# ── Fetch data ────────────────────────────────────────────────────────────────

ticker = yf.Ticker('SPY')
S      = ticker.history(period='1d')['Close'].iloc[-1]
r      = 0.05
today  = datetime.today()

print(f"SPY spot: {S:.2f}")

expiries = ['2026-11-11', '2026-11-13', '2026-11-20', '2026-11-27', '2026-11-30']
records  = []

for exp in expiries:
    T = (datetime.strptime(exp, '%Y-%m-%d') - today).days / 365.0
    if T <= 0:
        continue

    chain = ticker.option_chain(exp).calls
    chain = chain[(chain['bid'] > 0) & (chain['ask'] > 0)]
    chain = chain[chain['openInterest'] > 50]
    chain['mid']    = (chain['bid'] + chain['ask']) / 2
    chain['spread'] = (chain['ask'] - chain['bid']) / chain['mid']
    chain = chain[chain['spread'] < 0.5]
    chain = chain[(chain['strike'] >= 0.90 * S) & (chain['strike'] <= 1.10 * S)]
    chain['T']      = T
    chain['expiry'] = exp

    for _, row in chain.iterrows():
        iv = implied_vol(row['mid'], S, row['strike'], T, r)
        if iv is not None and 0.05 < iv < 0.6:
            records.append({
                'expiry':    exp,
                'T':         T,
                'strike':    row['strike'],
                'moneyness': np.log(row['strike'] / S),
                'iv':        iv,
            })

df = pd.DataFrame(records)
print(df.groupby('expiry')[['iv']].describe().round(3))

# ── Smile plot ────────────────────────────────────────────────────────────────

valid    = [e for e in expiries if len(df[df['expiry'] == e]) >= 5]
fig, axes = plt.subplots(1, len(valid), figsize=(16, 4), sharey=True)
k_grid   = np.linspace(-0.12, 0.12, 200)

for ax, exp in zip(axes, valid):
    sub    = df[df['expiry'] == exp]
    T      = sub['T'].iloc[0]
    params = fit_svi(sub['moneyness'].values, sub['iv'].values, T)
    w_fit  = svi_w(k_grid, *params)
    iv_fit = np.sqrt(np.maximum(w_fit / T, 0))

    ax.scatter(sub['moneyness'], sub['iv'], s=12, alpha=0.6, label='Market')
    ax.plot(k_grid, iv_fit, color='red', linewidth=1.5, label='SVI fit')
    ax.set_title(exp[5:])
    ax.set_xlabel('Log-moneyness')
    ax.axvline(0, color='black', linestyle='--', alpha=0.3)

axes[0].set_ylabel('Implied Volatility')
axes[0].legend()
plt.suptitle('SPY Implied Volatility Smile — SVI Fit', y=1.02)
plt.tight_layout()
plt.savefig('vol_smile.png', dpi=150)
plt.close()

# ── 3D surface plot ───────────────────────────────────────────────────────────

fig = plt.figure(figsize=(10, 7))
ax  = fig.add_subplot(111, projection='3d')

K_grid = np.linspace(-0.12, 0.12, 80)
T_vals = []
IV_grid = []

for exp in valid:
    sub    = df[df['expiry'] == exp]
    T      = sub['T'].iloc[0]
    params = fit_svi(sub['moneyness'].values, sub['iv'].values, T)
    w_fit  = svi_w(K_grid, *params)
    iv_fit = np.sqrt(np.maximum(w_fit / T, 0))
    T_vals.append(T)
    IV_grid.append(iv_fit)

K_mesh = np.tile(K_grid, (len(T_vals), 1))
T_mesh = np.array([[t] * len(K_grid) for t in T_vals])
IV_mesh = np.array(IV_grid)

ax.plot_surface(K_mesh, T_mesh, IV_mesh, cmap='viridis', alpha=0.85)
ax.set_xlabel('Log-moneyness')
ax.set_ylabel('Time to expiry (yrs)')
ax.set_zlabel('Implied Vol')
ax.set_title('SPY Implied Volatility Surface')
plt.tight_layout()
plt.savefig('vol_surface_3d.png', dpi=150)
plt.close()
print("Saved: vol_surface_3d.png")