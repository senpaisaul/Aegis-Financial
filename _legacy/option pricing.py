#!/usr/bin/env python
# coding: utf-8

# In[9]:


"""
Cell 1: Environment Setup and Library Imports
Option Pricing Calculator - Production Version
"""

import sys
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# Optional: Numba for performance optimization
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Validation
print("=" * 60)
print("ENVIRONMENT SETUP")
print("=" * 60)
print(f"Python Version:    {sys.version.split()[0]}")
print(f"NumPy Version:     {np.__version__}")
print(f"Pandas Version:    {pd.__version__}")
print(f"Numba Available:   {NUMBA_AVAILABLE}")
print("=" * 60)
print("Setup complete. Proceed to Cell 2.")
print("=" * 60)


# In[10]:


"""
Cell 2: User Input and Market Data Fetching
Collects option parameters and retrieves market data from Yahoo Finance
"""

def validate_ticker(ticker):
    """Validate ticker symbol by attempting to fetch data"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        if hist.empty:
            return False, None
        return True, stock
    except Exception:
        return False, None


def get_user_inputs():
    """Collect and validate user inputs for option pricing"""
    
    print("=" * 60)
    print("OPTION PRICING CALCULATOR - INPUT PARAMETERS")
    print("=" * 60)
    
    # Ticker Symbol
    while True:
        ticker = input("\nEnter Stock Ticker Symbol (e.g., AAPL, MSFT, SPY): ").strip().upper()
        if not ticker:
            print("Error: Ticker cannot be empty.")
            continue
        valid, stock = validate_ticker(ticker)
        if valid:
            print(f"Validated: {ticker}")
            break
        print(f"Error: Invalid ticker '{ticker}'. Please try again.")
    
    # Option Type
    while True:
        option_type = input("\nEnter Option Type (call/put): ").strip().lower()
        if option_type in ['call', 'put']:
            print(f"Selected: {option_type.upper()}")
            break
        print("Error: Please enter 'call' or 'put'.")
    
    # Strike Price
    while True:
        strike_input = input("\nEnter Strike Price (or press Enter for ATM): ").strip()
        if strike_input == "":
            strike_price = None
            print("Selected: At-The-Money (ATM)")
            break
        try:
            strike_price = float(strike_input)
            if strike_price <= 0:
                print("Error: Strike price must be positive.")
                continue
            print(f"Selected: ${strike_price:.2f}")
            break
        except ValueError:
            print("Error: Please enter a valid number.")
    
    # Days to Expiry
    while True:
        days_input = input("\nEnter Days to Expiry (e.g., 30, 60, 90): ").strip()
        try:
            days_to_expiry = int(days_input)
            if days_to_expiry <= 0:
                print("Error: Days to expiry must be positive.")
                continue
            if days_to_expiry > 1095:
                print("Warning: Extended expiry period may reduce accuracy.")
            print(f"Selected: {days_to_expiry} days")
            break
        except ValueError:
            print("Error: Please enter a whole number.")
    
    # Confirmation
    print("\n" + "-" * 60)
    print("INPUT SUMMARY")
    print("-" * 60)
    print(f"Ticker:          {ticker}")
    print(f"Option Type:     {option_type.upper()}")
    print(f"Strike Price:    {'ATM' if strike_price is None else f'${strike_price:.2f}'}")
    print(f"Days to Expiry:  {days_to_expiry}")
    print("-" * 60)
    
    confirm = input("\nConfirm inputs (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y', '']:
        print("\nRestarting input collection...\n")
        return get_user_inputs()
    
    return ticker, strike_price, option_type, days_to_expiry


class MarketDataFetcher:
    """Fetches and caches market data from Yahoo Finance"""
    
    def __init__(self, ticker):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self._spot_price = None
        self._volatility = None
        self._dividend_yield = None
        
    @property
    def spot_price(self):
        """Current stock price"""
        if self._spot_price is None:
            hist = self.stock.history(period="5d")
            self._spot_price = float(hist['Close'].iloc[-1])
        return self._spot_price
    
    @property
    def historical_volatility(self):
        """Annualized historical volatility (1-year)"""
        if self._volatility is None:
            hist = self.stock.history(period="1y")
            returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
            self._volatility = float(returns.std() * np.sqrt(252))
        return self._volatility
    
    @property
    def dividend_yield(self):
        """Annual dividend yield"""
        if self._dividend_yield is None:
            try:
                info = self.stock.info
                self._dividend_yield = float(info.get('dividendYield', 0) or 0)
            except Exception:
                self._dividend_yield = 0.0
        return self._dividend_yield
    
    def get_risk_free_rate(self):
        """Fetch risk-free rate from 13-week Treasury Bill"""
        try:
            treasury = yf.Ticker("^IRX")
            hist = treasury.history(period="5d")
            return float(hist['Close'].iloc[-1] / 100)
        except Exception:
            return 0.05  # Default fallback


# Execute input collection
TICKER, STRIKE_PRICE, OPTION_TYPE, DAYS_TO_EXPIRY = get_user_inputs()

# Fetch market data
print("\nFetching market data...")
data = MarketDataFetcher(TICKER)

S = data.spot_price
K = STRIKE_PRICE if STRIKE_PRICE else S
T = DAYS_TO_EXPIRY / 365
r = data.get_risk_free_rate()
sigma = data.historical_volatility
q = data.dividend_yield

# Calculate moneyness
moneyness = S / K
if OPTION_TYPE == "call":
    if moneyness > 1.02:
        moneyness_status = "ITM"
    elif moneyness < 0.98:
        moneyness_status = "OTM"
    else:
        moneyness_status = "ATM"
else:
    if moneyness < 0.98:
        moneyness_status = "ITM"
    elif moneyness > 1.02:
        moneyness_status = "OTM"
    else:
        moneyness_status = "ATM"

# Display market data
print("\n" + "=" * 60)
print(f"MARKET DATA: {TICKER}")
print("=" * 60)
print(f"Spot Price (S):          ${S:.2f}")
print(f"Strike Price (K):        ${K:.2f}")
print(f"Time to Expiry (T):      {DAYS_TO_EXPIRY} days ({T:.4f} years)")
print(f"Risk-Free Rate (r):      {r*100:.2f}%")
print(f"Volatility (sigma):      {sigma*100:.2f}%")
print(f"Dividend Yield (q):      {q*100:.2f}%")
print(f"Option Type:             {OPTION_TYPE.upper()}")
print(f"Moneyness:               {moneyness_status} ({moneyness:.2%})")
print("=" * 60)
print("Data loaded. Proceed to Cell 3.")
print("=" * 60)


# In[11]:


"""
Cell 3: Black-Scholes Model
Analytical solution for European option pricing
"""

# Dependency check
try:
    _ = S, K, T, r, sigma, q, OPTION_TYPE, TICKER
except NameError:
    raise RuntimeError("Error: Execute Cell 2 before running this cell.")

print("=" * 60)
print("BLACK-SCHOLES MODEL")
print("=" * 60)


class BlackScholesModel:
    """
    Black-Scholes-Merton model for European option pricing.
    
    Attributes:
        S: Spot price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate
        sigma: Volatility
        q: Dividend yield
    """
    
    def __init__(self, S, K, T, r, sigma, q=0):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self._calculate_d1_d2()
    
    def _calculate_d1_d2(self):
        """Calculate d1 and d2 parameters"""
        sqrt_T = np.sqrt(self.T)
        self.d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * sqrt_T)
        self.d2 = self.d1 - self.sigma * sqrt_T
    
    def call_price(self):
        """European call option price"""
        return (self.S * np.exp(-self.q * self.T) * norm.cdf(self.d1) - 
                self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2))
    
    def put_price(self):
        """European put option price"""
        return (self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - 
                self.S * np.exp(-self.q * self.T) * norm.cdf(-self.d1))
    
    def price(self, option_type="call"):
        """Get option price by type"""
        return self.call_price() if option_type.lower() == "call" else self.put_price()
    
    def delta(self, option_type="call"):
        """Delta: dV/dS"""
        if option_type.lower() == "call":
            return np.exp(-self.q * self.T) * norm.cdf(self.d1)
        return np.exp(-self.q * self.T) * (norm.cdf(self.d1) - 1)
    
    def gamma(self):
        """Gamma: d2V/dS2"""
        return (np.exp(-self.q * self.T) * norm.pdf(self.d1)) / (self.S * self.sigma * np.sqrt(self.T))
    
    def theta(self, option_type="call"):
        """Theta: dV/dT (per day)"""
        term1 = -(self.S * self.sigma * np.exp(-self.q * self.T) * norm.pdf(self.d1)) / (2 * np.sqrt(self.T))
        if option_type.lower() == "call":
            term2 = self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(self.d1)
            term3 = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        else:
            term2 = -self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(-self.d1)
            term3 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
        return (term1 + term2 + term3) / 365
    
    def vega(self):
        """Vega: dV/dsigma (per 1% change)"""
        return self.S * np.exp(-self.q * self.T) * np.sqrt(self.T) * norm.pdf(self.d1) / 100
    
    def rho(self, option_type="call"):
        """Rho: dV/dr (per 1% change)"""
        if option_type.lower() == "call":
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2) / 100
        return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2) / 100
    
    def get_all_greeks(self, option_type="call"):
        """Return all Greeks as dictionary"""
        return {
            'delta': self.delta(option_type),
            'gamma': self.gamma(),
            'theta': self.theta(option_type),
            'vega': self.vega(),
            'rho': self.rho(option_type)
        }


# Calculate results
bs_model = BlackScholesModel(S, K, T, r, sigma, q)
bs_price = bs_model.price(OPTION_TYPE)
bs_greeks = bs_model.get_all_greeks(OPTION_TYPE)

# Display results
print(f"\nOption Price: ${bs_price:.4f}")
print("\nGreeks:")
print("-" * 40)
print(f"Delta:    {bs_greeks['delta']:>12.4f}")
print(f"Gamma:    {bs_greeks['gamma']:>12.4f}")
print(f"Theta:    ${bs_greeks['theta']:>11.4f}/day")
print(f"Vega:     ${bs_greeks['vega']:>11.4f}/1%")
print(f"Rho:      ${bs_greeks['rho']:>11.4f}/1%")
print("-" * 40)
print("\n" + "=" * 60)
print("Black-Scholes complete. Proceed to Cell 4.")
print("=" * 60)


# In[12]:


"""
Cell 4: Monte Carlo Simulation
Path-dependent option pricing for Asian, Lookback, and Barrier options
"""

# Dependency check
try:
    _ = S, K, T, r, sigma, q, OPTION_TYPE, TICKER
except NameError:
    raise RuntimeError("Error: Execute Cell 2 before running this cell.")

print("=" * 60)
print("MONTE CARLO SIMULATION")
print("=" * 60)


class MonteCarloModel:
    """
    Monte Carlo simulation for option pricing.
    
    Supports: European, Asian, Lookback, and Barrier options.
    Uses antithetic variates for variance reduction.
    """
    
    def __init__(self, S, K, T, r, sigma, q=0, n_simulations=100000, n_steps=252):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.dt = T / n_steps
        
    def _generate_paths(self, antithetic=True, seed=42):
        """Generate price paths using Geometric Brownian Motion"""
        np.random.seed(seed)
        
        n_sims = self.n_simulations // 2 if antithetic else self.n_simulations
        
        drift = (self.r - self.q - 0.5 * self.sigma**2) * self.dt
        vol = self.sigma * np.sqrt(self.dt)
        
        Z = np.random.standard_normal((n_sims, self.n_steps))
        
        if antithetic:
            Z = np.vstack([Z, -Z])
        
        log_returns = drift + vol * Z
        log_paths = np.cumsum(log_returns, axis=1)
        
        paths = self.S * np.exp(log_paths)
        paths = np.column_stack([np.full(self.n_simulations, self.S), paths])
        
        return paths
    
    def european_option_price(self, option_type="call"):
        """Price European option"""
        paths = self._generate_paths()
        final_prices = paths[:, -1]
        
        if option_type.lower() == "call":
            payoffs = np.maximum(final_prices - self.K, 0)
        else:
            payoffs = np.maximum(self.K - final_prices, 0)
        
        discount = np.exp(-self.r * self.T)
        price = discount * np.mean(payoffs)
        std_error = discount * np.std(payoffs) / np.sqrt(self.n_simulations)
        
        return price, std_error
    
    def asian_option_price(self, option_type="call", averaging="arithmetic"):
        """Price Asian option with arithmetic or geometric averaging"""
        paths = self._generate_paths()
        
        if averaging == "arithmetic":
            avg_prices = np.mean(paths, axis=1)
        else:
            avg_prices = np.exp(np.mean(np.log(paths), axis=1))
        
        if option_type.lower() == "call":
            payoffs = np.maximum(avg_prices - self.K, 0)
        else:
            payoffs = np.maximum(self.K - avg_prices, 0)
        
        discount = np.exp(-self.r * self.T)
        price = discount * np.mean(payoffs)
        std_error = discount * np.std(payoffs) / np.sqrt(self.n_simulations)
        
        return price, std_error
    
    def lookback_option_price(self, option_type="call"):
        """Price floating strike Lookback option"""
        paths = self._generate_paths()
        final_prices = paths[:, -1]
        
        if option_type.lower() == "call":
            min_prices = np.min(paths, axis=1)
            payoffs = np.maximum(final_prices - min_prices, 0)
        else:
            max_prices = np.max(paths, axis=1)
            payoffs = np.maximum(max_prices - final_prices, 0)
        
        discount = np.exp(-self.r * self.T)
        price = discount * np.mean(payoffs)
        std_error = discount * np.std(payoffs) / np.sqrt(self.n_simulations)
        
        return price, std_error
    
    def barrier_option_price(self, option_type="call", barrier_type="down-and-out", barrier_level=None):
        """Price Barrier option"""
        if barrier_level is None:
            barrier_level = self.S * 0.9 if "down" in barrier_type else self.S * 1.1
        
        paths = self._generate_paths()
        final_prices = paths[:, -1]
        
        if "down" in barrier_type:
            barrier_hit = np.any(paths <= barrier_level, axis=1)
        else:
            barrier_hit = np.any(paths >= barrier_level, axis=1)
        
        if option_type.lower() == "call":
            base_payoffs = np.maximum(final_prices - self.K, 0)
        else:
            base_payoffs = np.maximum(self.K - final_prices, 0)
        
        if "out" in barrier_type:
            payoffs = np.where(barrier_hit, 0, base_payoffs)
        else:
            payoffs = np.where(barrier_hit, base_payoffs, 0)
        
        discount = np.exp(-self.r * self.T)
        price = discount * np.mean(payoffs)
        std_error = discount * np.std(payoffs) / np.sqrt(self.n_simulations)
        
        return price, std_error, barrier_level


# Calculate results
n_steps_calc = max(min(int(T * 252), 252), 21)
mc_model = MonteCarloModel(S, K, T, r, sigma, q, n_simulations=100000, n_steps=n_steps_calc)

print(f"\nSimulations: {mc_model.n_simulations:,}")
print(f"Time Steps:  {mc_model.n_steps}")
print("\nCalculating prices...")

mc_european_price, mc_european_se = mc_model.european_option_price(OPTION_TYPE)
mc_asian_arith_price, mc_asian_arith_se = mc_model.asian_option_price(OPTION_TYPE, "arithmetic")
mc_asian_geo_price, mc_asian_geo_se = mc_model.asian_option_price(OPTION_TYPE, "geometric")
mc_lookback_price, mc_lookback_se = mc_model.lookback_option_price(OPTION_TYPE)

barrier_type = "down-and-out" if OPTION_TYPE == "call" else "up-and-out"
mc_barrier_price, mc_barrier_se, barrier_level = mc_model.barrier_option_price(OPTION_TYPE, barrier_type)

# Display results
print("\nResults:")
print("-" * 55)
print(f"{'Option Type':<25} {'Price':>12} {'Std Error':>14}")
print("-" * 55)
print(f"{'European':<25} ${mc_european_price:>10.4f}   +/-${mc_european_se:.4f}")
print(f"{'Asian (Arithmetic)':<25} ${mc_asian_arith_price:>10.4f}   +/-${mc_asian_arith_se:.4f}")
print(f"{'Asian (Geometric)':<25} ${mc_asian_geo_price:>10.4f}   +/-${mc_asian_geo_se:.4f}")
print(f"{'Lookback (Floating)':<25} ${mc_lookback_price:>10.4f}   +/-${mc_lookback_se:.4f}")
print(f"{'Barrier (' + barrier_type + ')':<25} ${mc_barrier_price:>10.4f}   +/-${mc_barrier_se:.4f}")
print("-" * 55)
print(f"Barrier Level: ${barrier_level:.2f}")
print("\n" + "=" * 60)
print("Monte Carlo complete. Proceed to Cell 5.")
print("=" * 60)


# In[13]:


"""
Cell 5: Binomial Tree Model
Cox-Ross-Rubinstein model for American option pricing
"""

# Dependency check
try:
    _ = S, K, T, r, sigma, q, OPTION_TYPE, TICKER
except NameError:
    raise RuntimeError("Error: Execute Cell 2 before running this cell.")

print("=" * 60)
print("BINOMIAL TREE MODEL")
print("=" * 60)


class BinomialModel:
    """
    Cox-Ross-Rubinstein Binomial Tree model.
    
    Supports both European and American option pricing
    with early exercise valuation.
    """
    
    def __init__(self, S, K, T, r, sigma, q=0, n_steps=500):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.n_steps = n_steps
        self.dt = T / n_steps
        
        # CRR parameters
        self.u = np.exp(sigma * np.sqrt(self.dt))
        self.d = 1 / self.u
        self.p = (np.exp((r - q) * self.dt) - self.d) / (self.u - self.d)
        self.discount = np.exp(-r * self.dt)
        
    def _build_terminal_stock_prices(self):
        """Build stock prices at maturity"""
        n = self.n_steps
        return self.S * (self.u ** np.arange(n, -1, -1)) * (self.d ** np.arange(0, n + 1))
    
    def european_option_price(self, option_type="call"):
        """Price European option using backward induction"""
        n = self.n_steps
        stock_prices = self._build_terminal_stock_prices()
        
        if option_type.lower() == "call":
            option_values = np.maximum(stock_prices - self.K, 0)
        else:
            option_values = np.maximum(self.K - stock_prices, 0)
        
        for i in range(n - 1, -1, -1):
            option_values = self.discount * (self.p * option_values[:-1] + (1 - self.p) * option_values[1:])
        
        return option_values[0]
    
    def american_option_price(self, option_type="call"):
        """Price American option with early exercise"""
        n = self.n_steps
        
        # Build stock price tree
        stock_tree = np.zeros((n + 1, n + 1))
        stock_tree[0, 0] = self.S
        
        for i in range(1, n + 1):
            stock_tree[0:i, i] = stock_tree[0:i, i-1] * self.u
            stock_tree[i, i] = stock_tree[i-1, i-1] * self.d
        
        # Initialize option values at maturity
        option_tree = np.zeros((n + 1, n + 1))
        
        if option_type.lower() == "call":
            option_tree[:, n] = np.maximum(stock_tree[:, n] - self.K, 0)
        else:
            option_tree[:, n] = np.maximum(self.K - stock_tree[:, n], 0)
        
        # Backward induction with early exercise
        early_exercise_count = 0
        
        for i in range(n - 1, -1, -1):
            continuation = self.discount * (self.p * option_tree[0:i+1, i+1] + 
                                           (1 - self.p) * option_tree[1:i+2, i+1])
            
            if option_type.lower() == "call":
                exercise = np.maximum(stock_tree[0:i+1, i] - self.K, 0)
            else:
                exercise = np.maximum(self.K - stock_tree[0:i+1, i], 0)
            
            early_exercise_count += np.sum(exercise > continuation)
            option_tree[0:i+1, i] = np.maximum(continuation, exercise)
        
        return option_tree[0, 0], early_exercise_count
    
    def early_exercise_premium(self, option_type="call"):
        """Calculate early exercise premium"""
        american, _ = self.american_option_price(option_type)
        european = self.european_option_price(option_type)
        return american - european
    
    def calculate_greeks(self, option_type="call", american=True):
        """Calculate Greeks using finite differences"""
        if american:
            price_func = lambda m: m.american_option_price(option_type)[0]
        else:
            price_func = lambda m: m.european_option_price(option_type)
        
        base_price = price_func(self)
        
        # Delta
        dS = self.S * 0.01
        model_up = BinomialModel(self.S + dS, self.K, self.T, self.r, self.sigma, self.q, self.n_steps)
        model_down = BinomialModel(self.S - dS, self.K, self.T, self.r, self.sigma, self.q, self.n_steps)
        delta = (price_func(model_up) - price_func(model_down)) / (2 * dS)
        
        # Gamma
        gamma = (price_func(model_up) - 2 * base_price + price_func(model_down)) / (dS ** 2)
        
        # Theta
        if self.T > 1/365:
            dT = 1/365
            model_theta = BinomialModel(self.S, self.K, self.T - dT, self.r, self.sigma, self.q, self.n_steps)
            theta = price_func(model_theta) - base_price
        else:
            theta = 0
        
        # Vega
        d_sigma = 0.01
        model_vega_up = BinomialModel(self.S, self.K, self.T, self.r, self.sigma + d_sigma, self.q, self.n_steps)
        model_vega_down = BinomialModel(self.S, self.K, self.T, self.r, self.sigma - d_sigma, self.q, self.n_steps)
        vega = (price_func(model_vega_up) - price_func(model_vega_down)) / 2
        
        return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega}


# Calculate results
print(f"\nTree Steps: 500")
print("Calculating prices...")

binom_model = BinomialModel(S, K, T, r, sigma, q, n_steps=500)
binom_european_price = binom_model.european_option_price(OPTION_TYPE)
binom_american_price, early_exercise_nodes = binom_model.american_option_price(OPTION_TYPE)
early_exercise_premium = binom_model.early_exercise_premium(OPTION_TYPE)

print("Calculating Greeks...")
binom_greeks = binom_model.calculate_greeks(OPTION_TYPE, american=True)

# Display results
print("\nTree Parameters:")
print("-" * 40)
print(f"Up Factor (u):           {binom_model.u:.6f}")
print(f"Down Factor (d):         {binom_model.d:.6f}")
print(f"Risk-Neutral Prob (p):   {binom_model.p:.6f}")

print("\nPricing Results:")
print("-" * 40)
print(f"European Price:          ${binom_european_price:.4f}")
print(f"American Price:          ${binom_american_price:.4f}")
print(f"Early Exercise Premium:  ${early_exercise_premium:.4f}")

if early_exercise_premium > 0.01:
    print(f"\nNote: Significant early exercise value detected.")
    print(f"      Exercise nodes in tree: {early_exercise_nodes:,}")

print("\nGreeks (American):")
print("-" * 40)
print(f"Delta:    {binom_greeks['delta']:>12.4f}")
print(f"Gamma:    {binom_greeks['gamma']:>12.4f}")
print(f"Theta:    ${binom_greeks['theta']:>11.4f}/day")
print(f"Vega:     ${binom_greeks['vega']:>11.4f}/1%")
print("-" * 40)
print("\n" + "=" * 60)
print("Binomial model complete. Proceed to Cell 6.")
print("=" * 60)


# In[14]:


"""
Cell 6: Summary Report and Visualization
Consolidated output with model comparison and charts
"""

# Dependency check
try:
    _ = S, K, T, r, sigma, q, OPTION_TYPE, TICKER, DAYS_TO_EXPIRY
    _ = bs_price, bs_greeks, bs_model
    _ = mc_european_price, mc_european_se, mc_asian_arith_price, mc_asian_arith_se
    _ = mc_asian_geo_price, mc_asian_geo_se, mc_lookback_price, mc_lookback_se
    _ = mc_barrier_price, mc_barrier_se, barrier_type
    _ = binom_european_price, binom_american_price, early_exercise_premium, binom_greeks
except NameError as e:
    raise RuntimeError(f"Error: Execute Cells 2-5 before running this cell. Missing: {e}")

from matplotlib.gridspec import GridSpec


def generate_report():
    """Generate comprehensive pricing report"""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print("\n" + "=" * 70)
    print("OPTION PRICING REPORT")
    print(f"Generated: {timestamp}")
    print("=" * 70)
    
    # Underlying Asset
    print("\nUNDERLYING ASSET")
    print("-" * 70)
    print(f"Ticker:                  {TICKER}")
    print(f"Spot Price:              ${S:.2f}")
    print(f"Historical Volatility:   {sigma*100:.2f}%")
    print(f"Dividend Yield:          {q*100:.2f}%")
    
    # Option Parameters
    print("\nOPTION PARAMETERS")
    print("-" * 70)
    print(f"Strike Price:            ${K:.2f}")
    print(f"Time to Expiry:          {DAYS_TO_EXPIRY} days ({T:.4f} years)")
    print(f"Option Type:             {OPTION_TYPE.upper()}")
    print(f"Risk-Free Rate:          {r*100:.2f}%")
    
    # Moneyness
    moneyness = S / K
    if OPTION_TYPE == "call":
        status = "ITM" if moneyness > 1.02 else ("OTM" if moneyness < 0.98 else "ATM")
    else:
        status = "ITM" if moneyness < 0.98 else ("OTM" if moneyness > 1.02 else "ATM")
    print(f"Moneyness:               {status} ({moneyness:.2%})")
    
    # Pricing Results
    print("\n" + "=" * 70)
    print("PRICING RESULTS")
    print("=" * 70)
    print(f"\n{'Model':<24} {'Style':<16} {'Price':>12} {'Std Error':>14}")
    print("-" * 70)
    print(f"{'Black-Scholes':<24} {'European':<16} ${bs_price:>10.4f} {'N/A':>14}")
    print(f"{'Binomial Tree':<24} {'European':<16} ${binom_european_price:>10.4f} {'N/A':>14}")
    print(f"{'Binomial Tree':<24} {'American':<16} ${binom_american_price:>10.4f} {'N/A':>14}")
    print(f"{'Monte Carlo':<24} {'European':<16} ${mc_european_price:>10.4f} +/-${mc_european_se:>8.4f}")
    print(f"{'Monte Carlo':<24} {'Asian (Arith)':<16} ${mc_asian_arith_price:>10.4f} +/-${mc_asian_arith_se:>8.4f}")
    print(f"{'Monte Carlo':<24} {'Asian (Geo)':<16} ${mc_asian_geo_price:>10.4f} +/-${mc_asian_geo_se:>8.4f}")
    print(f"{'Monte Carlo':<24} {'Lookback':<16} ${mc_lookback_price:>10.4f} +/-${mc_lookback_se:>8.4f}")
    print(f"{'Monte Carlo':<24} {'Barrier':<16} ${mc_barrier_price:>10.4f} +/-${mc_barrier_se:>8.4f}")
    print("-" * 70)
    
    # Model Accuracy
    print("\nMODEL CONVERGENCE")
    print("-" * 70)
    bs_binom_diff = abs(bs_price - binom_european_price)
    bs_mc_diff = abs(bs_price - mc_european_price)
    print(f"Black-Scholes vs Binomial:     ${bs_binom_diff:.6f} ({bs_binom_diff/bs_price*100:.4f}%)")
    print(f"Black-Scholes vs Monte Carlo:  ${bs_mc_diff:.6f} ({bs_mc_diff/bs_price*100:.4f}%)")
    
    # Greeks Comparison
    print("\n" + "=" * 70)
    print("GREEKS COMPARISON")
    print("=" * 70)
    print(f"\n{'Greek':<16} {'Black-Scholes':>18} {'Binomial (Amer)':>18}")
    print("-" * 54)
    print(f"{'Delta':<16} {bs_greeks['delta']:>18.4f} {binom_greeks['delta']:>18.4f}")
    print(f"{'Gamma':<16} {bs_greeks['gamma']:>18.4f} {binom_greeks['gamma']:>18.4f}")
    print(f"{'Theta/day':<16} ${bs_greeks['theta']:>17.4f} ${binom_greeks['theta']:>17.4f}")
    print(f"{'Vega/1%':<16} ${bs_greeks['vega']:>17.4f} ${binom_greeks['vega']:>17.4f}")
    print(f"{'Rho/1%':<16} ${bs_greeks['rho']:>17.4f} {'N/A':>18}")
    print("-" * 54)
    
    # Recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    
    if early_exercise_premium > 0.01:
        rec_price = binom_american_price
        rec_model = "Binomial Tree (American)"
        rec_reason = f"Early exercise premium: ${early_exercise_premium:.4f}"
    else:
        rec_price = bs_price
        rec_model = "Black-Scholes (European)"
        rec_reason = "No significant early exercise value"
    
    print(f"\nRecommended Price:  ${rec_price:.4f}")
    print(f"Model:              {rec_model}")
    print(f"Rationale:          {rec_reason}")
    
    print("\nAlternative Prices (Exotic Options):")
    print(f"  Asian (Arithmetic): ${mc_asian_arith_price:.4f}")
    print(f"  Asian (Geometric):  ${mc_asian_geo_price:.4f}")
    print(f"  Lookback:           ${mc_lookback_price:.4f}")
    print(f"  Barrier:            ${mc_barrier_price:.4f}")
    
    print("\n" + "=" * 70)
    
    return rec_price, rec_model


def generate_visualizations():
    """Generate analysis charts"""
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'{TICKER} {OPTION_TYPE.upper()} Option Analysis\n'
                 f'Strike: ${K:.2f} | Expiry: {DAYS_TO_EXPIRY} days | Spot: ${S:.2f}', 
                 fontsize=12, fontweight='bold', y=1.01)
    
    # 1. Price vs Spot
    ax1 = fig.add_subplot(gs[0, 0])
    spot_range = np.linspace(S * 0.7, S * 1.3, 50)
    bs_prices_spot = [BlackScholesModel(s, K, T, r, sigma, q).price(OPTION_TYPE) for s in spot_range]
    binom_prices_spot = [BinomialModel(s, K, T, r, sigma, q, 100).american_option_price(OPTION_TYPE)[0] for s in spot_range]
    
    ax1.plot(spot_range, bs_prices_spot, 'b-', linewidth=2, label='Black-Scholes')
    ax1.plot(spot_range, binom_prices_spot, 'r--', linewidth=2, label='Binomial (American)')
    ax1.axvline(S, color='green', linestyle=':', alpha=0.7, label=f'Spot: ${S:.2f}')
    ax1.axvline(K, color='orange', linestyle=':', alpha=0.7, label=f'Strike: ${K:.2f}')
    ax1.set_xlabel('Spot Price ($)')
    ax1.set_ylabel('Option Price ($)')
    ax1.set_title('Price Sensitivity to Spot')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Price vs Volatility
    ax2 = fig.add_subplot(gs[0, 1])
    vol_range = np.linspace(0.1, 0.8, 50)
    bs_prices_vol = [BlackScholesModel(S, K, T, r, v, q).price(OPTION_TYPE) for v in vol_range]
    
    ax2.plot(vol_range * 100, bs_prices_vol, 'b-', linewidth=2)
    ax2.axvline(sigma * 100, color='red', linestyle='--', alpha=0.7, label=f'Current: {sigma*100:.1f}%')
    ax2.fill_between(vol_range * 100, bs_prices_vol, alpha=0.2)
    ax2.set_xlabel('Volatility (%)')
    ax2.set_ylabel('Option Price ($)')
    ax2.set_title('Price Sensitivity to Volatility')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Time Decay
    ax3 = fig.add_subplot(gs[0, 2])
    if DAYS_TO_EXPIRY > 1:
        days_range = np.linspace(1, DAYS_TO_EXPIRY, 50)
        bs_prices_time = [BlackScholesModel(S, K, d/365, r, sigma, q).price(OPTION_TYPE) for d in days_range]
        
        ax3.plot(days_range, bs_prices_time, 'b-', linewidth=2)
        ax3.fill_between(days_range, bs_prices_time, alpha=0.3)
        ax3.axhline(bs_price, color='red', linestyle='--', alpha=0.5, label=f'Current: ${bs_price:.2f}')
    ax3.set_xlabel('Days to Expiry')
    ax3.set_ylabel('Option Price ($)')
    ax3.set_title('Time Decay')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Model Comparison
    ax4 = fig.add_subplot(gs[1, 0])
    models = ['BS', 'Binom\n(Euro)', 'Binom\n(Amer)', 'MC\n(Euro)', 'MC\n(Asian)']
    prices = [bs_price, binom_european_price, binom_american_price, mc_european_price, mc_asian_arith_price]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
    
    bars = ax4.bar(models, prices, color=colors, edgecolor='black', linewidth=1)
    ax4.set_ylabel('Option Price ($)')
    ax4.set_title('Model Comparison')
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, price in zip(bars, prices):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(prices)*0.02,
                f'${price:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 5. Greeks
    ax5 = fig.add_subplot(gs[1, 1])
    greek_names = ['Delta', 'Gamma (x10)', 'Vega']
    greek_bs = [bs_greeks['delta'], bs_greeks['gamma'] * 10, bs_greeks['vega']]
    greek_binom = [binom_greeks['delta'], binom_greeks['gamma'] * 10, binom_greeks['vega']]
    
    x = np.arange(len(greek_names))
    width = 0.35
    
    ax5.bar(x - width/2, greek_bs, width, label='Black-Scholes', color='#3498db')
    ax5.bar(x + width/2, greek_binom, width, label='Binomial', color='#e74c3c')
    
    ax5.set_ylabel('Value')
    ax5.set_title('Greeks Comparison')
    ax5.set_xticks(x)
    ax5.set_xticklabels(greek_names)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Payoff Diagram
    ax6 = fig.add_subplot(gs[1, 2])
    spot_payoff = np.linspace(K * 0.5, K * 1.5, 100)
    
    if OPTION_TYPE == "call":
        intrinsic = np.maximum(spot_payoff - K, 0)
        option_value = [BlackScholesModel(s, K, T, r, sigma, q).call_price() for s in spot_payoff]
    else:
        intrinsic = np.maximum(K - spot_payoff, 0)
        option_value = [BlackScholesModel(s, K, T, r, sigma, q).put_price() for s in spot_payoff]
    
    ax6.plot(spot_payoff, intrinsic, 'r--', linewidth=2, label='Intrinsic Value')
    ax6.plot(spot_payoff, option_value, 'b-', linewidth=2, label=f'Option Value (T={DAYS_TO_EXPIRY}d)')
    ax6.fill_between(spot_payoff, intrinsic, option_value, alpha=0.2, color='green', label='Time Value')
    ax6.axvline(S, color='green', linestyle=':', alpha=0.7, label=f'Spot: ${S:.2f}')
    ax6.axvline(K, color='orange', linestyle=':', alpha=0.7, label=f'Strike: ${K:.2f}')
    ax6.set_xlabel('Stock Price ($)')
    ax6.set_ylabel('Value ($)')
    ax6.set_title('Payoff Diagram')
    ax6.legend(fontsize=8, loc='upper left')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def export_results():
    """Export results to DataFrames"""
    
    pricing_df = pd.DataFrame({
        'Model': ['Black-Scholes', 'Binomial (European)', 'Binomial (American)', 
                  'Monte Carlo (European)', 'Monte Carlo (Asian Arith)', 
                  'Monte Carlo (Asian Geo)', 'Monte Carlo (Lookback)', 'Monte Carlo (Barrier)'],
        'Style': ['European', 'European', 'American', 'European', 
                  'Asian', 'Asian', 'Lookback', 'Barrier'],
        'Price': [bs_price, binom_european_price, binom_american_price, 
                  mc_european_price, mc_asian_arith_price, mc_asian_geo_price,
                  mc_lookback_price, mc_barrier_price],
        'Std_Error': [None, None, None, mc_european_se, mc_asian_arith_se, 
                      mc_asian_geo_se, mc_lookback_se, mc_barrier_se]
    })
    
    greeks_df = pd.DataFrame({
        'Greek': ['Delta', 'Gamma', 'Theta', 'Vega', 'Rho'],
        'Black_Scholes': [bs_greeks['delta'], bs_greeks['gamma'], 
                          bs_greeks['theta'], bs_greeks['vega'], bs_greeks['rho']],
        'Binomial': [binom_greeks['delta'], binom_greeks['gamma'], 
                     binom_greeks['theta'], binom_greeks['vega'], None]
    })
    
    return pricing_df, greeks_df


# Execute
print("\nGenerating report...")
rec_price, rec_model = generate_report()

print("\nGenerating visualizations...")
generate_visualizations()

print("\nExporting data...")
pricing_df, greeks_df = export_results()

print("\n" + "=" * 70)
print("EXPORTED DATA")
print("=" * 70)
print("\nPricing Results:")
print(pricing_df.to_string(index=False))
print("\nGreeks:")
print(greeks_df.to_string(index=False))

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print("\nAvailable objects for further analysis:")
print("  - pricing_df: Pricing results DataFrame")
print("  - greeks_df: Greeks DataFrame")
print("  - bs_model: Black-Scholes model instance")
print("  - binom_model: Binomial model instance")
print("  - mc_model: Monte Carlo model instance")
print("\nTo run a new analysis, return to Cell 2.")
print("=" * 70)


# In[ ]:




