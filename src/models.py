# src/models.py

import pandas as pd
import numpy as np
from scipy.stats import norm, t
# import streamlit as st # REMOVE THIS LINE

# Assuming calculate_metrics is in analysis.py and calculate_returns is in utils.py
from .analysis import calculate_metrics
from .utils import calculate_returns

def calculate_var_historical(returns, confidence_level=0.99):
    """Calcula el VaR Histórico."""
    if returns is None or returns.empty:
        return np.nan
    return returns.quantile(1 - confidence_level) # Pérdida, por lo que es el cuantil (1-alpha)

def calculate_es_historical(returns, confidence_level=0.99):
    """Calcula el Expected Shortfall (CVaR) Histórico."""
    if returns is None or returns.empty:
        return np.nan
    var_hist = calculate_var_historical(returns, confidence_level)
    if pd.isna(var_hist): return np.nan
    return returns[returns <= var_hist].mean()

def calculate_var_parametric(returns, confidence_level=0.99, dist_type='normal'):
    """Calcula el VaR Paramétrico (Normal o t-Student)."""
    if returns is None or returns.empty or len(returns) < 2: # Necesita al menos 2 para std
        return np.nan
    mu = returns.mean()
    sigma = returns.std()
    if pd.isna(mu) or pd.isna(sigma) or sigma == 0: return np.nan

    if dist_type == 'normal':
        return norm.ppf(1 - confidence_level, loc=mu, scale=sigma)
    elif dist_type == 't':
        # Asegurar que no hay NaNs antes de t.fit
        returns_cleaned_for_tfit = returns.dropna()
        if len(returns_cleaned_for_tfit) < 3 : return np.nan # t.fit necesita al menos 3 puntos
        df, _, _ = t.fit(returns_cleaned_for_tfit)
        return t.ppf(1 - confidence_level, df=df, loc=mu, scale=sigma)
    return np.nan

def calculate_es_parametric(returns, confidence_level=0.99, dist_type='normal'):
    """Calcula el Expected Shortfall (CVaR) Paramétrico."""
    if returns is None or returns.empty or len(returns) < 2:
        return np.nan
    mu = returns.mean()
    sigma = returns.std()
    if pd.isna(mu) or pd.isna(sigma) or sigma == 0: return np.nan

    alpha = 1 - confidence_level
    if dist_type == 'normal':
        return mu - sigma * (norm.pdf(norm.ppf(alpha)) / alpha)
    elif dist_type == 't':
        returns_cleaned_for_tfit = returns.dropna()
        if len(returns_cleaned_for_tfit) < 3 : return np.nan
        df, _, _ = t.fit(returns_cleaned_for_tfit)
        x_alpha = t.ppf(alpha, df=df)
        if (df - 1) <= 0: return np.nan # Evitar división por cero o df inválido
        return mu - sigma * ( (df + x_alpha**2) / (df - 1) ) * (t.pdf(x_alpha, df=df) / alpha)
    return np.nan

def apply_hypothetical_shock(portfolio_value_series, shock_percentage):
    """Aplica un shock porcentual al último valor de la cartera."""
    if portfolio_value_series is None or portfolio_value_series.empty:
        return np.nan, np.nan
    last_value = portfolio_value_series.iloc[-1]
    shocked_value = last_value * (1 + shock_percentage / 100.0)
    loss_value = last_value - shocked_value
    return shocked_value, loss_value

def analyze_historical_scenario(portfolio_returns, all_asset_prices, weights_dict, scenario_dates, initial_investment_for_scenario=10000):
    """
    Simula el rendimiento de la cartera actual (con sus pesos fijos) durante un periodo histórico de crisis.
    """
    if portfolio_returns is None or all_asset_prices is None or not weights_dict or scenario_dates is None:
        return None, {}

    scenario_start, scenario_end = scenario_dates
    scenario_data = all_asset_prices.loc[scenario_start:scenario_end]

    if scenario_data.empty or scenario_data.shape[0] < 2:
        # st.warning(f"No hay suficientes datos para el escenario histórico seleccionado ({scenario_start.date()} - {scenario_end.date()}).") # REMOVED
        return None, {}

    asset_columns_original = list(weights_dict.keys())
    asset_columns_in_scenario = [col for col in asset_columns_original if col in scenario_data.columns]

    if not asset_columns_in_scenario:
        # st.error(f"Ninguno de los activos de la cartera ({', '.join(asset_columns_original)}) tiene datos para el escenario histórico seleccionado.") # REMOVED
        return None, {}

    scenario_prices = scenario_data[asset_columns_in_scenario].copy().ffill().bfill()

    weights_dict_scenario = {k: weights_dict[k] for k in asset_columns_in_scenario if k in weights_dict}
    current_total_weight_scenario = sum(weights_dict_scenario.values())
    if current_total_weight_scenario > 1e-6:
        weights_dict_scenario = {k: v / current_total_weight_scenario for k, v in weights_dict_scenario.items()}
    else:
        # st.error("Los pesos de los activos disponibles en el escenario suman cero. No se puede simular.") # REMOVED
        return None, {}

    if scenario_prices.isnull().values.any():
        # st.warning(f"Faltan datos para algunos activos en el escenario histórico incluso después de ffill/bfill. Resultados pueden ser imprecisos.") # REMOVED
        pass # handle message in app.py

    scenario_portfolio_value = pd.Series(index=scenario_prices.index, dtype=float)
    if scenario_prices.empty: return None, {}
    scenario_portfolio_value.iloc[0] = initial_investment_for_scenario

    shares = {}
    for fund in asset_columns_in_scenario:
        initial_price = scenario_prices[fund].iloc[0]
        if pd.notna(initial_price) and initial_price != 0:
            shares[fund] = (initial_investment_for_scenario * weights_dict_scenario.get(fund, 0)) / initial_price
        else:
            shares[fund] = 0

    for i in range(1, len(scenario_prices)):
        current_val = 0
        for fund in asset_columns_in_scenario:
            current_price = scenario_prices[fund].iloc[i]
            if pd.notna(current_price):
                current_val += shares.get(fund, 0) * current_price
            else:
                prev_price = scenario_prices[fund].iloc[i-1]
                if pd.notna(prev_price):
                    current_val += shares.get(fund, 0) * prev_price
        scenario_portfolio_value.iloc[i] = current_val

    scenario_portfolio_returns = scenario_portfolio_value.pct_change().dropna()

    metrics_scenario = calculate_metrics(scenario_portfolio_value, scenario_portfolio_returns, risk_free_rate_annual=0.0)
    return scenario_portfolio_value, metrics_scenario

def run_monte_carlo_simulation(portfolio_returns, initial_investment, num_simulations, num_days_projection):
    """Ejecuta una simulación de Montecarlo para el valor futuro de la cartera."""
    if portfolio_returns is None or portfolio_returns.empty or len(portfolio_returns) < 20:
        # st.warning("Insuficientes retornos históricos para una simulación de Montecarlo fiable.") # REMOVED
        return None, None

    mean_return = portfolio_returns.mean()
    std_dev = portfolio_returns.std()

    if pd.isna(mean_return) or pd.isna(std_dev) or std_dev == 0:
        # st.warning("Media o desviación estándar de retornos no válida para Montecarlo.") # REMOVED
        return None, None

    simulations_df = pd.DataFrame()

    for i in range(num_simulations):
        daily_returns_sim = np.random.normal(mean_return, std_dev, num_days_projection)
        price_series = [initial_investment]
        for r in daily_returns_sim:
            price_series.append(price_series[-1] * (1 + r))
        simulations_df[f'Sim {i+1}'] = price_series[1:]

    last_date = portfolio_returns.index[-1] if not portfolio_returns.index.empty else pd.to_datetime('today')
    sim_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_days_projection, freq='B')

    if len(sim_index) == simulations_df.shape[0]:
        simulations_df.index = sim_index
    else:
        simulations_df.index = pd.RangeIndex(start=0, stop=len(simulations_df))

    final_values = simulations_df.iloc[-1]
    results_summary = {
        'mean': final_values.mean(),
        'median': final_values.median(),
        'q5': final_values.quantile(0.05),
        'q95': final_values.quantile(0.95),
        'min': final_values.min(),
        'max': final_values.max()
    }
    return simulations_df, results_summary