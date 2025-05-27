# src/analysis.py

import pandas as pd
import numpy as np
import empyrical # Keep empyrical for beta and alpha
# import streamlit as st # REMOVE THIS LINE

# Assuming calculate_sortino_ratio is in utils.py
from .utils import calculate_sortino_ratio, calculate_returns

def calculate_metrics(portfolio_value, portfolio_returns, risk_free_rate_annual=0.0):
    """Calcula las métricas de rendimiento de la cartera."""
    if portfolio_returns is None or portfolio_value is None or portfolio_returns.empty or portfolio_value.empty:
        return {}
    metrics = {}
    start_value = portfolio_value.iloc[0]
    end_value = portfolio_value.iloc[-1]
    total_return = (end_value / start_value) - 1 if start_value != 0 else 0
    metrics['Rentabilidad Total'] = total_return

    years = (portfolio_value.index[-1] - portfolio_value.index[0]).days / 365.25
    if years <= 0 or start_value <= 0: cagr = 0
    else:
        base = end_value / start_value
        cagr = (base ** (1 / years)) - 1 if base > 0 else -1.0 # Manejar caso de pérdida total
    metrics['Rentabilidad Anualizada (CAGR)'] = cagr

    volatility = portfolio_returns.std() * np.sqrt(252) if not portfolio_returns.empty else 0
    metrics['Volatilidad Anualizada'] = volatility

    sharpe_ratio = (cagr - risk_free_rate_annual) / volatility if volatility != 0 else np.nan
    metrics['Ratio de Sharpe'] = sharpe_ratio

    rolling_max = portfolio_value.cummax()
    daily_drawdown = portfolio_value / rolling_max - 1
    max_drawdown = daily_drawdown.min() if not daily_drawdown.empty else 0
    metrics['Máximo Drawdown'] = max_drawdown
    return metrics

def calculate_individual_metrics(fund_prices, risk_free_rate_annual=0.0):
    """Calcula métricas clave para cada fondo individual."""
    if fund_prices is None or fund_prices.empty: return pd.DataFrame()
    fund_prices = fund_prices.apply(pd.to_numeric, errors='coerce').ffill().bfill()
    fund_returns = fund_prices.pct_change().dropna(how='all')

    individual_metrics = {}
    for fund_name in fund_prices.columns:
        fund_series = fund_prices[fund_name].dropna()
        fund_ret_series = fund_returns[fund_name].dropna() if fund_name in fund_returns.columns else pd.Series(dtype=float)

        if fund_series.empty or fund_series.shape[0] < 2 or fund_series.iloc[0] == 0 or fund_ret_series.empty:
            metrics_nan = {'Rentabilidad Total': np.nan, 'Rentabilidad Anualizada (CAGR)': np.nan,
                           'Volatilidad Anualizada': np.nan, 'Ratio de Sharpe': np.nan,
                           'Máximo Drawdown': np.nan, 'Sortino Ratio': np.nan}
            individual_metrics[fund_name] = metrics_nan
            continue

        metrics = calculate_metrics(fund_series, fund_ret_series, risk_free_rate_annual)

        daily_rf_for_individual_sortino = (1 + risk_free_rate_annual)**(1/252) - 1 if risk_free_rate_annual != 0 else 0.0
        metrics['Sortino Ratio'] = calculate_sortino_ratio(fund_ret_series, daily_required_return_for_empyrical=daily_rf_for_individual_sortino)
        individual_metrics[fund_name] = metrics
    return pd.DataFrame(individual_metrics)


def calculate_covariance_matrix(returns):
    if returns is None or returns.empty: return pd.DataFrame()
    return returns.cov() * 252

def calculate_correlation_matrix(returns):
    if returns is None or returns.empty: return pd.DataFrame()
    return returns.corr()

def calculate_rolling_correlation(returns, window, pair_list=None):
    if returns is None or returns.empty or returns.shape[1] < 2 or window <= 0 or window > returns.shape[0]:
        return None, None
    rolling_corr_pairs = None
    if pair_list:
        rolling_corr_pairs = pd.DataFrame(index=returns.index[window-1:])
        for pair in pair_list:
            if len(pair) == 2 and pair[0] in returns.columns and pair[1] in returns.columns:
                col1, col2 = pair
                try:
                    rolling_corr_pairs.loc[:, f'{col1} vs {col2}'] = returns[col1].rolling(window=window).corr(returns[col2])
                except Exception as e:
                    # st.warning(f"Corr rodante {col1} vs {col2} falló: {e}") # REMOVED
                    pass # handle message in app.py
        rolling_corr_pairs = rolling_corr_pairs.dropna(how='all')

    rolling_corr_avg = None
    try:
        rolling_corr_matrices = returns.rolling(window=window).corr()
        if not rolling_corr_matrices.empty:
            avg_corrs = []
            for date_idx in rolling_corr_matrices.index.get_level_values(0).unique():
                matrix_slice = rolling_corr_matrices.loc[date_idx]
                if matrix_slice.shape[0] > 1:
                    avg_corrs.append(matrix_slice.values[np.triu_indices_from(matrix_slice.values, k=1)].mean())
                else:
                    avg_corrs.append(np.nan)

            if avg_corrs:
                rolling_corr_avg = pd.Series(avg_corrs, index=rolling_corr_matrices.index.get_level_values(0).unique())
                rolling_corr_avg.name = "Correlación Promedio Rodante"
                rolling_corr_avg = rolling_corr_avg.dropna()
    except Exception as e:
        # st.warning(f"Corr promedio rodante falló: {e}") # REMOVED
        pass # handle message in app.py
    return rolling_corr_avg, rolling_corr_pairs

def calculate_risk_contribution(returns, weights_dict):
    if returns is None or returns.empty or not weights_dict or returns.isnull().values.any():
        return pd.Series(dtype=float)

    funds = list(weights_dict.keys())
    funds_in_returns = [f for f in funds if f in returns.columns]
    if not funds_in_returns: return pd.Series(dtype=float)

    weights_dict_adj = {f: weights_dict[f] for f in funds_in_returns}
    funds_adj = funds_in_returns
    weights_adj = np.array([weights_dict_adj[f] for f in funds_adj])

    total_weight_adj = weights_adj.sum()
    if total_weight_adj <= 1e-9: return pd.Series(dtype=float) # Evitar división por cero
    weights_norm_adj = weights_adj / total_weight_adj # Normalizar pesos

    returns_subset = returns[funds_adj].dropna() # Asegurar que no hay NaNs en los retornos usados
    if returns_subset.shape[0] < 2: return pd.Series(dtype=float) # Necesita al menos 2 puntos para covarianza

    try:
        cov_matrix = returns_subset.cov() * 252
        if cov_matrix.isnull().values.any(): return pd.Series(dtype=float)

        portfolio_var = weights_norm_adj.T @ cov_matrix @ weights_norm_adj
        if portfolio_var <= 1e-10: # Si la varianza es casi cero, la contribución es proporcional al peso
            return pd.Series({fund: weights_norm_adj[i] for i, fund in enumerate(funds_adj)}, name="Contribución al Riesgo (%)")

        portfolio_vol = np.sqrt(portfolio_var)

        # Contribución Marginal al Riesgo (MCTR) = Cov(Ri, Rp) / Sigma_p = (Cov_matrix @ w) / Sigma_p
        mctr = (cov_matrix.values @ weights_norm_adj) / portfolio_vol

        # Contribución Total al Riesgo (CCTR) = w_i * MCTR_i
        cctr = weights_norm_adj * mctr

        # Contribución Porcentual al Riesgo = CCTR_i / Sigma_p (o CCTR_i / sum(CCTR) que es Sigma_p^2 / Sigma_p)
        risk_contribution_percent = cctr / portfolio_vol

        risk_contribution_series = pd.Series(risk_contribution_percent, index=funds_adj, name="Contribución al Riesgo (%)")
        return risk_contribution_series.fillna(0.0)
    except Exception as e:
        # st.error(f"Error en contribución al riesgo: {e}") # REMOVED
        return pd.Series(dtype=float)

def calculate_rolling_metrics(portfolio_returns, window, annual_risk_free_rate=0.0): # Renombrado para claridad
    if portfolio_returns is None or portfolio_returns.empty or window <= 1 or window > portfolio_returns.shape[0]:
        return None, None, None

    rolling_vol = (portfolio_returns.rolling(window=window).std(ddof=1) * np.sqrt(252)).dropna()
    rolling_vol.name = f"Volatilidad Rodante ({window}d)"

    # Rentabilidad anualizada rodante
    rolling_annual_ret = portfolio_returns.rolling(window=window).mean() * 252

    rolling_sharpe = ((rolling_annual_ret - annual_risk_free_rate) / rolling_vol).replace([np.inf, -np.inf], np.nan).dropna()
    rolling_sharpe.name = f"Sharpe Rodante ({window}d)"

    # CORRECCIÓN: Calcular tasa diaria para Sortino aquí
    daily_rf_for_rolling_sortino = (1 + annual_risk_free_rate)**(1/252) - 1 if annual_risk_free_rate != 0 else 0.0
    rolling_sortino = portfolio_returns.rolling(window=window).apply(
        lambda x: calculate_sortino_ratio(x, daily_required_return_for_empyrical=daily_rf_for_rolling_sortino),
        raw=True
    ).dropna()
    rolling_sortino.name = f"Sortino Rodante ({window}d)"

    return rolling_vol, rolling_sharpe, rolling_sortino

def calculate_benchmark_metrics(portfolio_returns, benchmark_returns, risk_free_rate=0.0):
    metrics = {}
    if portfolio_returns is None or benchmark_returns is None or portfolio_returns.empty or benchmark_returns.empty:
        return metrics
    try:
        common_index = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common_index) < 2: return {}

        portfolio_returns_aligned = portfolio_returns.loc[common_index].replace([np.inf, -np.inf], np.nan).dropna()
        benchmark_returns_aligned = benchmark_returns.loc[common_index].replace([np.inf, -np.inf], np.nan).dropna()

        # Re-alinear después de dropear NaNs individuales
        common_index_final = portfolio_returns_aligned.index.intersection(benchmark_returns_aligned.index)
        if len(common_index_final) < 2: return {}

        portfolio_returns_aligned = portfolio_returns_aligned.loc[common_index_final]
        benchmark_returns_aligned = benchmark_returns_aligned.loc[common_index_final]

        # Empyrical espera la tasa libre de riesgo diaria para algunas funciones
        daily_rf = (1 + risk_free_rate)**(1/252) - 1 if risk_free_rate != 0 else 0.0

        metrics['Beta'] = empyrical.beta(portfolio_returns_aligned, benchmark_returns_aligned, risk_free=daily_rf)
        metrics['Alpha (anual)'] = empyrical.alpha(portfolio_returns_aligned, benchmark_returns_aligned, risk_free=daily_rf, annualization=252) # Empyrical anualiza

        excess_returns = portfolio_returns_aligned - benchmark_returns_aligned
        metrics['Tracking Error (anual)'] = excess_returns.std() * np.sqrt(252)

        alpha_val, te_val = metrics.get('Alpha (anual)'), metrics.get('Tracking Error (anual)')
        if alpha_val is not None and te_val is not None and not pd.isna(alpha_val) and not pd.isna(te_val) and te_val != 0:
            metrics['Information Ratio'] = alpha_val / te_val
        else: metrics['Information Ratio'] = np.nan

    except Exception as e:
        # st.error(f"Error métricas vs benchmark: {e}") # REMOVED
        pass # handle message in app.py
    return metrics

def calculate_rolling_beta(portfolio_returns, benchmark_returns, window):
    if portfolio_returns is None or benchmark_returns is None or portfolio_returns.empty or benchmark_returns.empty: return None

    common_index = portfolio_returns.index.intersection(benchmark_returns.index)
    if window <= 1 or window > len(common_index): return None

    try:
        portfolio_returns_aligned = portfolio_returns.loc[common_index]
        benchmark_returns_aligned = benchmark_returns.loc[common_index]

        df_combined = pd.DataFrame({'portfolio': portfolio_returns_aligned, 'benchmark': benchmark_returns_aligned})

        rolling_cov = df_combined['portfolio'].rolling(window=window).cov(df_combined['benchmark'])
        rolling_var = df_combined['benchmark'].rolling(window=window).var()

        rolling_beta = (rolling_cov / rolling_var).replace([np.inf, -np.inf], np.nan).dropna()
        rolling_beta.name = f"Beta Rodante ({window}d)"
        return rolling_beta
    except Exception as e:
        # st.error(f"Error beta rodante: {e}") # REMOVED
        return None

def calculate_diversification_ratio(asset_returns, weights_dict):
    if asset_returns is None or asset_returns.empty or not weights_dict: return np.nan

    funds_in_returns = [f for f in weights_dict.keys() if f in asset_returns.columns]
    if not funds_in_returns: return np.nan

    weights_adj = np.array([weights_dict[f] for f in funds_in_returns])
    total_w = weights_adj.sum()
    if total_w <= 1e-9 : return np.nan
    weights_norm = weights_adj / total_w

    asset_returns_subset = asset_returns[funds_in_returns].dropna() # Usar retornos limpios
    if asset_returns_subset.shape[0] < 2: return np.nan

    asset_vols = asset_returns_subset.std() * np.sqrt(252)
    if asset_vols.isnull().any() or (asset_vols == 0).any(): return np.nan # Evitar problemas con vol cero

    weighted_avg_vol = np.sum(weights_norm * asset_vols)
    if weighted_avg_vol <= 1e-9: return np.nan # Evitar división por cero o resultados extraños

    cov_matrix = asset_returns_subset.cov() * 252
    if cov_matrix.isnull().values.any(): return np.nan

    portfolio_var = weights_norm.T @ cov_matrix @ weights_norm
    if portfolio_var <= 1e-10: return np.nan # Varianza de cartera casi cero
    portfolio_vol = np.sqrt(portfolio_var)

    if portfolio_vol <= 1e-9: return np.nan # Volatilidad de cartera casi cero

    return weighted_avg_vol / portfolio_vol