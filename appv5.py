# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
import locale
import copy
import empyrical # Para métricas avanzadas

# Para métricas de riesgo y simulaciones
from scipy.stats import norm, t

# Para optimización (ya presente)
import pypfopt
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions
from pypfopt import plotting # Para helpers de ploteo si son necesarios
import cvxpy as cp # Para definir restricciones


# --- Configuración de la Página y Estilo ---
st.set_page_config(
    page_title="Backtester Quant v5.0 (Extendido)",
    page_icon="🚀",
    layout="wide"
)

# --- Constantes y Configuraciones ---
RISK_FREE_RATE_DEFAULT = 0.0
CONFIDENCE_LEVEL_VAR_DEFAULT = 0.99 # 99% para VaR/ES
HISTORICAL_SCENARIOS = {
    "Ninguno": None,
    "Crisis Financiera Global (Sep 2008 - Mar 2009)": (pd.to_datetime("2008-09-01"), pd.to_datetime("2009-03-31")),
    "Lunes Negro (Oct 1987)": (pd.to_datetime("1987-10-01"), pd.to_datetime("1987-10-31")),
    "Burbuja .com (Mar 2000 - Sep 2001)": (pd.to_datetime("2000-03-01"), pd.to_datetime("2001-09-30")),
    "COVID-19 Crash (Feb 2020 - Mar 2020)": (pd.to_datetime("2020-02-15"), pd.to_datetime("2020-03-31")),
    "Crisis Deuda Europea (May 2010 - Oct 2011)": (pd.to_datetime("2010-05-01"), pd.to_datetime("2011-10-31")),
}

# --- Funciones Principales ---

def load_data(uploaded_file):
    """Carga y procesa el archivo CSV subido por el usuario."""
    if uploaded_file is not None:
        try:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            # Intentar detectar separador y decimal
            try: # Coma decimal, punto y coma separador
                data = pd.read_csv(stringio, sep=';', decimal=',', parse_dates=[0], index_col=0)
            except (pd.errors.ParserError, ValueError, IndexError, KeyError):
                stringio.seek(0)
                try: # Coma decimal, coma separador
                    data = pd.read_csv(stringio, sep=',', decimal=',', parse_dates=[0], index_col=0)
                except (pd.errors.ParserError, ValueError, IndexError, KeyError):
                    stringio.seek(0)
                    try: # Punto decimal, punto y coma separador
                        data = pd.read_csv(stringio, sep=';', decimal='.', parse_dates=[0], index_col=0)
                    except (pd.errors.ParserError, ValueError, IndexError, KeyError):
                        stringio.seek(0) # Punto decimal, coma separador
                        data = pd.read_csv(stringio, sep=',', decimal='.', parse_dates=[0], index_col=0)

            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)

            data.dropna(axis=0, how='all', inplace=True)
            data.sort_index(inplace=True)
            data.ffill(inplace=True)
            data.bfill(inplace=True)

            for col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    try:
                        if data[col].dtype == 'object':
                            cleaned_col = data[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
                            data[col] = pd.to_numeric(cleaned_col)
                        else:
                            data[col] = pd.to_numeric(data[col])
                    except (ValueError, AttributeError, TypeError) as e_conv:
                        st.error(f"Error convirtiendo columna '{col}' a numérico: {e_conv}. Verifica el formato.")
                        return None
            return data
        except Exception as e:
            st.error(f"Error crítico al procesar el archivo CSV: {e}")
            st.error("Verifica el formato: 1ª col Fecha, siguientes ISINs/Benchmark; sep CSV (',' o ';'); decimal (',' o '.').")
            return None
    return None

def run_backtest(data, weights_dict, initial_investment, start_date, end_date, rebalance_freq, transaction_cost_bps=0.0):
    """Ejecuta la simulación del backtesting para la cartera, incluyendo costes de transacción."""
    asset_columns = list(weights_dict.keys())
    start_date_ts = pd.to_datetime(start_date)
    end_date_ts = pd.to_datetime(end_date)

    data_in_range = data.loc[data.index.intersection(pd.date_range(start_date_ts, end_date_ts))]
    if data_in_range.empty:
        st.warning("No hay datos en el índice para el rango de fechas seleccionado.")
        return None, None, 0.0 # portfolio_value, portfolio_returns, total_transaction_costs

    prices = data_in_range[asset_columns].copy()

    if prices.empty or prices.isnull().all().all():
        st.warning("No hay datos válidos para los activos en el rango de fechas seleccionado.")
        return None, None, 0.0

    prices.ffill(inplace=True)
    prices.bfill(inplace=True)

    if prices.iloc[0].isnull().any():
        missing_funds = prices.columns[prices.iloc[0].isnull()].tolist()
        st.warning(f"Faltan datos iniciales para: {', '.join(missing_funds)} en {prices.index[0].date()}.")
        first_valid_date = prices.dropna(axis=0, how='any').index.min()
        if pd.isna(first_valid_date) or first_valid_date > end_date_ts:
            st.error("No hay datos comunes suficientes para iniciar el backtest en el rango seleccionado.")
            return None, None, 0.0
        prices = prices.loc[first_valid_date:].copy()
        st.warning(f"Backtest comenzará en {prices.index[0].date()}.")
        if prices.empty: return None, None, 0.0

    total_weight = sum(weights_dict.values())
    if not np.isclose(total_weight, 1.0) and total_weight != 0:
        weights_dict = {k: v / total_weight for k, v in weights_dict.items()}

    portfolio_value = pd.Series(index=prices.index, dtype=float)
    if not prices.empty:
        portfolio_value.loc[prices.index[0]] = initial_investment
    else:
        st.error("No se pueden inicializar los valores de la cartera.")
        return None, None, 0.0

    current_weights = weights_dict.copy()
    last_rebalance_date = prices.index[0]
    
    # Costes de transacción
    total_transaction_costs_value = 0.0
    transaction_cost_rate = transaction_cost_bps / 10000.0 # Convertir bps a decimal

    # Asignación inicial y cálculo de participaciones
    initial_alloc_value = {}
    shares = {}
    initial_turnover = 0 # Para el coste de la primera asignación
    
    for fund in current_weights:
        initial_price = prices[fund].iloc[0]
        if pd.notna(initial_price) and initial_price != 0:
            target_value_fund = initial_investment * current_weights[fund]
            initial_alloc_value[fund] = target_value_fund
            shares[fund] = target_value_fund / initial_price
            initial_turnover += target_value_fund # El "turnover" inicial es el valor total invertido
        else:
            shares[fund] = 0
            initial_alloc_value[fund] = 0
            st.warning(f"Precio inicial inválido para {fund} en {prices.index[0].date()}. Peso inicial no aplicado.")
    
    cost_initial_alloc = initial_turnover * transaction_cost_rate
    total_transaction_costs_value += cost_initial_alloc
    portfolio_value.loc[prices.index[0]] -= cost_initial_alloc # Deducir coste de la inversión inicial

    rebalance_offset = {'Mensual': pd.DateOffset(months=1), 'Trimestral': pd.DateOffset(months=3),
                        'Anual': pd.DateOffset(years=1), 'No Rebalancear': None}
    offset = rebalance_offset[rebalance_freq]

    for i in range(1, len(prices)):
        current_date = prices.index[i]
        prev_date = prices.index[i-1]
        
        # Calcular valor de cartera antes de rebalanceo
        current_portfolio_value_before_rebal = 0.0
        for fund in shares:
            current_price = prices.loc[current_date, fund] if current_date in prices.index and pd.notna(prices.loc[current_date, fund]) else \
                            prices.loc[prev_date, fund] if prev_date in prices.index and pd.notna(prices.loc[prev_date, fund]) else np.nan
            
            if pd.notna(current_price):
                current_portfolio_value_before_rebal += shares[fund] * current_price
            else: # Si no hay precio actual ni previo, usar el valor de cartera del día anterior
                current_portfolio_value_before_rebal = portfolio_value.loc[prev_date] if prev_date in portfolio_value else 0
                st.warning(f"Usando valor de cartera previo para {current_date} debido a precio faltante para {fund}")
                break 
        
        if pd.isna(current_portfolio_value_before_rebal) or (current_portfolio_value_before_rebal == 0 and prev_date in portfolio_value and portfolio_value.loc[prev_date] != 0):
             current_portfolio_value_before_rebal = portfolio_value.loc[prev_date] if prev_date in portfolio_value and pd.notna(portfolio_value.loc[prev_date]) else 0


        portfolio_value.loc[current_date] = current_portfolio_value_before_rebal

        # Lógica de Rebalanceo
        if offset and current_date >= last_rebalance_date + offset:
            if pd.notna(current_portfolio_value_before_rebal) and current_portfolio_value_before_rebal > 0:
                turnover_rebalance = 0
                new_shares = {}
                
                # Calcular valor actual de cada activo antes de rebalancear
                current_asset_values = {}
                for fund in shares:
                    price_at_rebal = prices.loc[current_date, fund] if current_date in prices.index and pd.notna(prices.loc[current_date, fund]) else np.nan
                    if pd.notna(price_at_rebal):
                         current_asset_values[fund] = shares[fund] * price_at_rebal
                    else: # Si falta precio en rebalanceo, no se puede rebalancear ese activo
                        current_asset_values[fund] = 0 
                        new_shares[fund] = shares[fund] # Mantener participaciones antiguas
                        st.warning(f"Precio inválido para {fund} en rebalanceo {current_date}. No se rebalanceó este activo.")

                for fund in weights_dict:
                    target_value_fund = current_portfolio_value_before_rebal * weights_dict[fund]
                    price_at_rebal = prices.loc[current_date, fund] if current_date in prices.index and pd.notna(prices.loc[current_date, fund]) else np.nan

                    if pd.notna(price_at_rebal) and price_at_rebal > 0:
                        new_shares[fund] = target_value_fund / price_at_rebal
                        # Calcular turnover (valor absoluto de la diferencia entre valor objetivo y valor actual)
                        # Solo se cuenta la mitad del turnover (compras o ventas)
                        turnover_rebalance += abs(target_value_fund - current_asset_values.get(fund, 0))
                    elif fund not in new_shares: # Si no se pudo calcular antes y no está en new_shares
                        new_shares[fund] = shares.get(fund, 0) # Mantener participaciones si no se puede rebalancear
                
                shares = new_shares # Actualizar participaciones
                
                cost_this_rebalance = (turnover_rebalance / 2) * transaction_cost_rate # Se divide por 2 porque el turnover cuenta compras y ventas
                total_transaction_costs_value += cost_this_rebalance
                portfolio_value.loc[current_date] -= cost_this_rebalance # Deducir costes del valor de la cartera
                
                last_rebalance_date = current_date
            else:
                st.warning(f"Valor de cartera inválido ({current_portfolio_value_before_rebal}) en rebalanceo {current_date}. Omitido.")

    portfolio_returns = portfolio_value.pct_change().dropna()
    return portfolio_value.dropna(), portfolio_returns, total_transaction_costs_value

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
    individual_metrics = {}
    fund_prices = fund_prices.apply(pd.to_numeric, errors='coerce').ffill().bfill()
    fund_returns = fund_prices.pct_change().dropna(how='all')

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
        
        # CORRECCIÓN: Pasar tasa diaria a calculate_sortino_ratio
        daily_rf_for_individual_sortino = (1 + risk_free_rate_annual)**(1/252) - 1 if risk_free_rate_annual != 0 else 0.0
        metrics['Sortino Ratio'] = calculate_sortino_ratio(fund_ret_series, daily_required_return_for_empyrical=daily_rf_for_individual_sortino)
        individual_metrics[fund_name] = metrics
    return pd.DataFrame(individual_metrics)

def calculate_returns(prices):
    """Calcula los retornos diarios a partir de los precios."""
    if prices is None or prices.empty: return None
    prices_numeric = prices.apply(pd.to_numeric, errors='coerce')
    if isinstance(prices_numeric, pd.Series):
        prices_numeric = prices_numeric.replace(0, np.nan).ffill().bfill() # Evitar división por cero y propagar
        if prices_numeric.isnull().all(): return None
        return prices_numeric.pct_change().dropna()
    else:
        prices_numeric = prices_numeric.replace(0, np.nan).ffill(axis=0).bfill(axis=0)
        if prices_numeric.isnull().all().all(): return None
        return prices_numeric.pct_change().dropna(how='all')

def calculate_sortino_ratio(returns, daily_required_return_for_empyrical=0.0):
    """
    Calcula el ratio de Sortino anualizado.
    Asume que 'returns' son retornos diarios y 'daily_required_return_for_empyrical' es la tasa requerida diaria.
    """
    # CORRECCIÓN: Convertir a pd.Series si es np.ndarray para un manejo uniforme y evitar AttributeError
    if isinstance(returns, np.ndarray):
        if returns.size == 0: # Si el array NumPy está vacío
            returns = pd.Series(dtype=float) 
        else:
            returns = pd.Series(returns)

    # Comprobaciones iniciales después de asegurar que 'returns' es una Serie (o None)
    if returns is None or returns.empty or returns.isnull().all():
        return np.nan
    
    # Asegurar que los retornos sean numéricos y finitos después de la conversión y antes de empyrical
    # Empyrical puede manejar Series con NaNs, pero es mejor limpiarlos antes si es posible.
    returns_cleaned = pd.to_numeric(returns, errors='coerce').dropna()
    if returns_cleaned.empty or not np.all(np.isfinite(returns_cleaned)):
        return np.nan
    if len(returns_cleaned) < 2: # Necesita al menos 2 puntos para calcular std dev en empyrical
        return np.nan
        
    try:
        # empyrical.sortino_ratio toma retornos y tasa requerida en la misma frecuencia (diaria aquí)
        # y luego anualiza el ratio final si se especifica 'annualization'.
        sortino = empyrical.sortino_ratio(returns_cleaned, 
                                          required_return=daily_required_return_for_empyrical, 
                                          annualization=252)
        return sortino if np.isfinite(sortino) else np.nan
    except Exception: 
        # Fallback manual (simplificado, ya que empyrical es preferido)
        returns_arr = returns_cleaned.values
        
        downside_returns = returns_arr[returns_arr < daily_required_return_for_empyrical]
        
        if downside_returns.size == 0:
            mean_daily_ret = np.mean(returns_arr)
            if mean_daily_ret > daily_required_return_for_empyrical: return np.inf
            return 0.0

        mean_daily_portfolio_return = np.mean(returns_arr)
        target_downside_dev_daily = np.sqrt(np.mean(np.square(downside_returns - daily_required_return_for_empyrical)))
        
        if target_downside_dev_daily == 0: 
            return np.inf if mean_daily_portfolio_return > daily_required_return_for_empyrical else 0.0

        sortino_daily = (mean_daily_portfolio_return - daily_required_return_for_empyrical) / target_downside_dev_daily
        return sortino_daily * np.sqrt(252)


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
                except Exception as e: st.warning(f"Corr rodante {col1} vs {col2} falló: {e}")
        rolling_corr_pairs = rolling_corr_pairs.dropna(how='all')
    
    rolling_corr_avg = None
    try:
        # Calcular la media de las correlaciones por pares (off-diagonal)
        rolling_corr_matrices = returns.rolling(window=window).corr()
        if not rolling_corr_matrices.empty:
            avg_corrs = []
            for date_idx in rolling_corr_matrices.index.get_level_values(0).unique():
                matrix_slice = rolling_corr_matrices.loc[date_idx]
                if matrix_slice.shape[0] > 1: # Necesita al menos 2x2 para tener off-diagonal
                    avg_corrs.append(matrix_slice.values[np.triu_indices_from(matrix_slice.values, k=1)].mean())
                else:
                    avg_corrs.append(np.nan) # O 1.0 si solo hay un activo, pero aquí esperamos múltiples
            
            if avg_corrs:
                 rolling_corr_avg = pd.Series(avg_corrs, index=rolling_corr_matrices.index.get_level_values(0).unique())
                 rolling_corr_avg.name = "Correlación Promedio Rodante"
                 rolling_corr_avg = rolling_corr_avg.dropna()
    except Exception as e: st.warning(f"Corr promedio rodante falló: {e}")
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
        st.error(f"Error en contribución al riesgo: {e}")
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
            
    except Exception as e: st.error(f"Error métricas vs benchmark: {e}")
    return metrics

def calculate_rolling_beta(portfolio_returns, benchmark_returns, window):
    if portfolio_returns is None or benchmark_returns is None or portfolio_returns.empty or benchmark_returns.empty: return None
    
    common_index = portfolio_returns.index.intersection(benchmark_returns.index)
    if window <= 1 or window > len(common_index): return None
    
    try:
        portfolio_returns_aligned = portfolio_returns.loc[common_index]
        benchmark_returns_aligned = benchmark_returns.loc[common_index]
        
        # Usar empyrical para beta rodante si es posible, o manual
        # Empyrical no tiene una función directa de beta rodante, así que se hace manual
        df_combined = pd.DataFrame({'portfolio': portfolio_returns_aligned, 'benchmark': benchmark_returns_aligned})
        
        # Covarianza rodante entre cartera y benchmark
        rolling_cov = df_combined['portfolio'].rolling(window=window).cov(df_combined['benchmark'])
        # Varianza rodante del benchmark
        rolling_var = df_combined['benchmark'].rolling(window=window).var()
        
        rolling_beta = (rolling_cov / rolling_var).replace([np.inf, -np.inf], np.nan).dropna()
        rolling_beta.name = f"Beta Rodante ({window}d)"
        return rolling_beta
    except Exception as e: st.error(f"Error beta rodante: {e}"); return None

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

# --- NUEVAS FUNCIONES CUANTITATIVAS ---
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
        st.warning(f"No hay suficientes datos para el escenario histórico seleccionado ({scenario_start.date()} - {scenario_end.date()}).")
        return None, {}

    # Usar una versión simplificada del backtest para este escenario, sin rebalanceo complejo
    # y asumiendo que los pesos se aplican al inicio del escenario y se mantienen.
    asset_columns_original = list(weights_dict.keys())
    # Filtrar solo los activos que existen en scenario_data
    asset_columns_in_scenario = [col for col in asset_columns_original if col in scenario_data.columns]
    
    if not asset_columns_in_scenario:
        st.error(f"Ninguno de los activos de la cartera ({', '.join(asset_columns_original)}) tiene datos para el escenario histórico seleccionado.")
        return None, {}

    scenario_prices = scenario_data[asset_columns_in_scenario].copy().ffill().bfill()

    # Re-normalizar pesos para los activos disponibles en el escenario
    weights_dict_scenario = {k: weights_dict[k] for k in asset_columns_in_scenario if k in weights_dict}
    current_total_weight_scenario = sum(weights_dict_scenario.values())
    if current_total_weight_scenario > 1e-6:
        weights_dict_scenario = {k: v / current_total_weight_scenario for k, v in weights_dict_scenario.items()}
    else: # Si los pesos de los activos disponibles suman cero
        st.error("Los pesos de los activos disponibles en el escenario suman cero. No se puede simular.")
        return None, {}
        
    if scenario_prices.isnull().values.any(): 
        st.warning(f"Faltan datos para algunos activos en el escenario histórico incluso después de ffill/bfill. Resultados pueden ser imprecisos.")
        # No se eliminan columnas aquí, se manejará en el bucle de cálculo de valor

    scenario_portfolio_value = pd.Series(index=scenario_prices.index, dtype=float)
    if scenario_prices.empty: return None, {} # Doble chequeo
    scenario_portfolio_value.iloc[0] = initial_investment_for_scenario 

    shares = {}
    for fund in asset_columns_in_scenario: # Usar solo los activos con datos en el escenario
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
        st.warning("Insuficientes retornos históricos para una simulación de Montecarlo fiable.")
        return None, None

    mean_return = portfolio_returns.mean()
    std_dev = portfolio_returns.std()

    if pd.isna(mean_return) or pd.isna(std_dev) or std_dev == 0:
        st.warning("Media o desviación estándar de retornos no válida para Montecarlo.")
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


# --- Función de Optimización (ya presente, con ajustes menores si son necesarios) ---
def optimize_portfolio(prices, risk_free_rate=0.0, fixed_income_assets=None, money_market_assets=None):
    """
    Calcula la frontera eficiente y carteras óptimas (Min Vol, Max Sharpe)
    con restricciones opcionales para Renta Fija y Monetario.
    """
    if prices is None or prices.empty or prices.shape[1] < 2:
        st.warning("Se necesitan al menos 2 activos con datos válidos para la optimización.")
        return None, None, None, None, None

    results = {}
    try:
        prices_cleaned = prices.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
        prices_cleaned = prices_cleaned.dropna(axis=1, how='all').dropna(axis=0, how='all').ffill().bfill()
        prices_cleaned = prices_cleaned.dropna(axis=1, how='any') # Asegurar que no hay columnas enteras NaN

        if prices_cleaned.shape[1] < 2:
            st.warning(f"Menos de 2 activos con datos suficientes para optimización. Activos: {prices_cleaned.columns.tolist()}")
            return None, None, None, None, None
        if prices_cleaned.shape[0] < 20: # PyPortfolioOpt puede necesitar más datos
            st.warning(f"No hay suficientes filas de datos ({prices_cleaned.shape[0]}) para una optimización robusta. Se requieren al menos 20.")
            return None, None, None, None, None

        mu = expected_returns.mean_historical_return(prices_cleaned, frequency=252)
        S = risk_models.CovarianceShrinkage(prices_cleaned, frequency=252).ledoit_wolf()

        ef = EfficientFrontier(mu, S, weight_bounds=(0, 1)) # Solo largos

        # Renta Fija (Máx 9%)
        if fixed_income_assets:
            fixed_income_in_model = [asset for asset in fixed_income_assets if asset in ef.tickers]
            if fixed_income_in_model:
                fi_indices = [ef.tickers.index(asset) for asset in fixed_income_in_model]
                if fi_indices:
                    ef.add_constraint(lambda w: cp.sum(w[fi_indices]) <= 0.09)
                    # st.info(f"Restricción aplicada: Suma de pesos de Renta Fija ({', '.join(fixed_income_in_model)}) <= 9%.")

        # Monetario (Máx 1%)
        if money_market_assets:
            money_market_in_model = [asset for asset in money_market_assets if asset in ef.tickers]
            if money_market_in_model:
                mm_indices = [ef.tickers.index(asset) for asset in money_market_in_model]
                if mm_indices:
                    ef.add_constraint(lambda w: cp.sum(w[mm_indices]) <= 0.01)
                    # st.info(f"Restricción aplicada: Suma de pesos de Monetario ({', '.join(money_market_in_model)}) <= 1%.")
        
        # MVP
        try:
            ef_min_vol = copy.deepcopy(ef)
            mvp_weights_raw = ef_min_vol.min_volatility()
            mvp_weights = {k: v for k, v in mvp_weights_raw.items() if abs(v) > 1e-5} # Umbral pequeño para limpieza
            mvp_performance = ef_min_vol.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            results['mvp_weights'] = mvp_weights
            results['mvp_performance'] = {'expected_return': mvp_performance[0], 'annual_volatility': mvp_performance[1], 'sharpe_ratio': mvp_performance[2]}
        except (ValueError, cp.error.SolverError) as e_mvp:
            if "infeasible" in str(e_mvp).lower() or "optimal_inaccurate" in str(e_mvp).lower():
                st.warning(f"Optimización MVP infactible o imprecisa (posiblemente por restricciones): {e_mvp}")
            else: st.warning(f"No se pudo calcular MVP: {e_mvp}")
            results['mvp_weights'], results['mvp_performance'] = None, None
        except Exception as e_mvp_other:
            st.warning(f"Error inesperado en MVP: {e_mvp_other}")
            results['mvp_weights'], results['mvp_performance'] = None, None

        # Max Sharpe
        try:
            if not (mu > risk_free_rate).any(): # Si ningún activo supera Rf, Max Sharpe puede ser problemático
                st.warning("Ningún activo supera la tasa libre de riesgo; Max Sharpe puede ser igual a MVP o fallar.")
            
            ef_max_sharpe = copy.deepcopy(ef)
            ef_max_sharpe.add_objective(objective_functions.L2_reg, gamma=0.1) # Regularización para estabilidad
            msr_weights_raw = ef_max_sharpe.max_sharpe(risk_free_rate=risk_free_rate)
            msr_weights = {k: v for k, v in msr_weights_raw.items() if abs(v) > 1e-5}
            msr_performance = ef_max_sharpe.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            results['msr_weights'] = msr_weights
            results['msr_performance'] = {'expected_return': msr_performance[0], 'annual_volatility': msr_performance[1], 'sharpe_ratio': msr_performance[2]}
        except (ValueError, cp.error.SolverError) as e_msr:
            if "infeasible" in str(e_msr).lower() or "optimal_inaccurate" in str(e_msr).lower():
                st.warning(f"Optimización Max Sharpe infactible o imprecisa (posiblemente por restricciones): {e_msr}")
            elif "exceeding the risk-free rate" not in str(e_msr).lower(): # No mostrar este error si es el de Rf
                 st.warning(f"No se pudo calcular Max Sharpe: {e_msr}")
            results['msr_weights'], results['msr_performance'] = None, None
        except Exception as e_msr_other:
            st.warning(f"Error inesperado en Max Sharpe: {e_msr_other}")
            results['msr_weights'], results['msr_performance'] = None, None

        # Frontera Eficiente
        try:
            ef_frontier_calc = copy.deepcopy(ef)
            n_samples = 100
            
            # Determinar rango de retornos para la frontera
            # Usar el retorno del MVP como mínimo si está disponible y es razonable
            min_ret_for_frontier = mu.min()
            if results.get('mvp_performance') and results['mvp_performance']['expected_return'] > mu.min():
                min_ret_for_frontier = results['mvp_performance']['expected_return']
            
            max_ret_for_frontier = mu.max()

            if min_ret_for_frontier >= max_ret_for_frontier - 1e-4: # Si el rango es demasiado pequeño o inválido
                 st.warning("Rango de retornos para la frontera es inválido o demasiado pequeño. No se puede trazar.")
                 raise ValueError("Rango inválido para frontera")

            target_returns = np.linspace(min_ret_for_frontier, max_ret_for_frontier, n_samples)
            frontier_volatility, frontier_sharpe = [], []
            
            for target_ret in target_returns:
                try:
                    ef_point = copy.deepcopy(ef_frontier_calc) # Usar una copia fresca para cada punto
                    ef_point.efficient_return(target_return=target_ret)
                    perf = ef_point.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
                    frontier_volatility.append(perf[1])
                    frontier_sharpe.append(perf[2])
                except (ValueError, cp.error.SolverError): # Si un punto es infactible
                    frontier_volatility.append(np.nan)
                    frontier_sharpe.append(np.nan)
            
            frontier_df = pd.DataFrame({'Return': target_returns, 'Volatility': frontier_volatility, 'Sharpe': frontier_sharpe}).dropna().sort_values(by='Volatility')
            results['frontier_df'] = frontier_df
        except Exception as e_frontier:
            st.warning(f"No se pudieron generar puntos de la frontera: {e_frontier}")
            results['frontier_df'] = None

        return (results.get('mvp_weights'), results.get('msr_weights'),
                results.get('mvp_performance'), results.get('msr_performance'),
                results.get('frontier_df'))
    except Exception as e:
        st.error(f"Error general en optimización: {e}")
        return None, None, None, None, None


# --- Interfaz de Streamlit ---
st.title("🚀 Backtester Quant v5.0 (Funcionalidades Extendidas)")
st.markdown("""
Sube tu archivo CSV con los precios históricos de tus fondos **y tu benchmark** para analizar, optimizar tu cartera y realizar análisis de riesgo avanzados.
**Formato esperado:** 1ª col Fecha, siguientes ISINs/Tickers y Benchmark; sep CSV (',' o ';'); decimal (',' o '.').
""")

# --- Barra Lateral ---
st.sidebar.header("⚙️ Configuración General")
uploaded_file = st.sidebar.file_uploader("1. Carga tu archivo CSV (con Benchmark)", type=["csv"])

# --- Estado de Sesión ---
default_session_state = {
    'data': None, 'weights': {}, 'run_results': None, 'last_uploaded_id': None,
    'benchmark_col': None, 'rolling_window': 60, 'risk_free_rate': RISK_FREE_RATE_DEFAULT,
    'fixed_income_selection': [], 'money_market_selection': [],
    'transaction_cost_bps': 0.0, 
    'var_confidence_level': CONFIDENCE_LEVEL_VAR_DEFAULT, 
    'var_dist_type': 'histórica', 
    'hypothetical_shock_pct': -10.0, 
    'selected_historical_scenario': "Ninguno", 
    'mc_num_simulations': 1000, 
    'mc_projection_days': 252, 
}
for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Carga y Selección de Benchmark y Categorías de Activos ---
benchmark_col = None
asset_columns_for_selection = []

if uploaded_file is not None:
    uploaded_file_id = uploaded_file.name + str(uploaded_file.size)
    if st.session_state.last_uploaded_id != uploaded_file_id:
        st.session_state['data'] = load_data(uploaded_file)
        st.session_state.last_uploaded_id = uploaded_file_id
        st.session_state['run_results'] = None 
        st.session_state['benchmark_col'] = None
        st.session_state['fixed_income_selection'] = []
        st.session_state['money_market_selection'] = []
        if st.session_state['data'] is not None:
            temp_cols = st.session_state['data'].columns.tolist()
            common_benchmarks = ['^GSPC', 'SPY', 'IBEX', '^IBEX', 'benchmark', 'indice', 'Benchmark', 'Index', 'MSCI World']
            guessed_bm = next((col for col in temp_cols if col in common_benchmarks or col.lower() in [bm.lower() for bm in common_benchmarks]), None)
            st.session_state['weights'] = {fund: 0.0 for fund in temp_cols if fund != guessed_bm}
        else: st.session_state['weights'] = {}
data = st.session_state['data']

if data is not None:
    st.sidebar.success(f"Archivo '{uploaded_file.name}' cargado.")
    st.sidebar.markdown("---")
    available_columns = data.columns.tolist()
    
    common_benchmarks_ui = ['^GSPC', 'SPY', 'IBEX', '^IBEX', 'benchmark', 'indice', 'Benchmark', 'Index', 'MSCI World']
    detected_benchmark_index = 0
    saved_benchmark = st.session_state.get('benchmark_col')
    if saved_benchmark and saved_benchmark in available_columns:
        detected_benchmark_index = available_columns.index(saved_benchmark)
    else:
        for i, col in enumerate(available_columns):
            if col in common_benchmarks_ui or col.lower() in [bm.lower() for bm in common_benchmarks_ui]:
                detected_benchmark_index = i; break
    st.session_state['benchmark_col'] = st.sidebar.selectbox("2. Selecciona la Columna Benchmark", options=available_columns, index=detected_benchmark_index, key='benchmark_selector')
    benchmark_col = st.session_state['benchmark_col']
    
    asset_columns_for_selection = [col for col in available_columns if col != benchmark_col]

    st.sidebar.markdown("---")
    st.sidebar.subheader("2a. Activos para Restricciones de Optimización")
    st.session_state['fixed_income_selection'] = st.sidebar.multiselect(
        "Activos de Renta Fija (Máx. 9% total en optimización)",
        options=asset_columns_for_selection, default=st.session_state.get('fixed_income_selection', []),
        help="Selecciona los ISINs/activos que son de Renta Fija."
    )
    st.session_state['money_market_selection'] = st.sidebar.multiselect(
        "Activos Monetarios (Máx. 1% total en optimización)",
        options=asset_columns_for_selection, default=st.session_state.get('money_market_selection', []),
        help="Selecciona los ISINs/activos que son Monetarios."
    )
    st.sidebar.markdown("---")

    st.sidebar.subheader("Parámetros del Backtest")
    min_date, max_date = data.index.min().date(), data.index.max().date()
    start_date = st.sidebar.date_input("3. Fecha de Inicio", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("4. Fecha de Fin", max_date, min_value=start_date, max_value=max_date)
    initial_investment = st.sidebar.number_input("5. Inversión Inicial (€)", min_value=1, value=10000, step=100)
    rebalance_freq = st.sidebar.selectbox("6. Frecuencia de Rebalanceo", ['No Rebalancear', 'Mensual', 'Trimestral', 'Anual'])
    st.session_state['transaction_cost_bps'] = st.sidebar.number_input("Coste Transacción (pb por operación)", min_value=0.0, value=st.session_state.get('transaction_cost_bps', 0.0), step=0.5, format="%.2f")
    st.sidebar.markdown("---")

    st.sidebar.subheader("Parámetros de Análisis")
    try: 
        filtered_len = len(data.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)])
        max_window_val = max(20, filtered_len - 5) if filtered_len > 25 else 20 
    except Exception: max_window_val = 252*3 
    default_window_val = min(st.session_state.get('rolling_window', 60), max_window_val) if max_window_val >= 20 else 20
    st.session_state['rolling_window'] = st.sidebar.number_input("7. Ventana Análisis Rodante (días)", min_value=20, max_value=max_window_val, value=default_window_val, step=10)
    st.session_state['risk_free_rate'] = st.sidebar.number_input("8. Tasa Libre de Riesgo Anual (%)", min_value=-10.0, max_value=25.0, value=st.session_state.get('risk_free_rate', RISK_FREE_RATE_DEFAULT) * 100 , step=0.1, format="%.2f") / 100.0
    st.sidebar.markdown("---")

    st.sidebar.subheader("Análisis de Riesgo Avanzado y Escenarios")
    st.session_state['var_confidence_level'] = st.sidebar.slider("Nivel Confianza VaR/ES (%)", min_value=90.0, max_value=99.9, value=st.session_state.get('var_confidence_level', CONFIDENCE_LEVEL_VAR_DEFAULT)*100, step=0.1, format="%.1f") / 100.0
    st.session_state['var_dist_type'] = st.sidebar.selectbox("Distribución VaR/ES Paramétrico", ['normal', 't', 'histórica'], index=['histórica', 'normal', 't'].index(st.session_state.get('var_dist_type', 'histórica')))
    st.session_state['hypothetical_shock_pct'] = st.sidebar.number_input("Shock Hipotético de Mercado (%)", min_value=-99.0, max_value=50.0, value=st.session_state.get('hypothetical_shock_pct', -10.0), step=1.0, format="%.1f")
    st.session_state['selected_historical_scenario'] = st.sidebar.selectbox("Escenario Histórico de Crisis", options=list(HISTORICAL_SCENARIOS.keys()), index=list(HISTORICAL_SCENARIOS.keys()).index(st.session_state.get('selected_historical_scenario', "Ninguno")))
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("Simulación Montecarlo (Proyección Futura)")
    st.session_state['mc_num_simulations'] = st.sidebar.number_input("Nº Simulaciones Montecarlo", min_value=100, max_value=10000, value=st.session_state.get('mc_num_simulations', 1000), step=100)
    st.session_state['mc_projection_days'] = st.sidebar.number_input("Días de Proyección Montecarlo", min_value=20, max_value=252*5, value=st.session_state.get('mc_projection_days', 252), step=10) 
    st.sidebar.markdown("---")

    st.sidebar.subheader("9. Asignación de Pesos (%) [Excluye Benchmark]")
    with st.sidebar.form(key='weights_form'):
        weights_input = {}
        if not asset_columns_for_selection:
            st.sidebar.warning("No hay activos disponibles para asignar pesos (después de excluir benchmark).")
        for fund in asset_columns_for_selection:
            default_weight = st.session_state['weights'].get(fund, 0.0) * 100 if fund in st.session_state['weights'] else 0.0
            weights_input[fund] = st.number_input(f"{fund}", min_value=0.0, max_value=100.0, value=default_weight, step=1.0, format="%.2f") / 100.0
        
        submitted = st.form_submit_button("🚀 Ejecutar Análisis Completo")

        if submitted:
            st.session_state['weights'] = weights_input
            total_input_weight_sum = sum(weights_input.values())
            weights_norm = {}
            if not np.isclose(total_input_weight_sum, 1.0, atol=0.01) and total_input_weight_sum > 1e-6 : 
                st.sidebar.warning(f"Pesos suman {total_input_weight_sum*100:.2f}%. Se normalizarán.")
                weights_norm = {k: v / total_input_weight_sum for k, v in weights_input.items()}
            elif np.isclose(total_input_weight_sum, 0.0, atol=1e-6): 
                st.sidebar.error("Suma de pesos es 0%. Asigna pesos."); weights_norm = weights_input
            else: 
                weights_norm = weights_input

            active_weights = {k: v for k, v in weights_norm.items() if v > 1e-6} 
            if not active_weights:
                st.error("No hay fondos con peso > 0 asignado.")
                st.session_state['run_results'] = None
            else:
                funds_in_portfolio = list(active_weights.keys())
                cols_needed_for_backtest = funds_in_portfolio 
                
                all_cols_in_range = data.columns.tolist()
                data_full_range_unfiltered = data.loc[pd.to_datetime(start_date):pd.to_datetime(end_date), all_cols_in_range].copy()
                data_full_range_unfiltered.ffill(inplace=True); data_full_range_unfiltered.bfill(inplace=True)

                asset_data_for_backtest = data.loc[pd.to_datetime(start_date):pd.to_datetime(end_date), cols_needed_for_backtest].copy()
                asset_data_for_backtest.ffill(inplace=True); asset_data_for_backtest.bfill(inplace=True)
                
                if asset_data_for_backtest.empty or asset_data_for_backtest.shape[0] < 2:
                    st.error("No hay suficientes datos comunes para los activos de la cartera en el rango seleccionado tras limpiar NaNs.")
                    st.session_state['run_results'] = None
                elif benchmark_col and benchmark_col not in data_full_range_unfiltered.columns:
                    st.error(f"Benchmark '{benchmark_col}' no encontrado en los datos del rango seleccionado.")
                    st.session_state['run_results'] = None
                else:
                    with st.spinner("Realizando cálculos avanzados y optimización..."):
                        
                        portfolio_value, portfolio_returns, total_tx_costs = run_backtest(
                            asset_data_for_backtest, active_weights, initial_investment,
                            asset_data_for_backtest.index.min(), asset_data_for_backtest.index.max(), 
                            rebalance_freq, st.session_state['transaction_cost_bps']
                        )
                        results_dict = {'run_successful': False}

                        if portfolio_value is not None and not portfolio_value.empty:
                            annual_rf_for_metrics = st.session_state['risk_free_rate'] # Tasa anual
                            daily_rf_for_sortino_calc = (1 + annual_rf_for_metrics)**(1/252) - 1 if annual_rf_for_metrics != 0 else 0.0

                            portfolio_metrics = calculate_metrics(portfolio_value, portfolio_returns, annual_rf_for_metrics)
                            portfolio_metrics['Sortino Ratio'] = calculate_sortino_ratio(portfolio_returns, daily_required_return_for_empyrical=daily_rf_for_sortino_calc)
                            portfolio_metrics['Costes Totales Transacción (€)'] = total_tx_costs
                            portfolio_metrics['Costes Totales Transacción (%)'] = (total_tx_costs / initial_investment) if initial_investment > 0 else 0.0
                            
                            asset_data_analysis = data_full_range_unfiltered[funds_in_portfolio].loc[portfolio_value.index]
                            individual_metrics = calculate_individual_metrics(asset_data_analysis, annual_rf_for_metrics)
                            asset_returns = calculate_returns(asset_data_analysis) 

                            benchmark_data_analysis = None
                            benchmark_returns = None
                            individual_asset_benchmark_metrics = pd.DataFrame(index=funds_in_portfolio, columns=['Beta', 'Alpha (anual)'])
                            rolling_beta_portfolio = None
                            if benchmark_col and benchmark_col in data_full_range_unfiltered.columns:
                                benchmark_data_analysis = data_full_range_unfiltered[benchmark_col].loc[portfolio_value.index] 
                                benchmark_returns = calculate_returns(benchmark_data_analysis)
                                if benchmark_returns is not None and not benchmark_returns.empty:
                                    common_idx_bm = portfolio_returns.index.intersection(benchmark_returns.index)
                                    if len(common_idx_bm) >= max(2, st.session_state['rolling_window']): 
                                        portfolio_returns_aligned_bm = portfolio_returns.loc[common_idx_bm]
                                        benchmark_returns_aligned_bm = benchmark_returns.loc[common_idx_bm]
                                        
                                        benchmark_metrics_calc = calculate_benchmark_metrics(portfolio_returns_aligned_bm, benchmark_returns_aligned_bm, annual_rf_for_metrics)
                                        portfolio_metrics.update(benchmark_metrics_calc)
                                        
                                        temp_asset_bm_metrics = {} 
                                        if asset_returns is not None:
                                            for fund in funds_in_portfolio:
                                                if fund in asset_returns.columns:
                                                    fund_ret_aligned_bm = asset_returns[fund].loc[common_idx_bm]
                                                    if len(fund_ret_aligned_bm.dropna()) >= 2:
                                                        metrics_bm_ind = calculate_benchmark_metrics(fund_ret_aligned_bm, benchmark_returns_aligned_bm, annual_rf_for_metrics)
                                                        temp_asset_bm_metrics[fund] = {'Beta': metrics_bm_ind.get('Beta'), 'Alpha (anual)': metrics_bm_ind.get('Alpha (anual)')}
                                                    else: temp_asset_bm_metrics[fund] = {'Beta': np.nan, 'Alpha (anual)': np.nan}
                                                else: temp_asset_bm_metrics[fund] = {'Beta': np.nan, 'Alpha (anual)': np.nan}
                                            individual_asset_benchmark_metrics.update(pd.DataFrame(temp_asset_bm_metrics).T)
                                        
                                        rolling_beta_portfolio = calculate_rolling_beta(portfolio_returns_aligned_bm, benchmark_returns_aligned_bm, st.session_state['rolling_window'])
                                    else: st.warning("Insuficientes datos comunes con benchmark para algunas métricas relativas.")
                                else: st.warning("No se pudieron calcular retornos del benchmark.")
                            
                            normalized_individual_prices = pd.DataFrame()
                            if not asset_data_analysis.empty and (asset_data_analysis.iloc[0].abs() > 1e-9).all(): 
                                normalized_individual_prices = asset_data_analysis / asset_data_analysis.iloc[0] * 100
                            
                            normalized_portfolio_value = portfolio_value / portfolio_value.iloc[0] * 100
                            normalized_portfolio_value.name = "Cartera Total"
                            
                            normalized_benchmark = None
                            if benchmark_data_analysis is not None and not benchmark_data_analysis.empty:
                                first_bm_val = benchmark_data_analysis.iloc[0]
                                if pd.notna(first_bm_val) and abs(first_bm_val) > 1e-9:
                                    normalized_benchmark = benchmark_data_analysis / first_bm_val * 100
                                    normalized_benchmark.name = benchmark_col
                            
                            plot_data_normalized = pd.concat([normalized_portfolio_value, normalized_individual_prices], axis=1)
                            if normalized_benchmark is not None:
                                plot_data_normalized = pd.concat([plot_data_normalized, normalized_benchmark], axis=1)

                            corr_matrix = calculate_correlation_matrix(asset_returns)
                            avg_rolling_corr, pair_rolling_corr = calculate_rolling_correlation(asset_returns, window=st.session_state['rolling_window'])
                            risk_contribution = calculate_risk_contribution(asset_returns, active_weights)
                            diversification_ratio = calculate_diversification_ratio(asset_returns, active_weights)
                            portfolio_metrics['Diversification Ratio'] = diversification_ratio
                            rolling_vol, rolling_sharpe, rolling_sortino = calculate_rolling_metrics(portfolio_returns, st.session_state['rolling_window'], annual_rf_for_metrics)

                            var_hist = calculate_var_historical(portfolio_returns, st.session_state['var_confidence_level'])
                            es_hist = calculate_es_historical(portfolio_returns, st.session_state['var_confidence_level'])
                            var_param_dist_type = st.session_state['var_dist_type'] if st.session_state['var_dist_type'] != 'histórica' else 'normal'
                            var_param = calculate_var_parametric(portfolio_returns, st.session_state['var_confidence_level'], dist_type=var_param_dist_type)
                            es_param = calculate_es_parametric(portfolio_returns, st.session_state['var_confidence_level'], dist_type=var_param_dist_type)
                            
                            advanced_risk_metrics = {
                                f"VaR Histórico ({st.session_state['var_confidence_level']:.1%}) Diario": var_hist,
                                f"ES Histórico ({st.session_state['var_confidence_level']:.1%}) Diario": es_hist,
                                f"VaR Paramétrico ({var_param_dist_type}, {st.session_state['var_confidence_level']:.1%}) Diario": var_param,
                                f"ES Paramétrico ({var_param_dist_type}, {st.session_state['var_confidence_level']:.1%}) Diario": es_param,
                            }

                            stress_test_results = {}
                            shocked_val, shock_loss = apply_hypothetical_shock(portfolio_value, st.session_state['hypothetical_shock_pct'])
                            stress_test_results['hypothetical_shock'] = {
                                'shock_pct': st.session_state['hypothetical_shock_pct'],
                                'valor_cartera_post_shock': shocked_val,
                                'perdida_absoluta_shock': shock_loss,
                                'ultimo_valor_cartera': portfolio_value.iloc[-1] if not portfolio_value.empty else np.nan
                            }
                            scenario_dates_selected = HISTORICAL_SCENARIOS.get(st.session_state['selected_historical_scenario'])
                            scenario_portfolio_evolution = None
                            scenario_metrics = {}
                            if scenario_dates_selected:
                                scenario_portfolio_evolution, scenario_metrics = analyze_historical_scenario(
                                    portfolio_returns, 
                                    data, 
                                    active_weights, 
                                    scenario_dates_selected,
                                    initial_investment_for_scenario=initial_investment 
                                )
                            stress_test_results['historical_scenario'] = {
                                'scenario_name': st.session_state['selected_historical_scenario'],
                                'evolution': scenario_portfolio_evolution,
                                'metrics': scenario_metrics
                            }
                            
                            mc_simulations_df, mc_summary = run_monte_carlo_simulation(
                                portfolio_returns, 
                                portfolio_value.iloc[-1] if not portfolio_value.empty else initial_investment, 
                                st.session_state['mc_num_simulations'],
                                st.session_state['mc_projection_days']
                            )

                            mvp_weights, msr_weights, mvp_performance, msr_performance, frontier_df = optimize_portfolio(
                                asset_data_analysis, 
                                risk_free_rate=annual_rf_for_metrics,
                                fixed_income_assets=st.session_state['fixed_income_selection'],
                                money_market_assets=st.session_state['money_market_selection']
                            )

                            current_portfolio_performance_opt = None 
                            if active_weights and not asset_data_analysis.empty:
                                try:
                                    prices_cleaned_opt = asset_data_analysis.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='any')
                                    if not prices_cleaned_opt.empty and prices_cleaned_opt.shape[1] >=1 and prices_cleaned_opt.shape[0] >= 20:
                                        mu_current = expected_returns.mean_historical_return(prices_cleaned_opt, frequency=252)
                                        S_current = risk_models.CovarianceShrinkage(prices_cleaned_opt, frequency=252).ledoit_wolf()
                                        
                                        weights_array = np.array([active_weights.get(asset, 0) for asset in prices_cleaned_opt.columns])
                                        weights_sum = weights_array.sum()
                                        weights_array_norm = weights_array / weights_sum if weights_sum > 1e-9 else weights_array
                                        
                                        current_ret = pypfopt.objective_functions.portfolio_return(weights_array_norm, mu_current, negative=False)
                                        current_variance = pypfopt.objective_functions.portfolio_variance(weights_array_norm, S_current) 
                                        current_vol = np.sqrt(current_variance) if current_variance >= 0 else np.nan
                                        current_sharpe = (current_ret - annual_rf_for_metrics) / current_vol if pd.notna(current_vol) and current_vol > 1e-9 else np.nan
                                        current_portfolio_performance_opt = {'expected_return': current_ret, 'annual_volatility': current_vol, 'sharpe_ratio': current_sharpe}
                                except Exception: pass 


                            results_dict = {
                                'portfolio_value': portfolio_value, 'portfolio_metrics': portfolio_metrics, 'plot_data_normalized': plot_data_normalized,
                                'individual_metrics': individual_metrics, 'individual_asset_benchmark_metrics': individual_asset_benchmark_metrics,
                                'asset_returns': asset_returns, 'benchmark_returns': benchmark_returns, 'corr_matrix': corr_matrix,
                                'avg_rolling_corr': avg_rolling_corr, 'pair_rolling_corr': pair_rolling_corr, 'risk_contribution': risk_contribution,
                                'rolling_vol': rolling_vol, 'rolling_sharpe': rolling_sharpe, 'rolling_sortino': rolling_sortino,
                                'rolling_beta_portfolio': rolling_beta_portfolio, 'weights': active_weights, 'rolling_window_used': st.session_state['rolling_window'],
                                'benchmark_col_used': benchmark_col if benchmark_data_analysis is not None and benchmark_returns is not None else None,
                                'advanced_risk_metrics': advanced_risk_metrics, 
                                'stress_test_results': stress_test_results, 
                                'mc_simulations_df': mc_simulations_df, 
                                'mc_summary': mc_summary, 
                                'mvp_weights': mvp_weights, 'msr_weights': msr_weights, 'mvp_performance': mvp_performance,
                                'msr_performance': msr_performance, 'frontier_df': frontier_df, 'current_portfolio_performance_opt': current_portfolio_performance_opt,
                                'run_successful': True
                            }
                            st.session_state['run_results'] = results_dict
                            st.sidebar.success("Análisis y optimización completados.")
                        else:
                            st.error("La simulación del backtest no produjo resultados válidos.")
                            st.session_state['run_results'] = {'run_successful': False}

# --- Mostrar Resultados (Pestañas) ---
if 'run_results' in st.session_state and st.session_state['run_results'] is not None and st.session_state['run_results'].get('run_successful', False):
    results = st.session_state['run_results']
    benchmark_col_used = results.get('benchmark_col_used')
    portfolio_value = results.get('portfolio_value')
    portfolio_metrics = results.get('portfolio_metrics', {})
    plot_data_normalized = results.get('plot_data_normalized')
    individual_metrics = results.get('individual_metrics', pd.DataFrame())
    individual_asset_benchmark_metrics = results.get('individual_asset_benchmark_metrics', pd.DataFrame())
    benchmark_returns = results.get('benchmark_returns')
    corr_matrix = results.get('corr_matrix')
    avg_rolling_corr = results.get('avg_rolling_corr')
    risk_contribution = results.get('risk_contribution', pd.Series(dtype=float))
    rolling_vol, rolling_sharpe, rolling_sortino = results.get('rolling_vol'), results.get('rolling_sharpe'), results.get('rolling_sortino')
    rolling_beta_portfolio = results.get('rolling_beta_portfolio')
    rolling_window_used = results.get('rolling_window_used', 0)
    
    advanced_risk_metrics = results.get('advanced_risk_metrics', {})
    stress_test_results = results.get('stress_test_results', {})
    mc_simulations_df = results.get('mc_simulations_df')
    mc_summary = results.get('mc_summary')

    mvp_weights, msr_weights = results.get('mvp_weights'), results.get('msr_weights')
    mvp_performance, msr_performance = results.get('mvp_performance'), results.get('msr_performance')
    frontier_df = results.get('frontier_df')
    current_portfolio_performance_opt = results.get('current_portfolio_performance_opt')

    tabs_titles = ["📊 Visión General", "🔗 Correlación", "🧩 Activos y Riesgo", "🔬 Optimización", "⚠️ Riesgo Avanzado", "📉 Estrés y Escenarios", "🎰 Montecarlo"]
    if benchmark_col_used and benchmark_returns is not None:
        tabs_titles.insert(1, f"🆚 vs {benchmark_col_used}")
        tabs = st.tabs(tabs_titles)
        tab_vg, tab_bm, tab_corr, tab_ar, tab_opt, tab_ra, tab_stress, tab_mc = tabs[0], tabs[1], tabs[2], tabs[3], tabs[4], tabs[5], tabs[6], tabs[7]
    else:
        tabs = st.tabs(tabs_titles)
        tab_vg, tab_corr, tab_ar, tab_opt, tab_ra, tab_stress, tab_mc = tabs[0], tabs[1], tabs[2], tabs[3], tabs[4], tabs[5], tabs[6]
        tab_bm = None 

    with tab_vg:
        st.header("Visión General de la Cartera")
        st.subheader("Métricas Principales (Cartera Total)")
        try: locale.setlocale(locale.LC_ALL, 'es_ES.UTF-8') 
        except locale.Error: locale.setlocale(locale.LC_ALL, '') 

        col1, col2, col3 = st.columns(3)
        col1.metric("Rentabilidad Total", f"{portfolio_metrics.get('Rentabilidad Total', np.nan):.2%}")
        col2.metric("Rentab. Anual (CAGR)", f"{portfolio_metrics.get('Rentabilidad Anualizada (CAGR)', np.nan):.2%}")
        col3.metric("Volatilidad Anualizada", f"{portfolio_metrics.get('Volatilidad Anualizada', np.nan):.2%}")
        
        col1b, col2b, col3b = st.columns(3)
        col1b.metric("Máximo Drawdown", f"{portfolio_metrics.get('Máximo Drawdown', np.nan):.2%}")
        col2b.metric("Ratio de Sharpe", f"{portfolio_metrics.get('Ratio de Sharpe', np.nan):.2f}")
        col3b.metric("Ratio Sortino", f"{portfolio_metrics.get('Sortino Ratio', np.nan):.2f}")
        
        col1c, col2c, col3c = st.columns(3)
        col1c.metric("Ratio Diversificación", f"{portfolio_metrics.get('Diversification Ratio', np.nan):.2f}", help="Vol. media pond. activos / Vol. Cartera")
        col2c.metric("Costes Transacción (€)", f"{portfolio_metrics.get('Costes Totales Transacción (€)', 0.0):,.2f}")
        col3c.metric("Costes Transacción (%)", f"{portfolio_metrics.get('Costes Totales Transacción (%)', 0.0):.2%}", help="% sobre inversión inicial")

        st.markdown("---"); st.subheader("Evolución Normalizada (Base 100)")
        if plot_data_normalized is not None and not plot_data_normalized.empty:
            plot_data_to_show = plot_data_normalized.dropna(axis=1, how='all')
            if not plot_data_to_show.empty:
                fig_perf = px.line(plot_data_to_show, title="Cartera vs. Activos" + (f" vs {benchmark_col_used}" if benchmark_col_used else ""), labels={'value': 'Valor (Base 100)', 'variable': 'Activo'})
                fig_perf.update_layout(xaxis_title="Fecha", yaxis_title="Valor Normalizado", legend_title_text='Activos')
                st.plotly_chart(fig_perf, use_container_width=True)
        else: st.warning("No hay datos para gráfico de evolución.")
        
        st.markdown("---"); st.subheader(f"Análisis Rodante de la Cartera (Ventana: {rolling_window_used} días)")
        col_roll1, col_roll2, col_roll3 = st.columns(3)
        with col_roll1:
            if rolling_vol is not None and not rolling_vol.empty:
                fig = px.line(rolling_vol, title="Volatilidad Anualizada Rodante", labels={'value': 'Vol. Anual.'})
                fig.update_layout(showlegend=False, yaxis_title="Volatilidad Anualizada"); st.plotly_chart(fig, use_container_width=True)
        with col_roll2:
            if rolling_sharpe is not None and not rolling_sharpe.empty:
                fig = px.line(rolling_sharpe, title="Ratio de Sharpe Rodante", labels={'value': 'Sharpe'})
                fig.update_layout(showlegend=False, yaxis_title="Ratio de Sharpe"); st.plotly_chart(fig, use_container_width=True)
        with col_roll3:
            if rolling_sortino is not None and not rolling_sortino.empty:
                fig = px.line(rolling_sortino, title="Ratio de Sortino Rodante", labels={'value': 'Sortino'})
                fig.update_layout(showlegend=False, yaxis_title="Ratio de Sortino"); st.plotly_chart(fig, use_container_width=True)

    if tab_bm:
        with tab_bm:
            st.header(f"Comparativa vs Benchmark ({benchmark_col_used})")
            st.subheader("Métricas Relativas al Benchmark")
            col_bm1, col_bm2, col_bm3, col_bm4 = st.columns(4)
            col_bm1.metric("Beta (β)", f"{portfolio_metrics.get('Beta', np.nan):.2f}")
            col_bm2.metric("Alpha (α) Anual", f"{portfolio_metrics.get('Alpha (anual)', np.nan):.2%}")
            col_bm3.metric("Tracking Error Anual", f"{portfolio_metrics.get('Tracking Error (anual)', np.nan):.2%}")
            col_bm4.metric("Information Ratio", f"{portfolio_metrics.get('Information Ratio', np.nan):.2f}")
            st.markdown("---")
            st.subheader(f"Beta Rodante vs {benchmark_col_used} (Ventana: {rolling_window_used} días)")
            if rolling_beta_portfolio is not None and not rolling_beta_portfolio.empty:
                fig = px.line(rolling_beta_portfolio, title="Beta Rodante de la Cartera", labels={'value': 'Beta'})
                fig.update_layout(showlegend=False, yaxis_title="Beta"); st.plotly_chart(fig, use_container_width=True)
            else: st.warning("Insuficientes datos comunes con benchmark para Beta rodante.")

    with tab_corr: 
        st.header("Análisis de Correlación entre Activos")
        st.subheader("Matriz de Correlación (Período Completo)")
        if corr_matrix is not None and not corr_matrix.empty:
            if corr_matrix.shape[0] > 1 and corr_matrix.shape[1] > 1:
                fig_heatmap, ax = plt.subplots(figsize=(max(6, corr_matrix.shape[1]*0.8), max(5, corr_matrix.shape[0]*0.7)))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax, vmin=-1, vmax=1, annot_kws={"size": 8})
                ax.set_title('Correlación entre Activos'); plt.xticks(rotation=45, ha='right', fontsize=8); plt.yticks(rotation=0, fontsize=8)
                plt.tight_layout(); st.pyplot(fig_heatmap)
        else: st.warning("No se pudo calcular matriz de correlación.")
        
        st.markdown("---"); st.subheader(f"Correlación Rodante (Ventana: {rolling_window_used} días)")
        if avg_rolling_corr is not None and not avg_rolling_corr.empty:
            fig = px.line(avg_rolling_corr, title="Correlación Promedio Rodante entre Activos", labels={'value': 'Corr. Promedio'})
            fig.update_layout(showlegend=False, yaxis_title="Correlación Promedio"); st.plotly_chart(fig, use_container_width=True)
            
            asset_list_corr = results.get('asset_returns').columns.tolist() if results.get('asset_returns') is not None else []
            if len(asset_list_corr) >= 2:
                pair_options = [(asset_list_corr[i], asset_list_corr[j]) for i in range(len(asset_list_corr)) for j in range(i + 1, len(asset_list_corr))]
                if pair_options:
                    selected_pairs = st.multiselect("Pares para correlación rodante específica:", options=pair_options, format_func=lambda p: f"{p[0]} vs {p[1]}")
                    if selected_pairs:
                        _, specific_pair_corr = calculate_rolling_correlation(results.get('asset_returns'), window=rolling_window_used, pair_list=selected_pairs)
                        if specific_pair_corr is not None and not specific_pair_corr.empty:
                            fig_pair = px.line(specific_pair_corr, title="Correlación Rodante (Pares Seleccionados)", labels={'value': 'Correlación', 'variable': 'Par'})
                            fig_pair.update_layout(yaxis_title="Correlación", legend_title_text='Pares'); st.plotly_chart(fig_pair, use_container_width=True)
        else: st.warning("Datos insuficientes para correlación rodante.")

    with tab_ar: 
        st.header("Análisis de Activos Individuales y Riesgo de Cartera")
        st.subheader("Posicionamiento Riesgo/Retorno (Activos vs Cartera)")
        if individual_metrics is not None and not individual_metrics.empty and portfolio_metrics:
            scatter_data = individual_metrics.T.copy()
            scatter_data['Rentab. Anual (CAGR)'] = pd.to_numeric(scatter_data.get('Rentabilidad Anualizada (CAGR)'), errors='coerce')
            scatter_data['Volatilidad Anualizada'] = pd.to_numeric(scatter_data.get('Volatilidad Anualizada'), errors='coerce')
            valid_cols_scatter = ['Rentab. Anual (CAGR)', 'Volatilidad Anualizada']
            if all(col in scatter_data.columns for col in valid_cols_scatter):
                scatter_data.dropna(subset=valid_cols_scatter, inplace=True)
            
            portfolio_scatter = pd.DataFrame({
                'Rentab. Anual (CAGR)': [portfolio_metrics.get('Rentabilidad Anualizada (CAGR)')], 
                'Volatilidad Anualizada': [portfolio_metrics.get('Volatilidad Anualizada')]}, 
                index=['Cartera Total'])
            portfolio_scatter['Rentab. Anual (CAGR)'] = pd.to_numeric(portfolio_scatter['Rentab. Anual (CAGR)'], errors='coerce')
            portfolio_scatter['Volatilidad Anualizada'] = pd.to_numeric(portfolio_scatter['Volatilidad Anualizada'], errors='coerce')
            portfolio_scatter.dropna(inplace=True)

            if not scatter_data.empty or not portfolio_scatter.empty:
                scatter_data_final = pd.concat([scatter_data.assign(Tipo='Activo Individual'), portfolio_scatter.assign(Tipo='Cartera Total')])
                if not scatter_data_final.empty and all(col in scatter_data_final.columns for col in valid_cols_scatter):
                    fig = px.scatter(scatter_data_final, x='Volatilidad Anualizada', y='Rentab. Anual (CAGR)', text=scatter_data_final.index, color='Tipo', hover_name=scatter_data_final.index, title="Rentabilidad vs. Volatilidad")
                    fig.update_traces(textposition='top center'); fig.update_layout(xaxis_title="Volatilidad Anualizada", yaxis_title="Rentab. Anual (CAGR)", xaxis_tickformat=".1%", yaxis_tickformat=".1%")
                    st.plotly_chart(fig, use_container_width=True)
        else: st.warning("Faltan métricas para gráfico Riesgo/Retorno.")
        
        st.markdown("---"); st.subheader("Contribución de Activos a la Volatilidad de Cartera")
        if risk_contribution is not None and not risk_contribution.empty:
            risk_contribution_pct = (risk_contribution * 100).reset_index(); risk_contribution_pct.columns = ['Activo', 'Contribución al Riesgo (%)']
            fig = px.bar(risk_contribution_pct.sort_values(by='Contribución al Riesgo (%)', ascending=False), x='Activo', y='Contribución al Riesgo (%)', title='Contribución Porcentual al Riesgo Total', text='Contribución al Riesgo (%)')
            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside'); fig.update_layout(yaxis_title="Contribución (%)", xaxis_title="Activo")
            st.plotly_chart(fig, use_container_width=True)
        else: st.warning("No se pudo calcular contribución al riesgo.")
        
        st.markdown("---"); st.subheader("Ranking Avanzado de Activos")
        if individual_metrics is not None and not individual_metrics.empty:
            ranking_df = individual_metrics.T.copy()
            if risk_contribution is not None and not risk_contribution.empty:
                ranking_df = ranking_df.join((risk_contribution.rename('Contribución Riesgo (%)') * 100), how='left')
            if benchmark_col_used and individual_asset_benchmark_metrics is not None and not individual_asset_benchmark_metrics.empty:
                ranking_df = ranking_df.join(individual_asset_benchmark_metrics[['Beta', 'Alpha (anual)']], how='left')
            
            if 'Ratio de Sharpe' in ranking_df.columns:
                ranking_df['Ratio de Sharpe'] = pd.to_numeric(ranking_df['Ratio de Sharpe'], errors='coerce')
                ranking_df.sort_values(by='Ratio de Sharpe', ascending=False, inplace=True, na_position='last')
            
            ranking_display = ranking_df.copy()
            fmt_pct = lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"; fmt_flt = lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"; fmt_pct_risk = lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
            cols_pct = ['Rentabilidad Total', 'Rentabilidad Anualizada (CAGR)', 'Volatilidad Anualizada', 'Máximo Drawdown', 'Alpha (anual)']
            cols_flt = ['Ratio de Sharpe', 'Sortino Ratio', 'Beta']; cols_pct_risk = ['Contribución Riesgo (%)']
            for col in cols_pct:
                if col in ranking_display.columns: ranking_display[col] = ranking_display[col].map(fmt_pct)
            for col in cols_flt:
                if col in ranking_display.columns: ranking_display[col] = ranking_display[col].map(fmt_flt)
            for col in cols_pct_risk:
                if col in ranking_display.columns: ranking_display[col] = ranking_display[col].map(fmt_pct_risk)
            st.dataframe(ranking_display, use_container_width=True)
        else: st.warning("Faltan datos para ranking avanzado.")

    with tab_opt: 
        st.header("🔬 Optimización de Cartera (Frontera Eficiente)")
        st.markdown("Basado en datos históricos del periodo seleccionado. Solo posiciones largas. Las restricciones de Renta Fija/Monetario se aplican aquí.")
        if frontier_df is not None and not frontier_df.empty:
            fig_frontier = go.Figure()
            fig_frontier.add_trace(go.Scatter(x=frontier_df['Volatility'], y=frontier_df['Return'], mode='lines', name='Frontera Eficiente', line=dict(color='blue', width=2)))
            if mvp_performance:
                fig_frontier.add_trace(go.Scatter(x=[mvp_performance['annual_volatility']], y=[mvp_performance['expected_return']], mode='markers+text', marker=dict(color='green', size=12, symbol='star'), text="MVP", textposition="bottom center", name=f"Min Varianza (Sharpe: {mvp_performance['sharpe_ratio']:.2f})"))
            if msr_performance:
                fig_frontier.add_trace(go.Scatter(x=[msr_performance['annual_volatility']], y=[msr_performance['expected_return']], mode='markers+text', marker=dict(color='red', size=12, symbol='star'), text="MSR", textposition="bottom center", name=f"Max Sharpe (Sharpe: {msr_performance['sharpe_ratio']:.2f})"))
            if current_portfolio_performance_opt:
                fig_frontier.add_trace(go.Scatter(x=[current_portfolio_performance_opt['annual_volatility']], y=[current_portfolio_performance_opt['expected_return']], mode='markers+text', marker=dict(color='orange', size=12, symbol='circle'), text="Actual", textposition="bottom center", name=f"Tu Cartera (Sharpe: {current_portfolio_performance_opt['sharpe_ratio']:.2f})"))
            fig_frontier.update_layout(title='Frontera Eficiente y Carteras Óptimas', xaxis_title='Volatilidad Anualizada (Riesgo)', yaxis_title='Rentabilidad Esperada Anualizada', xaxis_tickformat=".1%", yaxis_tickformat=".1%", legend_title_text='Carteras', height=500)
            st.plotly_chart(fig_frontier, use_container_width=True)
            
            st.markdown("---"); st.subheader("Pesos Sugeridos por Optimización")
            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                st.markdown("**Cartera Mínima Varianza (MVP)**")
                if mvp_weights:
                    mvp_df = pd.DataFrame.from_dict(mvp_weights, orient='index', columns=['Peso Sugerido']).rename_axis('Activo')
                    mvp_df['Peso'] = mvp_df['Peso Sugerido'].map('{:.2%}'.format)
                    st.dataframe(mvp_df.sort_values(by='Peso Sugerido', ascending=False)[['Peso']])
            with col_opt2:
                st.markdown("**Cartera Máximo Sharpe (MSR)**")
                if msr_weights:
                    msr_df = pd.DataFrame.from_dict(msr_weights, orient='index', columns=['Peso Sugerido']).rename_axis('Activo')
                    msr_df['Peso'] = msr_df['Peso Sugerido'].map('{:.2%}'.format)
                    st.dataframe(msr_df.sort_values(by='Peso Sugerido', ascending=False)[['Peso']])
        else: st.warning("No se pudo calcular frontera eficiente. Revisa datos, rango y restricciones.")

    with tab_ra:
        st.header("⚠️ Análisis de Riesgo Avanzado")
        st.subheader("Valor en Riesgo (VaR) y Expected Shortfall (ES) - Diarios")
        if advanced_risk_metrics:
            df_adv_risk = pd.DataFrame.from_dict(advanced_risk_metrics, orient='index', columns=['Valor'])
            df_adv_risk['Valor (%)'] = df_adv_risk['Valor'].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
            st.table(df_adv_risk[['Valor (%)']])
            st.caption(f"Calculado sobre los retornos diarios de la cartera durante el periodo del backtest. Nivel de confianza: {st.session_state['var_confidence_level']:.1%}.")
            if st.session_state['var_dist_type'] != 'histórica':
                 st.caption(f"Paramétrico usa distribución {st.session_state['var_dist_type']}.")
        else:
            st.warning("No se pudieron calcular métricas de riesgo avanzado.")

    with tab_stress:
        st.header("📉 Pruebas de Estrés y Análisis de Escenarios")
        
        st.subheader(f"Shock Hipotético de Mercado ({st.session_state['hypothetical_shock_pct']:.1f}%)")
        hypo_shock_res = stress_test_results.get('hypothetical_shock', {})
        if hypo_shock_res.get('ultimo_valor_cartera') is not None and not pd.isna(hypo_shock_res['ultimo_valor_cartera']):
            st.metric(label="Último Valor de Cartera (Pre-Shock)", value=f"{hypo_shock_res['ultimo_valor_cartera']:,.2f} €")
            st.metric(label="Valor de Cartera Estimado (Post-Shock)", value=f"{hypo_shock_res['valor_cartera_post_shock']:,.2f} €",
                      delta=f"{hypo_shock_res['perdida_absoluta_shock']:,.2f} € ({st.session_state['hypothetical_shock_pct']:.1f}%)",
                      delta_color="inverse") 
        else:
            st.warning("No se pudo calcular el shock hipotético (datos de cartera no disponibles).")
        st.markdown("---")

        st.subheader(f"Análisis de Escenario Histórico: {st.session_state['selected_historical_scenario']}")
        hist_scenario_res = stress_test_results.get('historical_scenario', {})
        if st.session_state['selected_historical_scenario'] == "Ninguno":
            st.info("Selecciona un escenario histórico en la barra lateral para verlo aquí.")
        elif hist_scenario_res.get('evolution') is not None and not hist_scenario_res['evolution'].empty:
            evo_df = hist_scenario_res['evolution']
            metrics_scen = hist_scenario_res.get('metrics', {})
            
            st.markdown(f"**Rendimiento Simulado de la Cartera Actual Durante el Escenario ({evo_df.index.min().date()} a {evo_df.index.max().date()})**")
            
            col_scen1, col_scen2, col_scen3 = st.columns(3)
            col_scen1.metric("Rentabilidad Total en Escenario", f"{metrics_scen.get('Rentabilidad Total', np.nan):.2%}")
            col_scen2.metric("Máximo Drawdown en Escenario", f"{metrics_scen.get('Máximo Drawdown', np.nan):.2%}")
            col_scen3.metric("Volatilidad Anual. en Escenario", f"{metrics_scen.get('Volatilidad Anualizada', np.nan):.2%}")

            fig_scenario = px.line(evo_df / evo_df.iloc[0] * 100, title=f"Evolución Normalizada en Escenario: {st.session_state['selected_historical_scenario']}")
            fig_scenario.update_layout(xaxis_title="Fecha del Escenario", yaxis_title="Valor Normalizado (Base 100)", showlegend=False)
            st.plotly_chart(fig_scenario, use_container_width=True)
        elif st.session_state['selected_historical_scenario'] != "Ninguno":
            st.warning(f"No se pudieron obtener datos para el escenario '{st.session_state['selected_historical_scenario']}'.")
            
    with tab_mc:
        st.header("🎰 Simulación de Montecarlo (Proyección Futura)")
        if mc_simulations_df is not None and not mc_simulations_df.empty and mc_summary:
            st.markdown(f"Proyección a **{st.session_state['mc_projection_days']} días hábiles** ({st.session_state['mc_num_simulations']} simulaciones) basada en retornos históricos de la cartera.")
            
            fig_mc = go.Figure()
            num_lines_to_plot = min(100, st.session_state['mc_num_simulations'])
            for col in mc_simulations_df.columns[:num_lines_to_plot]:
                fig_mc.add_trace(go.Scatter(x=mc_simulations_df.index, y=mc_simulations_df[col], mode='lines', line=dict(width=0.5, color='lightblue'), showlegend=False))
            
            fig_mc.add_trace(go.Scatter(x=mc_simulations_df.index, y=mc_simulations_df.mean(axis=1), mode='lines', line=dict(width=2, color='blue'), name='Media'))
            fig_mc.add_trace(go.Scatter(x=mc_simulations_df.index, y=mc_simulations_df.quantile(0.05, axis=1), mode='lines', line=dict(width=1, color='red', dash='dash'), name='Percentil 5%'))
            fig_mc.add_trace(go.Scatter(x=mc_simulations_df.index, y=mc_simulations_df.quantile(0.95, axis=1), mode='lines', line=dict(width=1, color='green', dash='dash'), name='Percentil 95%'))
            
            fig_mc.update_layout(title="Distribución de Valores Futuros de Cartera (Montecarlo)", xaxis_title="Fecha Proyectada", yaxis_title="Valor de Cartera (€)", height=500)
            st.plotly_chart(fig_mc, use_container_width=True)

            st.subheader("Resumen Estadístico al Final de la Proyección:")
            col_mc1, col_mc2, col_mc3 = st.columns(3)
            col_mc1.metric("Valor Medio Esperado", f"{mc_summary['mean']:,.2f} €")
            col_mc2.metric("Mediana", f"{mc_summary['median']:,.2f} €")
            col_mc3.metric("Peor 5% (Percentil 5)", f"{mc_summary['q5']:,.2f} €")
            col_mc1b, col_mc2b, col_mc3b = st.columns(3)
            col_mc1b.metric("Mejor 5% (Percentil 95)", f"{mc_summary['q95']:,.2f} €")
            col_mc2b.metric("Mínimo Simulado", f"{mc_summary['min']:,.2f} €")
            col_mc3b.metric("Máximo Simulado", f"{mc_summary['max']:,.2f} €")
        else:
            st.warning("No se pudo ejecutar la simulación de Montecarlo. Revisa los datos de retornos de la cartera.")


elif 'run_results' not in st.session_state or st.session_state['run_results'] is None:
    if data is not None: st.info("Configura parámetros y haz clic en '🚀 Ejecutar Análisis Completo'.")
    elif uploaded_file is not None: pass 
    else: st.info("Por favor, carga un archivo CSV para comenzar.")
elif not st.session_state['run_results'].get('run_successful', False):
    st.error("La ejecución del análisis falló. Revisa mensajes de error o los parámetros de entrada.")

st.sidebar.markdown("---"); st.sidebar.markdown("Backtester Quant v5.0"); st.sidebar.markdown("Dios Familia y Cojones")

