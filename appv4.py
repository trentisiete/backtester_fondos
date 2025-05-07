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

# NUEVO: Importar PyPortfolioOpt y CVXPY para restricciones
import pypfopt
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions
from pypfopt import plotting # Para helpers de ploteo si son necesarios
import cvxpy as cp # Para definir restricciones


# --- Configuración de la Página y Estilo ---
st.set_page_config(
    page_title="Backtester Quant v4.5 (Con Restricciones)", # Título actualizado
    page_icon="⚖️", # Icono actualizado
    layout="wide"
)

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
                        # st.info("Detectado CSV con separador ';' y decimal '.'")
                    except (pd.errors.ParserError, ValueError, IndexError, KeyError):
                        stringio.seek(0) # Punto decimal, coma separador
                        data = pd.read_csv(stringio, sep=',', decimal='.', parse_dates=[0], index_col=0)
                        # st.info("Detectado CSV con separador ',' y decimal '.'")

            # Procesamiento estándar
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)

            data.dropna(axis=0, how='all', inplace=True)
            data.sort_index(inplace=True)
            # Rellenar NaNs (método actualizado)
            data.ffill(inplace=True)
            data.bfill(inplace=True)

            # Asegurar datos numéricos
            for col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    try:
                        # Intentar reemplazar comas si no se hizo bien en la lectura y convertir
                        if data[col].dtype == 'object':
                            # Intentar reemplazar comas y puntos (miles) y luego convertir a float
                            cleaned_col = data[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
                            data[col] = pd.to_numeric(cleaned_col)
                        else:
                            data[col] = pd.to_numeric(data[col])
                    except (ValueError, AttributeError, TypeError) as e_conv:
                        st.error(f"Error convirtiendo columna '{col}' a numérico: {e_conv}. Verifica el formato (separador decimal/miles).")
                        return None
            return data
        except Exception as e:
            st.error(f"Error crítico al procesar el archivo CSV: {e}")
            st.error("Verifica el formato: 1ª col Fecha, siguientes ISINs/Benchmark; sep CSV (',' o ';'); decimal (',' o '.').")
            return None
    return None

def run_backtest(data, weights_dict, initial_investment, start_date, end_date, rebalance_freq):
    """Ejecuta la simulación del backtesting para la cartera."""
    asset_columns = list(weights_dict.keys())
    start_date_ts = pd.to_datetime(start_date)
    end_date_ts = pd.to_datetime(end_date)

    data_in_range = data.loc[data.index.intersection(pd.date_range(start_date_ts, end_date_ts))]
    if data_in_range.empty:
        st.warning("No hay datos en el índice para el rango de fechas seleccionado.")
        return None, None

    prices = data_in_range[asset_columns].copy()

    if prices.empty or prices.isnull().all().all():
        st.warning("No hay datos válidos para los activos en el rango de fechas seleccionado.")
        return None, None

    prices.ffill(inplace=True)
    prices.bfill(inplace=True)

    if prices.iloc[0].isnull().any():
        missing_funds = prices.columns[prices.iloc[0].isnull()].tolist()
        st.warning(f"Faltan datos iniciales para: {', '.join(missing_funds)} en {prices.index[0].date()}.")
        first_valid_date = prices.dropna(axis=0, how='any').index.min()
        if pd.isna(first_valid_date) or first_valid_date > end_date_ts:
            st.error("No hay datos comunes suficientes para iniciar el backtest en el rango seleccionado.")
            return None, None
        prices = prices.loc[first_valid_date:].copy()
        st.warning(f"Backtest comenzará en {prices.index[0].date()}.")
        if prices.empty: return None, None

    total_weight = sum(weights_dict.values())
    if not np.isclose(total_weight, 1.0) and total_weight != 0:
        weights_dict = {k: v / total_weight for k, v in weights_dict.items()}

    portfolio_value = pd.Series(index=prices.index, dtype=float)
    if not prices.empty:
        portfolio_value.loc[prices.index[0]] = initial_investment
    else:
        st.error("No se pueden inicializar los valores de la cartera.")
        return None, None

    current_weights = weights_dict.copy()
    last_rebalance_date = prices.index[0]
    initial_alloc = {fund: initial_investment * weight for fund, weight in current_weights.items()}
    shares = {}
    for fund in current_weights:
        initial_price = prices[fund].iloc[0]
        if pd.notna(initial_price) and initial_price != 0:
            shares[fund] = initial_alloc[fund] / initial_price
        else:
            shares[fund] = 0
            st.warning(f"Precio inicial inválido para {fund} en {prices.index[0].date()}. Peso inicial no aplicado.")

    rebalance_offset = {'Mensual': pd.DateOffset(months=1), 'Trimestral': pd.DateOffset(months=3),
                        'Anual': pd.DateOffset(years=1), 'No Rebalancear': None}
    offset = rebalance_offset[rebalance_freq]

    for i in range(1, len(prices)):
        current_date = prices.index[i]
        prev_date = prices.index[i-1]
        current_portfolio_value = 0.0
        for fund in shares:
            if current_date in prices.index:
                current_price = prices.loc[current_date, fund]
                if pd.notna(current_price):
                    current_portfolio_value += shares[fund] * current_price
                elif prev_date in prices.index:
                    prev_price = prices.loc[prev_date, fund]
                    if pd.notna(prev_price):
                        current_portfolio_value += shares[fund] * prev_price
            elif prev_date in portfolio_value.index and pd.notna(portfolio_value.loc[prev_date]):
                current_portfolio_value = portfolio_value.loc[prev_date]
                break
            else:
                st.error(f"Fecha {current_date} no encontrada y sin valor previo de cartera.")
                return None, None

        if pd.isna(current_portfolio_value) or (current_portfolio_value == 0 and prev_date in portfolio_value.index and portfolio_value.loc[prev_date] != 0):
            prev_portfolio_value = portfolio_value.loc[prev_date]
            if pd.notna(prev_portfolio_value):
                current_portfolio_value = prev_portfolio_value
            else:
                st.error(f"Error irrecuperable calculando valor de cartera en {current_date}.")
                return None, None
        portfolio_value.loc[current_date] = current_portfolio_value

        if offset and current_date >= last_rebalance_date + offset:
            if pd.notna(current_portfolio_value) and current_portfolio_value > 0:
                target_alloc = {fund: current_portfolio_value * weight for fund, weight in weights_dict.items()}
                for fund in weights_dict:
                    if current_date in prices.index:
                        current_price_rebal = prices.loc[current_date, fund]
                        if pd.notna(current_price_rebal) and current_price_rebal != 0:
                            shares[fund] = target_alloc[fund] / current_price_rebal
                        else:
                            st.warning(f"Precio inválido para {fund} en rebalanceo {current_date}. No se rebalanceó.")
                    else:
                        st.warning(f"Fecha {current_date} no encontrada para rebalanceo de {fund}.")
                last_rebalance_date = current_date
            else:
                st.warning(f"Valor de cartera inválido ({current_portfolio_value}) en rebalanceo {current_date}. Omitido.")

    portfolio_returns = portfolio_value.pct_change().dropna()
    return portfolio_value.dropna(), portfolio_returns

def calculate_metrics(portfolio_value, portfolio_returns):
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
        cagr = (base ** (1 / years)) - 1 if base > 0 else -1.0
    metrics['Rentabilidad Anualizada (CAGR)'] = cagr
    volatility = portfolio_returns.std() * np.sqrt(252) if not portfolio_returns.empty else 0
    metrics['Volatilidad Anualizada'] = volatility
    risk_free_rate_annual = st.session_state.get('risk_free_rate', 0.0)
    sharpe_ratio = (cagr - risk_free_rate_annual) / volatility if volatility != 0 else np.nan
    metrics['Ratio de Sharpe'] = sharpe_ratio
    rolling_max = portfolio_value.cummax()
    daily_drawdown = portfolio_value / rolling_max - 1
    max_drawdown = daily_drawdown.min() if not daily_drawdown.empty else 0
    metrics['Máximo Drawdown'] = max_drawdown
    return metrics

def calculate_individual_metrics(fund_prices):
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
        metrics = calculate_metrics(fund_series, fund_ret_series)
        metrics['Sortino Ratio'] = calculate_sortino_ratio(fund_ret_series, required_return=st.session_state.get('risk_free_rate', 0.0))
        individual_metrics[fund_name] = metrics
    return pd.DataFrame(individual_metrics)

def calculate_returns(prices):
    """Calcula los retornos diarios a partir de los precios."""
    if prices is None or prices.empty: return None
    prices_numeric = prices.apply(pd.to_numeric, errors='coerce')
    if isinstance(prices_numeric, pd.Series):
        prices_numeric = prices_numeric.replace(0, np.nan).ffill().bfill()
        if prices_numeric.isnull().all(): return None
        return prices_numeric.pct_change().dropna()
    else:
        prices_numeric = prices_numeric.replace(0, np.nan).ffill(axis=0).bfill(axis=0)
        if prices_numeric.isnull().all().all(): return None
        return prices_numeric.pct_change().dropna(how='all')

def calculate_sortino_ratio(returns, required_return=0.0):
    """Calcula el ratio de Sortino."""
    if returns is None or (isinstance(returns, np.ndarray) and returns.size == 0) or (isinstance(returns, pd.Series) and returns.empty):
        return np.nan
    try:
        daily_required_return = (1 + required_return)**(1/252) - 1 if required_return != 0 else 0.0
        returns_series = pd.Series(returns) if isinstance(returns, np.ndarray) else returns
        if not np.all(np.isfinite(returns_series.dropna())): return np.nan
        returns_series = returns_series.dropna()
        if len(returns_series) < 2: return np.nan
        sortino = empyrical.sortino_ratio(returns_series, required_return=daily_required_return)
        return sortino if np.isfinite(sortino) else np.nan
    except Exception: # Fallback manual
        if isinstance(returns, pd.Series): returns = returns.values
        elif not isinstance(returns, np.ndarray): returns = np.array(returns)
        returns = returns[np.isfinite(returns)]
        if returns.size < 2: return np.nan
        daily_required_return = (1 + required_return)**(1/252) - 1 if required_return != 0 else 0.0
        downside_returns = returns[returns < daily_required_return]
        if downside_returns.size == 0:
            return np.inf if np.mean(returns) > daily_required_return else 0.0
        mean_return = np.mean(returns)
        downside_std = np.sqrt(np.mean(np.square(downside_returns - daily_required_return)))
        if downside_std == 0: return np.nan
        return (mean_return - daily_required_return) / downside_std * np.sqrt(252)

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
        rolling_corr_avg = returns.rolling(window=window).corr().groupby(level=0).apply(lambda x: x.values[np.triu_indices_from(x.values, k=1)].mean())
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
    if total_weight_adj <= 1e-9: return pd.Series(dtype=float)
    weights_norm_adj = weights_adj / total_weight_adj
    returns_subset = returns[funds_adj]
    try:
        cov_matrix = returns_subset.cov() * 252
        if cov_matrix.isnull().values.any(): return pd.Series(dtype=float)
        portfolio_var = weights_norm_adj.T @ cov_matrix @ weights_norm_adj
        if portfolio_var <= 1e-10:
            return pd.Series({fund: weights_norm_adj[i] for i, fund in enumerate(funds_adj)}, name="Contribución al Riesgo (%)")
        portfolio_vol = np.sqrt(portfolio_var)
        mcsr = (cov_matrix.values @ weights_norm_adj) / portfolio_vol
        cctr = weights_norm_adj * mcsr
        risk_contribution_percent = cctr / portfolio_vol
        risk_contribution_series = pd.Series(risk_contribution_percent, index=funds_adj, name="Contribución al Riesgo (%)")
        return risk_contribution_series.fillna(0.0)
    except Exception as e:
        st.error(f"Error en contribución al riesgo: {e}")
        return pd.Series(dtype=float)

def calculate_rolling_metrics(portfolio_returns, window, required_return=0.0):
    if portfolio_returns is None or portfolio_returns.empty or window <= 1 or window > portfolio_returns.shape[0]:
        return None, None, None
    rolling_vol = (portfolio_returns.rolling(window=window).std(ddof=1) * np.sqrt(252)).dropna()
    rolling_vol.name = f"Volatilidad Rodante ({window}d)"
    rolling_annual_ret = portfolio_returns.rolling(window=window).mean() * 252
    rolling_sharpe = ((rolling_annual_ret - required_return) / rolling_vol).replace([np.inf, -np.inf], np.nan).dropna()
    rolling_sharpe.name = f"Sharpe Rodante ({window}d)"
    daily_required_return = (1 + required_return)**(1/252) - 1 if required_return != 0 else 0.0
    rolling_sortino = portfolio_returns.rolling(window=window).apply(lambda x: calculate_sortino_ratio(x, required_return=daily_required_return), raw=True).dropna()
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
        common_index_final = portfolio_returns_aligned.index.intersection(benchmark_returns_aligned.index)
        if len(common_index_final) < 2: return {}
        portfolio_returns_aligned = portfolio_returns_aligned.loc[common_index_final]
        benchmark_returns_aligned = benchmark_returns_aligned.loc[common_index_final]
        daily_rf = (1 + risk_free_rate)**(1/252) - 1 if risk_free_rate != 0 else 0.0
        metrics['Beta'] = empyrical.beta(portfolio_returns_aligned, benchmark_returns_aligned, risk_free=daily_rf)
        metrics['Alpha (anual)'] = empyrical.alpha(portfolio_returns_aligned, benchmark_returns_aligned, risk_free=daily_rf, annualization=252)
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
        df_combined = pd.DataFrame({'portfolio': portfolio_returns_aligned, 'benchmark': benchmark_returns_aligned})
        rolling_cov = df_combined['portfolio'].rolling(window=window).cov(df_combined['benchmark'])
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
    asset_returns_subset = asset_returns[funds_in_returns]
    asset_vols = asset_returns_subset.std() * np.sqrt(252)
    if asset_vols.isnull().any(): return np.nan
    weighted_avg_vol = np.sum(weights_norm * asset_vols)
    cov_matrix = asset_returns_subset.cov() * 252
    if cov_matrix.isnull().values.any(): return np.nan
    portfolio_var = weights_norm.T @ cov_matrix @ weights_norm
    if portfolio_var <= 1e-10: return np.nan
    portfolio_vol = np.sqrt(portfolio_var)
    if portfolio_vol <= 1e-9: return np.nan
    return weighted_avg_vol / portfolio_vol

# --- NUEVA Función para Optimización con RESTRICCIONES ---
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
        prices_cleaned = prices_cleaned.dropna(axis=1, how='all')

        if prices_cleaned.shape[1] < 2:
            st.warning(f"Menos de 2 activos con datos suficientes para optimización. Activos: {prices_cleaned.columns.tolist()}")
            return None, None, None, None, None
        if prices_cleaned.shape[0] < 2:
            st.warning("No hay suficientes filas de datos para la optimización.")
            return None, None, None, None, None

        mu = expected_returns.mean_historical_return(prices_cleaned, frequency=252)
        S = risk_models.CovarianceShrinkage(prices_cleaned, frequency=252).ledoit_wolf()

        ef = EfficientFrontier(mu, S, weight_bounds=(0, 1)) # Solo largos

        # --- AÑADIR RESTRICCIONES ---
        # Filtrar los activos de las restricciones para que solo incluyan los presentes en el modelo (ef.tickers)
        # ef.tickers es una lista con los nombres de los activos que el optimizador está considerando

        # Renta Fija (Máx 9%)
        if fixed_income_assets: # Si el usuario seleccionó alguno
            # Obtener los ISINs de renta fija que están realmente en los tickers del optimizador
            fixed_income_in_model = [asset for asset in fixed_income_assets if asset in ef.tickers]
            if fixed_income_in_model:
                # Obtener los índices de estos activos en el vector de pesos 'w'
                fi_indices = [ef.tickers.index(asset) for asset in fixed_income_in_model]
                if fi_indices: # Solo añadir restricción si hay activos de RF en el modelo
                    ef.add_constraint(lambda w: cp.sum(w[fi_indices]) <= 0.09)
                    st.info(f"Restricción aplicada: Suma de pesos de Renta Fija ({', '.join(fixed_income_in_model)}) <= 9%.")

        # Monetario (Máx 1%)
        if money_market_assets: # Si el usuario seleccionó alguno
            money_market_in_model = [asset for asset in money_market_assets if asset in ef.tickers]
            if money_market_in_model:
                mm_indices = [ef.tickers.index(asset) for asset in money_market_in_model]
                if mm_indices: # Solo añadir restricción si hay activos monetarios en el modelo
                    ef.add_constraint(lambda w: cp.sum(w[mm_indices]) <= 0.01)
                    st.info(f"Restricción aplicada: Suma de pesos de Monetario ({', '.join(money_market_in_model)}) <= 1%.")
        # --- FIN DE AÑADIR RESTRICCIONES ---

        # MVP
        try:
            ef_min_vol = copy.deepcopy(ef) # Copia el 'ef' con las restricciones ya añadidas
            mvp_weights_raw = ef_min_vol.min_volatility()
            mvp_weights = {k: v for k, v in mvp_weights_raw.items() if abs(v) > 1e-4}
            mvp_performance = ef_min_vol.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            results['mvp_weights'] = mvp_weights
            results['mvp_performance'] = {'expected_return': mvp_performance[0], 'annual_volatility': mvp_performance[1], 'sharpe_ratio': mvp_performance[2]}
        except ValueError as e_mvp:
            if "optimization may fail" in str(e_mvp) or "infeasible" in str(e_mvp).lower():
                st.warning(f"Optimización MVP falló o es infactible (posiblemente por restricciones): {e_mvp}")
            else: st.warning(f"No se pudo calcular MVP: {e_mvp}")
            results['mvp_weights'], results['mvp_performance'] = None, None
        except Exception as e_mvp_other:
            st.warning(f"Error inesperado en MVP: {e_mvp_other}")
            results['mvp_weights'], results['mvp_performance'] = None, None

        # Max Sharpe
        try:
            if not (mu > risk_free_rate).any():
                st.warning("Ningún activo supera la tasa libre de riesgo; no se puede calcular Máx Sharpe.")
                raise ValueError("No retornos > Rf")
            ef_max_sharpe = copy.deepcopy(ef) # Copia el 'ef' con las restricciones
            ef_max_sharpe.add_objective(objective_functions.L2_reg, gamma=0.1)
            msr_weights_raw = ef_max_sharpe.max_sharpe(risk_free_rate=risk_free_rate)
            msr_weights = {k: v for k, v in msr_weights_raw.items() if abs(v) > 1e-4}
            msr_performance = ef_max_sharpe.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            results['msr_weights'] = msr_weights
            results['msr_performance'] = {'expected_return': msr_performance[0], 'annual_volatility': msr_performance[1], 'sharpe_ratio': msr_performance[2]}
        except ValueError as e_msr:
            if "optimization may fail" in str(e_msr) or "infeasible" in str(e_msr).lower():
                st.warning(f"Optimización Max Sharpe falló o es infactible (posiblemente por restricciones): {e_msr}")
            elif "exceeding the risk-free rate" not in str(e_msr):
                st.warning(f"No se pudo calcular Max Sharpe: {e_msr}")
            results['msr_weights'], results['msr_performance'] = None, None
        except Exception as e_msr_other:
            st.warning(f"Error inesperado en Max Sharpe: {e_msr_other}")
            results['msr_weights'], results['msr_performance'] = None, None

        # Frontera Eficiente
        try:
            ef_frontier = copy.deepcopy(ef) # Copia el 'ef' con las restricciones
            n_samples = 100
            min_ret_frontier = results.get('mvp_performance')['expected_return'] if results.get('mvp_performance') else mu.min()
            max_ret_frontier = mu.max()
            if min_ret_frontier >= max_ret_frontier: raise ValueError("Rango inválido para frontera")
            target_returns = np.linspace(min_ret_frontier, max_ret_frontier, n_samples)
            frontier_volatility, frontier_sharpe = [], []
            for target_ret in target_returns:
                try:
                    ef_point = copy.deepcopy(ef_frontier)
                    ef_point.efficient_return(target_return=target_ret)
                    perf = ef_point.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
                    frontier_volatility.append(perf[1]); frontier_sharpe.append(perf[2])
                except ValueError: frontier_volatility.append(np.nan); frontier_sharpe.append(np.nan)
            frontier_df = pd.DataFrame({'Return': target_returns, 'Volatility': frontier_volatility, 'Sharpe': frontier_sharpe}).dropna().sort_values(by='Volatility')
            results['frontier_df'] = frontier_df
        except Exception as e_frontier:
            st.warning(f"No se pudieron generar puntos de la frontera: {e_frontier}")
            results['frontier_df'] = None

        return (results.get('mvp_weights'), results.get('msr_weights'),
                results.get('mvp_performance'), results.get('msr_performance'),
                results.get('frontier_df'))
    except Exception as e:
        st.error(f"Error en optimización: {e}")
        return None, None, None, None, None


# --- Interfaz de Streamlit ---
st.title("⚖️ Backtester Quant v4.5 (Restricciones)") # Título actualizado
st.markdown("""
Sube tu archivo CSV con los precios históricos de tus fondos **y tu benchmark** para analizar y optimizar tu cartera.
**Formato esperado:** 1ª col Fecha, siguientes ISINs/Tickers y Benchmark; sep CSV (',' o ';'); decimal (',' o '.').
""")

# --- Barra Lateral ---
st.sidebar.header("Configuración del Backtest y Optimización") # Título de sección actualizado
uploaded_file = st.sidebar.file_uploader("1. Carga tu archivo CSV (con Benchmark)", type=["csv"])

# --- Estado de Sesión ---
default_session_state = {
    'data': None, 'weights': {}, 'run_results': None, 'last_uploaded_id': None,
    'benchmark_col': None, 'rolling_window': 60, 'risk_free_rate': 0.0,
    'fixed_income_selection': [], 'money_market_selection': [] # NUEVO para restricciones
}
for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Carga y Selección de Benchmark y Categorías de Activos ---
benchmark_col = None
asset_columns_for_selection = [] # Columnas disponibles para selección de RF y Monetario

if uploaded_file is not None:
    uploaded_file_id = uploaded_file.name + str(uploaded_file.size)
    if st.session_state.last_uploaded_id != uploaded_file_id:
        st.session_state['data'] = load_data(uploaded_file)
        st.session_state.last_uploaded_id = uploaded_file_id
        st.session_state['run_results'] = None
        st.session_state['benchmark_col'] = None
        st.session_state['fixed_income_selection'] = [] # Resetear selección
        st.session_state['money_market_selection'] = [] # Resetear selección
        if st.session_state['data'] is not None:
            temp_cols = st.session_state['data'].columns.tolist()
            common_benchmarks = ['^GSPC', 'SPY', 'IBEX', '^IBEX', 'benchmark', 'indice', 'Benchmark', 'Index']
            guessed_bm = next((col for col in temp_cols if col in common_benchmarks or col.lower() in [bm.lower() for bm in common_benchmarks]), None)
            st.session_state['weights'] = {fund: 0.0 for fund in temp_cols if fund != guessed_bm}
        else: st.session_state['weights'] = {}
data = st.session_state['data']

if data is not None:
    st.sidebar.success(f"Archivo '{uploaded_file.name}' cargado.")
    st.sidebar.markdown("---")
    available_columns = data.columns.tolist()
    common_benchmarks = ['^GSPC', 'SPY', 'IBEX', '^IBEX', 'benchmark', 'indice', 'Benchmark', 'Index']
    detected_benchmark_index = 0
    saved_benchmark = st.session_state.get('benchmark_col')
    if saved_benchmark and saved_benchmark in available_columns:
        detected_benchmark_index = available_columns.index(saved_benchmark)
    else:
        for i, col in enumerate(available_columns):
            if col in common_benchmarks or col.lower() in [bm.lower() for bm in common_benchmarks]:
                detected_benchmark_index = i; break

    st.session_state['benchmark_col'] = st.sidebar.selectbox("2. Selecciona la Columna Benchmark", options=available_columns, index=detected_benchmark_index, key='benchmark_selector')
    benchmark_col = st.session_state['benchmark_col']
    st.sidebar.info(f"Columna '{benchmark_col}' será usada como benchmark.")

    # Columnas de activos (excluyendo benchmark) para selección de RF/Monetario
    asset_columns_for_selection = [col for col in available_columns if col != benchmark_col]

    # NUEVO: Selección de Activos de Renta Fija
    st.sidebar.markdown("---")
    st.sidebar.subheader("2a. Selección de Activos para Restricciones")
    st.session_state['fixed_income_selection'] = st.sidebar.multiselect(
        "Activos de Renta Fija (Máx. 9% total)",
        options=asset_columns_for_selection,
        default=st.session_state.get('fixed_income_selection', []),
        help="Selecciona los ISINs/activos que son de Renta Fija."
    )
    # NUEVO: Selección de Activos Monetarios
    st.session_state['money_market_selection'] = st.sidebar.multiselect(
        "Activos Monetarios (Máx. 1% total)",
        options=asset_columns_for_selection,
        default=st.session_state.get('money_market_selection', []),
        help="Selecciona los ISINs/activos que son Monetarios."
    )
    st.sidebar.markdown("---")

    min_date, max_date = data.index.min().date(), data.index.max().date()
    start_date = st.sidebar.date_input("3. Fecha de Inicio", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("4. Fecha de Fin", max_date, min_value=start_date, max_value=max_date)
    st.sidebar.markdown("---")
    initial_investment = st.sidebar.number_input("5. Inversión Inicial (€)", min_value=1, value=10000, step=100)
    st.sidebar.markdown("---")
    rebalance_freq = st.sidebar.selectbox("6. Frecuencia de Rebalanceo", ['Mensual', 'Trimestral', 'Anual', 'No Rebalancear'])
    st.sidebar.markdown("---")
    try:
        filtered_len = len(data.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)])
        max_window = max(20, filtered_len - 1) if filtered_len > 20 else 20
    except Exception: max_window = 252*3
    default_window = min(st.session_state.get('rolling_window', 60), max_window) if max_window >= 20 else 20
    st.session_state['rolling_window'] = st.sidebar.number_input("7. Ventana Análisis Rodante (días)", min_value=20, max_value=max_window, value=default_window, step=10)
    st.session_state['risk_free_rate'] = st.sidebar.number_input("8. Tasa Libre de Riesgo Anual (%)", min_value=-5.0, max_value=20.0, value=st.session_state.get('risk_free_rate', 0.0) * 100 , step=0.1, format="%.2f") / 100.0
    st.sidebar.markdown("---")
    st.sidebar.subheader("9. Asignación de Pesos (%) [Excluye Benchmark]")

    with st.sidebar.form(key='weights_form'):
        weights_input = {}
        # Usar asset_columns_for_selection (que ya excluye benchmark) para los pesos
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
            if not np.isclose(total_input_weight_sum, 1.0, atol=0.01):
                st.sidebar.warning(f"Pesos suman {total_input_weight_sum*100:.2f}%. Se normalizarán.")
                if total_input_weight_sum > 1e-6:
                    weights_norm = {k: v / total_input_weight_sum for k, v in weights_input.items()}
                else: st.sidebar.error("Suma de pesos es 0%. Asigna pesos."); weights_norm = weights_input # Evitar error, pero no es ideal
            else: weights_norm = weights_input

            active_weights = {k: v for k, v in weights_norm.items() if v > 1e-6}
            if not active_weights:
                st.error("No hay fondos con peso > 0 asignado.")
                st.session_state['run_results'] = None
            else:
                funds_in_portfolio = list(active_weights.keys())
                cols_needed = funds_in_portfolio + ([benchmark_col] if benchmark_col else [])
                cols_to_filter = [col for col in cols_needed if col in data.columns]
                data_filtered = data.loc[pd.to_datetime(start_date):pd.to_datetime(end_date), cols_to_filter].copy()
                data_filtered.ffill(inplace=True); data_filtered.bfill(inplace=True)
                essential_cols = funds_in_portfolio + ([benchmark_col] if benchmark_col in data_filtered.columns else [])
                data_filtered.dropna(subset=essential_cols, how='any', inplace=True)

                if data_filtered.empty or data_filtered.shape[0] < 2:
                    st.error("No hay suficientes datos comunes para activos (y benchmark) en el rango tras limpiar NaNs.")
                    st.session_state['run_results'] = None
                elif benchmark_col and benchmark_col not in data_filtered.columns:
                    st.error(f"Benchmark '{benchmark_col}' no encontrado tras filtrar.")
                    st.session_state['run_results'] = None
                else:
                    with st.spinner("Realizando cálculos avanzados y optimización..."):
                        asset_data_final = data_filtered[funds_in_portfolio]
                        benchmark_data_final = data_filtered[benchmark_col] if benchmark_col in data_filtered.columns else None

                        portfolio_value, portfolio_returns = run_backtest(asset_data_final, active_weights, initial_investment, asset_data_final.index.min(), asset_data_final.index.max(), rebalance_freq)
                        results_dict = {'run_successful': False}
                        if portfolio_value is not None and not portfolio_value.empty:
                            portfolio_metrics = calculate_metrics(portfolio_value, portfolio_returns)
                            portfolio_metrics['Sortino Ratio'] = calculate_sortino_ratio(portfolio_returns, required_return=st.session_state['risk_free_rate'])
                            individual_metrics = calculate_individual_metrics(asset_data_final)
                            asset_returns = calculate_returns(asset_data_final)
                            benchmark_returns = None
                            individual_asset_benchmark_metrics = pd.DataFrame(index=funds_in_portfolio, columns=['Beta', 'Alpha (anual)'])
                            rolling_beta_portfolio = None
                            if benchmark_data_final is not None and asset_returns is not None:
                                benchmark_returns = calculate_returns(benchmark_data_final)
                                if benchmark_returns is not None and not benchmark_returns.empty:
                                    common_idx_returns = portfolio_returns.index.intersection(benchmark_returns.index)
                                    if len(common_idx_returns) >= st.session_state['rolling_window']:
                                        portfolio_returns_aligned = portfolio_returns.loc[common_idx_returns]
                                        benchmark_returns_aligned = benchmark_returns.loc[common_idx_returns]
                                        benchmark_metrics_calc = calculate_benchmark_metrics(portfolio_returns_aligned, benchmark_returns_aligned, st.session_state['risk_free_rate'])
                                        portfolio_metrics.update(benchmark_metrics_calc)
                                        temp_asset_bm_metrics = {}
                                        for fund in funds_in_portfolio:
                                            if fund in asset_returns.columns:
                                                fund_ret_aligned = asset_returns[fund].loc[common_idx_returns]
                                                if len(fund_ret_aligned) >= 2:
                                                    metrics_bm = calculate_benchmark_metrics(fund_ret_aligned, benchmark_returns_aligned, st.session_state['risk_free_rate'])
                                                    temp_asset_bm_metrics[fund] = {'Beta': metrics_bm.get('Beta'), 'Alpha (anual)': metrics_bm.get('Alpha (anual)')}
                                                else: temp_asset_bm_metrics[fund] = {'Beta': np.nan, 'Alpha (anual)': np.nan}
                                            else: temp_asset_bm_metrics[fund] = {'Beta': np.nan, 'Alpha (anual)': np.nan}
                                        individual_asset_benchmark_metrics.update(pd.DataFrame(temp_asset_bm_metrics).T)
                                        rolling_beta_portfolio = calculate_rolling_beta(portfolio_returns_aligned, benchmark_returns_aligned, st.session_state['rolling_window'])

                            normalized_individual_prices = pd.DataFrame()
                            asset_data_norm_source = asset_data_final.loc[portfolio_value.index]
                            if not asset_data_norm_source.empty and (asset_data_norm_source.iloc[0] > 1e-9).all():
                                normalized_individual_prices = asset_data_norm_source / asset_data_norm_source.iloc[0] * 100
                            normalized_portfolio_value = portfolio_value / portfolio_value.iloc[0] * 100
                            normalized_portfolio_value.name = "Cartera Total"
                            normalized_benchmark = None
                            if benchmark_data_final is not None:
                                benchmark_data_norm = benchmark_data_final.loc[portfolio_value.index]
                                if not benchmark_data_norm.empty:
                                    first_bm_val = benchmark_data_norm.iloc[0]
                                    if pd.notna(first_bm_val) and first_bm_val > 1e-9:
                                        normalized_benchmark = benchmark_data_norm / first_bm_val * 100
                                        normalized_benchmark.name = benchmark_col
                            plot_data_normalized = pd.concat([normalized_portfolio_value, normalized_individual_prices], axis=1)
                            if normalized_benchmark is not None:
                                plot_data_normalized = pd.concat([plot_data_normalized, normalized_benchmark], axis=1)

                            corr_matrix = calculate_correlation_matrix(asset_returns)
                            avg_rolling_corr, pair_rolling_corr = calculate_rolling_correlation(asset_returns, window=st.session_state['rolling_window'])
                            risk_contribution = calculate_risk_contribution(asset_returns, active_weights)
                            diversification_ratio = calculate_diversification_ratio(asset_returns, active_weights)
                            portfolio_metrics['Diversification Ratio'] = diversification_ratio
                            rolling_vol, rolling_sharpe, rolling_sortino = calculate_rolling_metrics(portfolio_returns, st.session_state['rolling_window'], st.session_state['risk_free_rate'])

                            # Optimización con restricciones
                            mvp_weights, msr_weights, mvp_performance, msr_performance, frontier_df = optimize_portfolio(
                                asset_data_final,
                                risk_free_rate=st.session_state['risk_free_rate'],
                                fixed_income_assets=st.session_state['fixed_income_selection'], # Pasar selección
                                money_market_assets=st.session_state['money_market_selection']  # Pasar selección
                            )

                            current_portfolio_performance_opt = None
                            if active_weights and not asset_data_final.empty:
                                try:
                                    prices_cleaned_opt = asset_data_final.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='any')
                                    if not prices_cleaned_opt.empty and prices_cleaned_opt.shape[1] >=1 :
                                        mu_current = expected_returns.mean_historical_return(prices_cleaned_opt, frequency=252)
                                        S_current = risk_models.CovarianceShrinkage(prices_cleaned_opt, frequency=252).ledoit_wolf()
                                        weights_array = np.array([active_weights.get(asset, 0) for asset in prices_cleaned_opt.columns])
                                        weights_sum = weights_array.sum()
                                        weights_array_norm = weights_array / weights_sum if weights_sum > 1e-9 else weights_array
                                        current_ret = pypfopt.objective_functions.portfolio_return(weights_array_norm, mu_current, negative=False)
                                        current_variance = weights_array_norm.T @ S_current @ weights_array_norm
                                        current_vol = np.sqrt(current_variance) if current_variance >= 0 else np.nan
                                        current_sharpe = (current_ret - st.session_state['risk_free_rate']) / current_vol if pd.notna(current_vol) and current_vol > 1e-9 else np.nan
                                        current_portfolio_performance_opt = {'expected_return': current_ret, 'annual_volatility': current_vol, 'sharpe_ratio': current_sharpe}
                                except Exception as e_curr: st.warning(f"No se pudo calcular rendimiento/riesgo cartera actual para optimización: {e_curr}")

                            results_dict = {
                                'portfolio_value': portfolio_value, 'portfolio_metrics': portfolio_metrics, 'plot_data_normalized': plot_data_normalized,
                                'individual_metrics': individual_metrics, 'individual_asset_benchmark_metrics': individual_asset_benchmark_metrics,
                                'asset_returns': asset_returns, 'benchmark_returns': benchmark_returns, 'corr_matrix': corr_matrix,
                                'avg_rolling_corr': avg_rolling_corr, 'pair_rolling_corr': pair_rolling_corr, 'risk_contribution': risk_contribution,
                                'rolling_vol': rolling_vol, 'rolling_sharpe': rolling_sharpe, 'rolling_sortino': rolling_sortino,
                                'rolling_beta_portfolio': rolling_beta_portfolio, 'weights': active_weights, 'rolling_window_used': st.session_state['rolling_window'],
                                'benchmark_col_used': benchmark_col if benchmark_data_final is not None and benchmark_returns is not None else None,
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
    # asset_returns = results.get('asset_returns') # No usado directamente en UI, pero disponible
    benchmark_returns = results.get('benchmark_returns')
    corr_matrix = results.get('corr_matrix')
    avg_rolling_corr = results.get('avg_rolling_corr')
    # pair_rolling_corr = results.get('pair_rolling_corr') # No usado directamente en UI, pero disponible
    risk_contribution = results.get('risk_contribution', pd.Series(dtype=float))
    rolling_vol, rolling_sharpe, rolling_sortino = results.get('rolling_vol'), results.get('rolling_sharpe'), results.get('rolling_sortino')
    rolling_beta_portfolio = results.get('rolling_beta_portfolio')
    # used_weights = results.get('weights', {}) # No usado directamente en UI, pero disponible
    rolling_window_used = results.get('rolling_window_used', 0)
    mvp_weights, msr_weights = results.get('mvp_weights'), results.get('msr_weights')
    mvp_performance, msr_performance = results.get('mvp_performance'), results.get('msr_performance')
    frontier_df = results.get('frontier_df')
    current_portfolio_performance_opt = results.get('current_portfolio_performance_opt')

    tabs_titles = ["📊 Visión General", "🔗 Correlación", "🧩 Activos y Riesgo", "🔬 Optimización"]
    if benchmark_col_used and benchmark_returns is not None:
        tabs_titles.insert(1, f"🆚 vs {benchmark_col_used}")
        tabs = st.tabs(tabs_titles)
        tab1, tab_bm, tab_corr, tab_risk, tab_opt = tabs[0], tabs[1], tabs[2], tabs[3], tabs[4]
    else:
        tabs = st.tabs(tabs_titles)
        tab1, tab_corr, tab_risk, tab_opt = tabs[0], tabs[1], tabs[2], tabs[3]
        tab_bm = None

    with tab1: # Visión General
        st.header("Visión General de la Cartera")
        st.subheader("Métricas Principales (Cartera Total)")
        col1, col2, col3 = st.columns(3)
        try: locale.setlocale(locale.LC_ALL, 'es_ES.UTF-8')
        except locale.Error: locale.setlocale(locale.LC_ALL, '')
        col1.metric("Rentabilidad Total", f"{portfolio_metrics.get('Rentabilidad Total', np.nan):.2%}")
        col2.metric("Rentab. Anual (CAGR)", f"{portfolio_metrics.get('Rentabilidad Anualizada (CAGR)', np.nan):.2%}")
        col3.metric("Volatilidad Anualizada", f"{portfolio_metrics.get('Volatilidad Anualizada', np.nan):.2%}")
        col1b, col2b, col3b = st.columns(3)
        col1b.metric("Máximo Drawdown", f"{portfolio_metrics.get('Máximo Drawdown', np.nan):.2%}")
        col2b.metric("Ratio de Sharpe", f"{portfolio_metrics.get('Ratio de Sharpe', np.nan):.2f}")
        col3b.metric("Ratio Sortino", f"{portfolio_metrics.get('Sortino Ratio', np.nan):.2f}")
        col1c, _, _ = st.columns(3)
        col1c.metric("Ratio Diversificación", f"{portfolio_metrics.get('Diversification Ratio', np.nan):.2f}", help="Vol. media pond. / Vol. Cartera")
        st.markdown("---")
        st.subheader("Evolución Normalizada (Base 100)")
        if plot_data_normalized is not None and not plot_data_normalized.empty:
            plot_data_to_show = plot_data_normalized.dropna(axis=1, how='all')
            if not plot_data_to_show.empty:
                fig_perf = px.line(plot_data_to_show, title="Cartera vs. Activos" + (f" vs {benchmark_col_used}" if benchmark_col_used else ""), labels={'value': 'Valor (Base 100)', 'variable': 'Activo'})
                fig_perf.update_layout(xaxis_title="Fecha", yaxis_title="Valor Normalizado", legend_title_text='Activos')
                st.plotly_chart(fig_perf, use_container_width=True)
            else: st.warning("No hay datos válidos para gráfico de evolución.")
        else: st.warning("No hay datos para gráfico de evolución.")
        st.markdown("---")
        st.subheader(f"Análisis Rodante de la Cartera (Ventana: {rolling_window_used} días)")
        col_roll1, col_roll2, col_roll3 = st.columns(3)
        with col_roll1:
            if rolling_vol is not None and not rolling_vol.empty:
                fig = px.line(rolling_vol, title="Volatilidad Anualizada Rodante", labels={'value': 'Vol. Anual.'})
                fig.update_layout(showlegend=False, yaxis_title="Volatilidad Anualizada"); st.plotly_chart(fig, use_container_width=True)
            else: st.warning("Insuficientes datos para volatilidad rodante.")
        with col_roll2:
            if rolling_sharpe is not None and not rolling_sharpe.empty:
                fig = px.line(rolling_sharpe, title="Ratio de Sharpe Rodante", labels={'value': 'Sharpe'})
                fig.update_layout(showlegend=False, yaxis_title="Ratio de Sharpe"); st.plotly_chart(fig, use_container_width=True)
            else: st.warning("Insuficientes datos para Sharpe rodante.")
        with col_roll3:
            if rolling_sortino is not None and not rolling_sortino.empty:
                fig = px.line(rolling_sortino, title="Ratio de Sortino Rodante", labels={'value': 'Sortino'})
                fig.update_layout(showlegend=False, yaxis_title="Ratio de Sortino"); st.plotly_chart(fig, use_container_width=True)
            else: st.warning("Insuficientes datos para Sortino rodante.")

    if tab_bm: # Benchmark
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
            else: st.warning("Insuficientes datos para Beta rodante.")

    with tab_corr: # Correlación
        st.header("Análisis de Correlación entre Activos")
        st.subheader("Matriz de Correlación (Período Completo)")
        if corr_matrix is not None and not corr_matrix.empty:
            if corr_matrix.shape[0] > 1 and corr_matrix.shape[1] > 1:
                fig_heatmap, ax = plt.subplots(figsize=(max(6, corr_matrix.shape[1]*0.8), max(5, corr_matrix.shape[0]*0.7)))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax, vmin=-1, vmax=1, annot_kws={"size": 8})
                ax.set_title('Correlación entre Activos'); plt.xticks(rotation=45, ha='right', fontsize=8); plt.yticks(rotation=0, fontsize=8)
                plt.tight_layout(); st.pyplot(fig_heatmap)
            elif corr_matrix.shape[0] == 1: st.info("Solo un activo, no hay matriz de correlación.")
            else: st.warning("No se pudo calcular matriz de correlación.")
        else: st.warning("No se pudo calcular matriz de correlación.")
        st.markdown("---")
        st.subheader(f"Correlación Rodante (Ventana: {rolling_window_used} días)")
        if avg_rolling_corr is not None and not avg_rolling_corr.empty:
            fig = px.line(avg_rolling_corr, title="Correlación Promedio Rodante", labels={'value': 'Corr. Promedio'})
            fig.update_layout(showlegend=False, yaxis_title="Correlación Promedio"); st.plotly_chart(fig, use_container_width=True)
            asset_list_corr = results.get('asset_returns').columns.tolist() if results.get('asset_returns') is not None else []
            if len(asset_list_corr) >= 2:
                pair_options = [(asset_list_corr[i], asset_list_corr[j]) for i in range(len(asset_list_corr)) for j in range(i + 1, len(asset_list_corr))]
                if pair_options:
                    selected_pairs = st.multiselect("Pares para correlación rodante específica:", options=pair_options, format_func=lambda p: f"{p[0]} vs {p[1]}")
                    if selected_pairs:
                        _, specific_pair_corr = calculate_rolling_correlation(results.get('asset_returns'), window=rolling_window_used, pair_list=selected_pairs)
                        if specific_pair_corr is not None and not specific_pair_corr.empty:
                            fig_pair = px.line(specific_pair_corr, title="Correlación Rodante (Pares)", labels={'value': 'Correlación', 'variable': 'Par'})
                            fig_pair.update_layout(yaxis_title="Correlación", legend_title_text='Pares'); st.plotly_chart(fig_pair, use_container_width=True)
                        else: st.warning("No se pudo calcular correlación para pares.")
        else: st.warning("Datos insuficientes para correlación rodante.")

    with tab_risk: # Activos y Riesgo
        st.header("Análisis de Activos Individuales y Riesgo")
        st.subheader("Posicionamiento Riesgo/Retorno")
        if individual_metrics is not None and not individual_metrics.empty and portfolio_metrics:
            scatter_data = individual_metrics.T.copy()
            scatter_data['Rentab. Anual (CAGR)'] = pd.to_numeric(scatter_data.get('Rentabilidad Anualizada (CAGR)'), errors='coerce')
            scatter_data['Volatilidad Anualizada'] = pd.to_numeric(scatter_data.get('Volatilidad Anualizada'), errors='coerce')
            valid_cols_scatter = ['Rentab. Anual (CAGR)', 'Volatilidad Anualizada']
            if all(col in scatter_data.columns for col in valid_cols_scatter):
                scatter_data.dropna(subset=valid_cols_scatter, inplace=True)
            else: scatter_data = pd.DataFrame()
            portfolio_scatter = pd.DataFrame({'Rentab. Anual (CAGR)': [portfolio_metrics.get('Rentabilidad Anualizada (CAGR)')], 'Volatilidad Anualizada': [portfolio_metrics.get('Volatilidad Anualizada')]}, index=['Cartera Total'])
            portfolio_scatter['Rentab. Anual (CAGR)'] = pd.to_numeric(portfolio_scatter['Rentab. Anual (CAGR)'], errors='coerce')
            portfolio_scatter['Volatilidad Anualizada'] = pd.to_numeric(portfolio_scatter['Volatilidad Anualizada'], errors='coerce')
            portfolio_scatter.dropna(inplace=True)
            if not scatter_data.empty or not portfolio_scatter.empty:
                scatter_data_final = pd.concat([scatter_data.assign(Tipo='Activo Individual'), portfolio_scatter.assign(Tipo='Cartera Total')])
                if not scatter_data_final.empty and all(col in scatter_data_final.columns for col in valid_cols_scatter):
                    fig = px.scatter(scatter_data_final, x='Volatilidad Anualizada', y='Rentab. Anual (CAGR)', text=scatter_data_final.index, color='Tipo', hover_name=scatter_data_final.index, title="Rentabilidad vs. Volatilidad")
                    fig.update_traces(textposition='top center'); fig.update_layout(xaxis_title="Volatilidad Anualizada", yaxis_title="Rentab. Anual (CAGR)", xaxis_tickformat=".1%", yaxis_tickformat=".1%")
                    st.plotly_chart(fig, use_container_width=True)
                else: st.warning("No hay datos válidos para gráfico Riesgo/Retorno.")
            else: st.warning("No hay datos válidos para gráfico Riesgo/Retorno.")
        else: st.warning("Faltan métricas para gráfico Riesgo/Retorno.")
        st.markdown("---")
        st.subheader("Contribución de Activos a la Volatilidad de Cartera")
        if risk_contribution is not None and not risk_contribution.empty:
            risk_contribution_pct = (risk_contribution * 100).reset_index(); risk_contribution_pct.columns = ['Activo', 'Contribución al Riesgo (%)']
            fig = px.bar(risk_contribution_pct.sort_values(by='Contribución al Riesgo (%)', ascending=False), x='Activo', y='Contribución al Riesgo (%)', title='Contribución Porcentual al Riesgo Total', text='Contribución al Riesgo (%)')
            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside'); fig.update_layout(yaxis_title="Contribución (%)", xaxis_title="Activo")
            st.plotly_chart(fig, use_container_width=True)
        else: st.warning("No se pudo calcular contribución al riesgo.")
        st.markdown("---")
        st.subheader("Ranking Avanzado de Activos")
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
            st.dataframe(ranking_display)
        else: st.warning("Faltan datos para ranking avanzado.")

    with tab_opt: # Optimización
        st.header("🔬 Optimización de Cartera (Frontera Eficiente)")
        st.markdown("Esta sección calcula la Frontera Eficiente. **Nota:** Basado en datos históricos, no garantiza resultados futuros. Solo posiciones largas.")
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
                    mvp_df['Peso'] = mvp_df['Peso Sugerido'].map('{:.2%}'.format) # Crear columna formateada
                    st.dataframe(mvp_df.sort_values(by='Peso Sugerido', ascending=False)[['Peso']]) # Sort by numeric, display formatted
                else: st.warning("No se pudo calcular MVP.")
            with col_opt2:
                st.markdown("**Cartera Máximo Sharpe (MSR)**")
                if msr_weights:
                    msr_df = pd.DataFrame.from_dict(msr_weights, orient='index', columns=['Peso Sugerido']).rename_axis('Activo')
                    msr_df['Peso'] = msr_df['Peso Sugerido'].map('{:.2%}'.format) # Crear columna formateada
                    st.dataframe(msr_df.sort_values(by='Peso Sugerido', ascending=False)[['Peso']]) # Sort by numeric, display formatted
                else: st.warning("No se pudo calcular MSR.")
        else: st.warning("No se pudo calcular frontera eficiente. Revisa datos y rango.")

elif 'run_results' not in st.session_state or st.session_state['run_results'] is None:
    if data is not None: st.info("Configura parámetros y haz clic en 'Ejecutar Análisis Completo'.")
    elif uploaded_file is not None: pass # Error ya mostrado
    else: st.info("Por favor, carga un archivo CSV para comenzar.")
elif not st.session_state['run_results'].get('run_successful', False):
    st.error("La ejecución del análisis falló. Revisa mensajes o parámetros.")

st.sidebar.markdown("---"); st.sidebar.markdown("Backtester Quant v4.5"); st.sidebar.markdown("Dios Familia y Cojones")

