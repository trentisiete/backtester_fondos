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

# NUEVO: Importar PyPortfolioOpt
import pypfopt
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions
from pypfopt import plotting # Para helpers de ploteo si son necesarios


# --- Configuración de la Página y Estilo ---
st.set_page_config(
    page_title="Backtester Quant v4.4 (Optimización)", # Título actualizado
    page_icon="💡", # Icono actualizado
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
                        st.info("Detectado CSV con separador ';' y decimal '.'")
                    except (pd.errors.ParserError, ValueError, IndexError, KeyError):
                         stringio.seek(0) # Punto decimal, coma separador
                         data = pd.read_csv(stringio, sep=',', decimal='.', parse_dates=[0], index_col=0)
                         st.info("Detectado CSV con separador ',' y decimal '.'")

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
    # Filtrar datos de precios solo para los activos de la cartera
    asset_columns = list(weights_dict.keys())
    # Asegurar que start_date y end_date son Timestamps
    start_date_ts = pd.to_datetime(start_date)
    end_date_ts = pd.to_datetime(end_date)

    # Filtrar asegurando que las fechas existen en el índice
    data_in_range = data.loc[data.index.intersection(pd.date_range(start_date_ts, end_date_ts))]
    if data_in_range.empty:
         st.warning("No hay datos en el índice para el rango de fechas seleccionado.")
         return None, None

    prices = data_in_range[asset_columns].copy()

    if prices.empty or prices.isnull().all().all():
        st.warning("No hay datos válidos para los activos en el rango de fechas seleccionado.")
        return None, None

    # Rellenar NaNs (método actualizado)
    prices.ffill(inplace=True)
    prices.bfill(inplace=True)

    # Comprobar si quedan NaNs en la primera fila después de rellenar
    if prices.iloc[0].isnull().any():
        missing_funds = prices.columns[prices.iloc[0].isnull()].tolist()
        st.warning(f"Faltan datos iniciales para: {', '.join(missing_funds)} en {prices.index[0].date()}.")
        # Intentar encontrar la primera fecha válida para TODOS los activos
        first_valid_date = prices.dropna(axis=0, how='any').index.min()
        if pd.isna(first_valid_date) or first_valid_date > end_date_ts:
            st.error("No hay datos comunes suficientes para iniciar el backtest en el rango seleccionado.")
            return None, None
        # Refiltrar prices desde la primera fecha válida
        prices = prices.loc[first_valid_date:].copy()
        st.warning(f"Backtest comenzará en {prices.index[0].date()}.")
        if prices.empty: return None, None

    # Normalizar pesos si es necesario (aunque se hace antes, doble check)
    total_weight = sum(weights_dict.values())
    if not np.isclose(total_weight, 1.0) and total_weight != 0:
        weights_dict = {k: v / total_weight for k, v in weights_dict.items()}

    # Inicialización de la cartera
    portfolio_value = pd.Series(index=prices.index, dtype=float)
    if not prices.empty:
         portfolio_value.loc[prices.index[0]] = initial_investment
    else:
         st.error("No se pueden inicializar los valores de la cartera.")
         return None, None

    current_weights = weights_dict.copy()
    last_rebalance_date = prices.index[0]

    initial_alloc = {fund: initial_investment * weight for fund, weight in current_weights.items()}

    # Calcular participaciones iniciales
    shares = {}
    for fund in current_weights:
        initial_price = prices[fund].iloc[0]
        if pd.notna(initial_price) and initial_price != 0:
            shares[fund] = initial_alloc[fund] / initial_price
        else:
            shares[fund] = 0
            st.warning(f"Precio inicial inválido para {fund} en {prices.index[0].date()}. Peso inicial no aplicado.")

    # Mapeo de frecuencia de rebalanceo
    rebalance_offset = {
        'Mensual': pd.DateOffset(months=1),
        'Trimestral': pd.DateOffset(months=3),
        'Anual': pd.DateOffset(years=1),
        'No Rebalancear': None
    }
    offset = rebalance_offset[rebalance_freq]

    # Simulación día a día
    for i in range(1, len(prices)):
        current_date = prices.index[i]
        prev_date = prices.index[i-1]

        # Calcular valor actual basado en precios del día o anteriores si faltan
        current_portfolio_value = 0.0
        for fund in shares:
            # Asegurarse que la fecha actual existe en el índice de precios
            if current_date in prices.index:
                current_price = prices.loc[current_date, fund] # Usar .loc para evitar warnings
                if pd.notna(current_price):
                    current_portfolio_value += shares[fund] * current_price
                else:
                    # Usar precio anterior si falta el actual
                    if prev_date in prices.index:
                        prev_price = prices.loc[prev_date, fund]
                        if pd.notna(prev_price):
                            current_portfolio_value += shares[fund] * prev_price
                    # Si ambos faltan, el valor de este fondo no contribuye en este paso
            else:
                 # Si la fecha actual no existe (poco probable con ffill/bfill, pero por seguridad)
                 # mantener el valor anterior de la cartera
                 if prev_date in portfolio_value.index and pd.notna(portfolio_value.loc[prev_date]):
                      current_portfolio_value = portfolio_value.loc[prev_date]
                 else: # Si no hay valor previo, no podemos continuar
                      st.error(f"Fecha {current_date} no encontrada y sin valor previo de cartera.")
                      return None, None
                 break # Salir del bucle de fondos para esta fecha

        # Manejar posible valor NaN o 0 si todos los precios fallan o son 0
        if pd.isna(current_portfolio_value) or (current_portfolio_value == 0 and prev_date in portfolio_value.index and portfolio_value.loc[prev_date] != 0):
            prev_portfolio_value = portfolio_value.loc[prev_date]
            if pd.notna(prev_portfolio_value):
                 current_portfolio_value = prev_portfolio_value # Mantener valor anterior
                 # st.warning(f"No se pudo calcular el valor de la cartera en {current_date}. Se mantiene el valor anterior.") # Opcional: puede ser muy verboso
            else: # Si el anterior también es NaN, problema grave
                 st.error(f"Error irrecuperable calculando valor de cartera en {current_date}.")
                 return None, None # Abortar

        portfolio_value.loc[current_date] = current_portfolio_value

        # Rebalanceo
        if offset and current_date >= last_rebalance_date + offset:
            # Asegurarse que el valor actual de la cartera es válido para rebalancear
            if pd.notna(current_portfolio_value) and current_portfolio_value > 0:
                target_alloc = {fund: current_portfolio_value * weight for fund, weight in weights_dict.items()}
                for fund in weights_dict:
                     # Asegurarse que la fecha actual existe para el rebalanceo
                     if current_date in prices.index:
                         current_price_rebal = prices.loc[current_date, fund]
                         if pd.notna(current_price_rebal) and current_price_rebal != 0:
                              shares[fund] = target_alloc[fund] / current_price_rebal
                         else:
                              st.warning(f"Precio inválido para {fund} en fecha de rebalanceo {current_date}. No se rebalanceó este fondo.")
                              pass # Mantener participaciones anteriores
                     else:
                          st.warning(f"Fecha {current_date} no encontrada para rebalanceo de {fund}.")

                last_rebalance_date = current_date
            else:
                 st.warning(f"Valor de cartera inválido ({current_portfolio_value}) en fecha de rebalanceo {current_date}. Rebalanceo omitido.")


    portfolio_returns = portfolio_value.pct_change().dropna()

    # Devolver valor y retornos sin NaNs finales
    return portfolio_value.dropna(), portfolio_returns


def calculate_metrics(portfolio_value, portfolio_returns):
    """Calcula las métricas de rendimiento de la cartera."""
    if portfolio_returns is None or portfolio_value is None or portfolio_returns.empty or portfolio_value.empty:
        return {}

    metrics = {}
    start_value = portfolio_value.iloc[0]
    end_value = portfolio_value.iloc[-1]

    # Rentabilidad Total
    total_return = (end_value / start_value) - 1 if start_value != 0 else 0
    metrics['Rentabilidad Total'] = total_return

    # CAGR (Rentabilidad Anualizada Compuesta)
    years = (portfolio_value.index[-1] - portfolio_value.index[0]).days / 365.25
    if years <= 0 or start_value <= 0:
        cagr = 0
    else:
        # Evitar errores con base negativa o cero en potencia
        base = end_value / start_value
        if base <= 0:
             cagr = -1.0 # O algún otro indicador de pérdida total o error
        else:
             cagr = (base ** (1 / years)) - 1
    metrics['Rentabilidad Anualizada (CAGR)'] = cagr

    # Volatilidad Anualizada
    volatility = portfolio_returns.std() * np.sqrt(252) if not portfolio_returns.empty else 0
    metrics['Volatilidad Anualizada'] = volatility

    # Ratio de Sharpe (Usa tasa libre de riesgo de sesión)
    risk_free_rate_annual = st.session_state.get('risk_free_rate', 0.0)
    sharpe_ratio = (cagr - risk_free_rate_annual) / volatility if volatility != 0 else np.nan # Usar NaN si vol es 0
    metrics['Ratio de Sharpe'] = sharpe_ratio # Nombre genérico

    # Máximo Drawdown
    rolling_max = portfolio_value.cummax()
    daily_drawdown = portfolio_value / rolling_max - 1
    max_drawdown = daily_drawdown.min() if not daily_drawdown.empty else 0
    metrics['Máximo Drawdown'] = max_drawdown

    return metrics

def calculate_individual_metrics(fund_prices):
    """Calcula métricas clave para cada fondo individual."""
    if fund_prices is None or fund_prices.empty:
        return pd.DataFrame()

    individual_metrics = {}
    # Asegurarse de que los precios son numéricos
    fund_prices = fund_prices.apply(pd.to_numeric, errors='coerce')
    # Rellenar posibles NaNs introducidos por coerce (método actualizado)
    fund_prices.ffill(inplace=True)
    fund_prices.bfill(inplace=True)

    # Calcular retornos solo después de limpiar NaNs
    fund_returns = fund_prices.pct_change().dropna(how='all') # No dropear filas si solo 1 activo es NaN

    for fund_name in fund_prices.columns:
        fund_series = fund_prices[fund_name].dropna()
        # Asegurarse de que fund_ret_series tenga el índice alineado y no esté vacío
        if fund_name in fund_returns.columns:
            fund_ret_series = fund_returns[fund_name].dropna()
        else:
            fund_ret_series = pd.Series(dtype=float) # Vacío si no hay retornos

        # Comprobar si hay suficientes datos y valor inicial válido
        if fund_series.empty or fund_series.shape[0] < 2 or fund_series.iloc[0] == 0 or fund_ret_series.empty:
            individual_metrics[fund_name] = {
                'Rentabilidad Total': np.nan, 'Rentabilidad Anualizada (CAGR)': np.nan,
                'Volatilidad Anualizada': np.nan, 'Ratio de Sharpe': np.nan,
                'Máximo Drawdown': np.nan, 'Sortino Ratio': np.nan # Añadir Sortino
            }
            continue

        metrics = calculate_metrics(fund_series, fund_ret_series) # Reutilizar función base
        # Calcular Sortino individual
        metrics['Sortino Ratio'] = calculate_sortino_ratio(fund_ret_series, required_return=st.session_state.get('risk_free_rate', 0.0))
        individual_metrics[fund_name] = metrics # Guardar diccionario completo

    return pd.DataFrame(individual_metrics)

def calculate_returns(prices):
    """Calcula los retornos diarios a partir de los precios."""
    if prices is None or prices.empty:
        return None
    # Asegurarse que los precios son numéricos antes de calcular
    prices_numeric = prices.apply(pd.to_numeric, errors='coerce')
    # Si solo hay una columna (es una Serie), pct_change funciona directamente
    if isinstance(prices_numeric, pd.Series):
         # Asegurar que no haya ceros antes de pct_change si es posible
         prices_numeric = prices_numeric.replace(0, np.nan).ffill().bfill()
         # Comprobar si después de rellenar sigue habiendo NaNs (si toda la serie era 0 o NaN)
         if prices_numeric.isnull().all(): return None
         return prices_numeric.pct_change().dropna()
    else: # Si es DataFrame
         prices_numeric = prices_numeric.replace(0, np.nan).ffill(axis=0).bfill(axis=0)
         if prices_numeric.isnull().all().all(): return None
         return prices_numeric.pct_change().dropna(how='all') # Evitar dropear filas si solo un activo es NaN

def calculate_sortino_ratio(returns, required_return=0.0):
    """
    Calcula el ratio de Sortino. Asume retornos diarios.
    Usa empyrical si está disponible, si no, cálculo manual.
    """
    # Usar .size para comprobar si el array NumPy está vacío
    if returns is None or (isinstance(returns, np.ndarray) and returns.size == 0) or (isinstance(returns, pd.Series) and returns.empty):
        return np.nan

    try:
        # Intentar usar empyrical primero (más robusto)
        # Empyrical espera MAR diario y devuelve ratio anualizado
        daily_required_return = (1 + required_return)**(1/252) - 1 if required_return != 0 else 0.0
        # Empyrical necesita una Serie de Pandas
        if isinstance(returns, np.ndarray):
            # Asegurar que no hay NaNs o Infs en el array NumPy
            if not np.all(np.isfinite(returns)):
                 return np.nan
            returns_series = pd.Series(returns)
        else: # Si ya es Serie, comprobar igual
             if not np.all(np.isfinite(returns.dropna())):
                  return np.nan
             returns_series = returns.dropna() # Quitar NaNs por si acaso

        # Comprobar longitud mínima para Empyrical
        if len(returns_series) < 2:
             return np.nan

        sortino = empyrical.sortino_ratio(returns_series, required_return=daily_required_return)
        # Empyrical puede devolver -inf si no hay retornos negativos, convertir a NaN
        return sortino if np.isfinite(sortino) else np.nan

    except Exception as e_emp:
         # st.warning(f"Fallo al usar empyrical para Sortino ({e_emp}), usando cálculo manual.") # Opcional
         # Fallback a cálculo manual simple si empyrical falla o no está
         if isinstance(returns, pd.Series):
             returns = returns.values
         elif not isinstance(returns, np.ndarray):
             returns = np.array(returns)

         # Asegurar que no hay NaNs/Infs
         returns = returns[np.isfinite(returns)]
         if returns.size < 2: return np.nan

         # Filtrar retornos por debajo del MAR (required_return ANUAL para comparación directa)
         daily_required_return = (1 + required_return)**(1/252) - 1 if required_return != 0 else 0.0
         downside_returns = returns[returns < daily_required_return]

         if downside_returns.size == 0:
             # Si el retorno medio es positivo, Sortino es infinito, devolver NaN o un número grande
             return np.inf if np.mean(returns) > daily_required_return else 0.0

         mean_return = np.mean(returns)
         # Calcular desviación estándar solo de los retornos negativos respecto al MAR diario
         downside_std = np.sqrt(np.mean(np.square(downside_returns - daily_required_return)))

         if downside_std == 0:
             # Si std es 0 pero hay retornos negativos (raro), devolver 0 o NaN
             return np.nan

         # Calcular ratio usando retornos medios diarios y std downside diaria, luego anualizar
         sortino_manual = (mean_return - daily_required_return) / downside_std * np.sqrt(252)
         return sortino_manual


# --- Funciones de Análisis Avanzado ---

def calculate_covariance_matrix(returns):
    """Calcula la matriz de covarianza anualizada."""
    if returns is None or returns.empty: return pd.DataFrame()
    return returns.cov() * 252

def calculate_correlation_matrix(returns):
    """Calcula la matriz de correlación."""
    if returns is None or returns.empty: return pd.DataFrame()
    return returns.corr()

def calculate_rolling_correlation(returns, window, pair_list=None):
    """Calcula la correlación rodante promedio o para pares específicos."""
    if returns is None or returns.empty or returns.shape[1] < 2 or window <= 0 or window > returns.shape[0]:
        return None, None

    rolling_corr_pairs = None
    if pair_list:
        # Inicializar con índice correcto para evitar errores de asignación
        rolling_corr_pairs = pd.DataFrame(index=returns.index[window-1:])
        for pair in pair_list:
            if len(pair) == 2 and pair[0] in returns.columns and pair[1] in returns.columns:
                col1, col2 = pair
                try:
                    pair_corr = returns[col1].rolling(window=window).corr(returns[col2])
                    # Asignar usando .loc para evitar SettingWithCopyWarning
                    rolling_corr_pairs.loc[:, f'{col1} vs {col2}'] = pair_corr
                except Exception as e:
                    st.warning(f"No se pudo calcular corr rodante para {col1} vs {col2}: {e}")
        rolling_corr_pairs = rolling_corr_pairs.dropna(how='all')

    rolling_corr_avg = None
    try:
        # Usar un método más directo para la media rodante
        rolling_corr_avg = returns.rolling(window=window).corr().groupby(level=0).apply(
            lambda x: x.values[np.triu_indices_from(x.values, k=1)].mean() # Media de triángulo superior (sin diagonal)
        )
        rolling_corr_avg.name = "Correlación Promedio Rodante"
        rolling_corr_avg = rolling_corr_avg.dropna()
    except Exception as e:
         st.warning(f"No se pudo calcular la correlación promedio rodante: {e}")

    return rolling_corr_avg, rolling_corr_pairs

def calculate_risk_contribution(returns, weights_dict):
    """Calcula la contribución porcentual de cada activo a la volatilidad total de la cartera."""
    if returns is None or returns.empty or not weights_dict or returns.isnull().values.any():
        # st.warning("Datos de retorno inválidos para calcular contribución al riesgo.")
        return pd.Series(dtype=float)

    funds = list(weights_dict.keys())
    # Asegurarse que los fondos existen en las columnas de retornos
    funds_in_returns = [f for f in funds if f in returns.columns]
    if not funds_in_returns:
         st.warning("Ninguno de los fondos con peso tiene datos de retorno válidos.")
         return pd.Series(dtype=float)
    # Ajustar pesos y fondos a los que realmente tienen datos
    weights_dict_adj = {f: weights_dict[f] for f in funds_in_returns}
    funds_adj = funds_in_returns
    weights_adj = np.array([weights_dict_adj[f] for f in funds_adj])
    # Normalizar pesos ajustados por si se eliminó algún fondo
    total_weight_adj = weights_adj.sum()
    if total_weight_adj <= 1e-9: return pd.Series(dtype=float) # Evitar división por cero
    weights_norm_adj = weights_adj / total_weight_adj


    returns_subset = returns[funds_adj]


    try:
        cov_matrix = returns_subset.cov() * 252
        if cov_matrix.isnull().values.any():
             # st.warning("Matriz de covarianza contiene NaNs.")
             return pd.Series(dtype=float)

        portfolio_var = weights_norm_adj.T @ cov_matrix @ weights_norm_adj
        if portfolio_var <= 1e-10: # Usar tolerancia pequeña en lugar de 0
            # Si la varianza es casi cero, la contribución es proporcional al peso normalizado
            return pd.Series({fund: weights_norm_adj[i] for i, fund in enumerate(funds_adj)}, name="Contribución al Riesgo (%)")

        portfolio_vol = np.sqrt(portfolio_var)

        mcsr = (cov_matrix.values @ weights_norm_adj) / portfolio_vol
        cctr = weights_norm_adj * mcsr
        risk_contribution_percent = cctr / portfolio_vol
        risk_contribution_series = pd.Series(risk_contribution_percent, index=funds_adj, name="Contribución al Riesgo (%)")
        risk_contribution_series.fillna(0.0, inplace=True) # Rellenar NaNs si MCSR fue NaN

    except Exception as e:
        st.error(f"Error calculando contribución al riesgo: {e}")
        return pd.Series(dtype=float)

    return risk_contribution_series

def calculate_rolling_metrics(portfolio_returns, window, required_return=0.0):
    """Calcula la volatilidad, Sharpe y Sortino rodantes."""
    if portfolio_returns is None or portfolio_returns.empty or window <= 1 or window > portfolio_returns.shape[0]:
        # st.warning(f"Datos insuficientes para ventana rodante {window}")
        return None, None, None

    rolling_vol = portfolio_returns.rolling(window=window).std(ddof=1) * np.sqrt(252) # ddof=1 es default
    rolling_vol = rolling_vol.dropna()
    rolling_vol.name = f"Volatilidad Rodante ({window}d)"

    rolling_annual_ret = portfolio_returns.rolling(window=window).mean() * 252

    rolling_sharpe = (rolling_annual_ret - required_return) / rolling_vol # Usar Rf anual
    rolling_sharpe.replace([np.inf, -np.inf], np.nan, inplace=True)
    rolling_sharpe = rolling_sharpe.dropna()
    rolling_sharpe.name = f"Sharpe Rodante ({window}d)"

    # Sortino rodante (usando la función definida arriba)
    # Pasar MAR diario a la función lambda que llama a calculate_sortino_ratio
    daily_required_return = (1 + required_return)**(1/252) - 1 if required_return != 0 else 0.0
    rolling_sortino = portfolio_returns.rolling(window=window).apply(
        lambda x: calculate_sortino_ratio(x, required_return=daily_required_return),
        raw=True # raw=True pasa arrays NumPy a la función lambda
    )
    rolling_sortino = rolling_sortino.dropna()
    rolling_sortino.name = f"Sortino Rodante ({window}d)"

    return rolling_vol, rolling_sharpe, rolling_sortino

def calculate_benchmark_metrics(portfolio_returns, benchmark_returns, risk_free_rate=0.0):
    """Calcula Alpha, Beta, Info Ratio, Tracking Error usando Empyrical."""
    metrics = {}
    if portfolio_returns is None or benchmark_returns is None or portfolio_returns.empty or benchmark_returns.empty:
        return metrics

    try:
        # Alinear retornos por fecha (muy importante)
        common_index = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common_index) < 2:
             st.warning("Pocos datos comunes entre cartera y benchmark para métricas relativas.")
             return {}
        portfolio_returns_aligned = portfolio_returns.loc[common_index]
        benchmark_returns_aligned = benchmark_returns.loc[common_index]

        # Empyrical espera retornos diarios y risk_free diario
        daily_rf = (1 + risk_free_rate)**(1/252) - 1 if risk_free_rate != 0 else 0.0

        # Beta y Alpha (anualizado)
        # Asegurarse que no hay NaNs/Infs en las series alineadas
        portfolio_returns_aligned = portfolio_returns_aligned.replace([np.inf, -np.inf], np.nan).dropna()
        benchmark_returns_aligned = benchmark_returns_aligned.replace([np.inf, -np.inf], np.nan).dropna()
        common_index_final = portfolio_returns_aligned.index.intersection(benchmark_returns_aligned.index)
        if len(common_index_final) < 2: return {} # Re-chequear después de dropna
        portfolio_returns_aligned = portfolio_returns_aligned.loc[common_index_final]
        benchmark_returns_aligned = benchmark_returns_aligned.loc[common_index_final]


        metrics['Beta'] = empyrical.beta(portfolio_returns_aligned, benchmark_returns_aligned, risk_free=daily_rf)
        metrics['Alpha (anual)'] = empyrical.alpha(portfolio_returns_aligned, benchmark_returns_aligned, risk_free=daily_rf, annualization=252)

        # Tracking Error (anualizado) - Usando la std de los retornos en exceso
        excess_returns = portfolio_returns_aligned - benchmark_returns_aligned
        metrics['Tracking Error (anual)'] = excess_returns.std() * np.sqrt(252)

        # Information Ratio (Alpha / Tracking Error)
        # Asegurarse que Alpha y TE no son None antes de dividir
        alpha_val = metrics.get('Alpha (anual)')
        te_val = metrics.get('Tracking Error (anual)')
        if alpha_val is not None and te_val is not None and not pd.isna(alpha_val) and not pd.isna(te_val) and te_val != 0:
            metrics['Information Ratio'] = alpha_val / te_val
        else:
            metrics['Information Ratio'] = np.nan

    except Exception as e:
        st.error(f"Error calculando métricas vs benchmark: {e}")
        return {}

    return metrics

# CORREGIDO: Implementación manual de rolling beta
def calculate_rolling_beta(portfolio_returns, benchmark_returns, window):
    """Calcula Beta rodante manualmente usando Pandas rolling."""
    if portfolio_returns is None or benchmark_returns is None or portfolio_returns.empty or benchmark_returns.empty:
        return None
    # Asegurar ventana válida
    min_len = min(len(portfolio_returns), len(benchmark_returns))
    # Asegurar que la ventana no sea mayor que la longitud de los datos alineados
    common_index = portfolio_returns.index.intersection(benchmark_returns.index)
    if window <= 1 or window > len(common_index):
        st.warning(f"Ventana ({window}) inválida para longitud de datos comunes ({len(common_index)}).")
        return None

    try:
        # Alinear retornos (ya hecho antes, pero re-asegurar)
        portfolio_returns_aligned = portfolio_returns.loc[common_index]
        benchmark_returns_aligned = benchmark_returns.loc[common_index]

        # Combinar en un DataFrame para cálculo de covarianza rodante
        df_combined = pd.DataFrame({'portfolio': portfolio_returns_aligned, 'benchmark': benchmark_returns_aligned})

        # Calcular covarianza rodante entre cartera y benchmark
        rolling_cov = df_combined['portfolio'].rolling(window=window).cov(df_combined['benchmark'])

        # Calcular varianza rodante del benchmark
        rolling_var = df_combined['benchmark'].rolling(window=window).var()

        # Calcular Beta rodante: Cov / Var
        # Evitar división por cero si la varianza rodante es cero
        rolling_beta = (rolling_cov / rolling_var).replace([np.inf, -np.inf], np.nan)
        rolling_beta = rolling_beta.dropna() # Eliminar NaNs iniciales y los de división por cero
        rolling_beta.name = f"Beta Rodante ({window}d)"
        return rolling_beta
    except Exception as e:
        st.error(f"Error calculando beta rodante manual: {e}")
        return None


def calculate_diversification_ratio(asset_returns, weights_dict):
     """Calcula el ratio de diversificación."""
     if asset_returns is None or asset_returns.empty or not weights_dict: return np.nan

     funds = list(weights_dict.keys())
     # Usar solo fondos presentes en los retornos
     funds_in_returns = [f for f in funds if f in asset_returns.columns]
     if not funds_in_returns: return np.nan
     weights_adj = np.array([weights_dict[f] for f in funds_in_returns])
     # Normalizar pesos por si se excluyó alguno
     total_w = weights_adj.sum()
     if total_w <= 1e-9 : return np.nan # Evitar división por cero
     weights_norm = weights_adj / total_w

     asset_returns_subset = asset_returns[funds_in_returns]


     # Volatilidad individual anualizada
     asset_vols = asset_returns_subset.std() * np.sqrt(252)
     # Asegurarse que no haya NaNs en volatilidades individuales
     if asset_vols.isnull().any(): return np.nan
     weighted_avg_vol = np.sum(weights_norm * asset_vols) # Usar pesos normalizados

     # Volatilidad de cartera
     cov_matrix = asset_returns_subset.cov() * 252
     if cov_matrix.isnull().values.any(): return np.nan
     portfolio_var = weights_norm.T @ cov_matrix @ weights_norm # Usar pesos normalizados
     if portfolio_var <= 1e-10: return np.nan # Evitar división por cero o negativo
     portfolio_vol = np.sqrt(portfolio_var)

     if portfolio_vol <= 1e-9: return np.nan # Evitar división por cero
     diversification_ratio = weighted_avg_vol / portfolio_vol
     return diversification_ratio


# --- NUEVA Función para Optimización ---
def optimize_portfolio(prices, risk_free_rate=0.0):
    """
    Calcula la frontera eficiente y carteras óptimas (Min Vol, Max Sharpe).

    Args:
        prices (pd.DataFrame): DataFrame de precios históricos de los activos.
        risk_free_rate (float): Tasa libre de riesgo anual.

    Returns:
        tuple: Pesos MVP, Pesos Max Sharpe, Performance MVP, Performance Max Sharpe,
               DataFrame de puntos de la frontera eficiente (retorno, volatilidad, sharpe).
               Retorna Nones si la optimización falla.
    """
    if prices is None or prices.empty or prices.shape[1] < 2:
        st.warning("Se necesitan al menos 2 activos con datos válidos para la optimización.")
        return None, None, None, None, None

    results = {}
    try:
        # 0. Limpiar precios: asegurar numérico y quitar NaNs/Infs
        prices_cleaned = prices.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
        # Quitar columnas que sean completamente NaN
        prices_cleaned = prices_cleaned.dropna(axis=1, how='all')
        # Quitar filas donde TODOS los precios sean NaN (aunque ffill/bfill debería evitarlo)
        prices_cleaned = prices_cleaned.dropna(axis=0, how='all')
        # Rellenar NaNs restantes (si un activo empieza más tarde)
        prices_cleaned.ffill(inplace=True)
        prices_cleaned.bfill(inplace=True)
        # Volver a quitar columnas si después de rellenar siguen siendo todo NaN
        prices_cleaned = prices_cleaned.dropna(axis=1, how='all')


        if prices_cleaned.shape[1] < 2:
             st.warning(f"Menos de 2 activos con datos suficientes para optimización después de limpiar. Activos considerados: {prices_cleaned.columns.tolist()}")
             return None, None, None, None, None
        # Comprobar si hay suficientes filas
        if prices_cleaned.shape[0] < 2:
             st.warning("No hay suficientes filas de datos para la optimización después de limpiar.")
             return None, None, None, None, None

        # 1. Calcular Retornos Esperados (media histórica simple)
        mu = expected_returns.mean_historical_return(prices_cleaned, frequency=252)

        # 2. Calcular Matriz de Covarianza (Ledoit-Wolf para más estabilidad)
        S = risk_models.CovarianceShrinkage(prices_cleaned, frequency=252).ledoit_wolf()

        # 3. Optimización - Frontera Eficiente (Restringir a solo largos)
        ef = EfficientFrontier(mu, S, weight_bounds=(0, 1)) # Solo largos

        # 3a. Cartera de Mínima Varianza (MVP)
        try:
            ef_min_vol = copy.deepcopy(ef)
            mvp_weights_raw = ef_min_vol.min_volatility()
            mvp_weights = {k: v for k, v in mvp_weights_raw.items() if abs(v) > 1e-4} # Limpiar
            mvp_performance = ef_min_vol.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            results['mvp_weights'] = mvp_weights
            results['mvp_performance'] = {'expected_return': mvp_performance[0], 'annual_volatility': mvp_performance[1], 'sharpe_ratio': mvp_performance[2]}
        except ValueError as e_mvp: # Capturar ValueError específico de PyPortfolioOpt
             if "optimization may fail" in str(e_mvp):
                  st.warning(f"Optimización de mínima varianza puede haber fallado o ser inestable: {e_mvp}")
             else:
                  st.warning(f"No se pudo calcular la cartera de mínima varianza: {e_mvp}")
             results['mvp_weights'] = None
             results['mvp_performance'] = None
        except Exception as e_mvp_other: # Capturar otros errores
             st.warning(f"Error inesperado en cálculo MVP: {e_mvp_other}")
             results['mvp_weights'] = None
             results['mvp_performance'] = None


        # 3b. Cartera de Máximo Ratio de Sharpe (Tangency)
        try:
            # Comprobar si algún retorno esperado supera la tasa libre de riesgo ANTES de intentar
            if not (mu > risk_free_rate).any():
                 st.warning("Ningún activo supera la tasa libre de riesgo; no se puede calcular Máx Sharpe.")
                 raise ValueError("No hay retornos esperados por encima de la tasa libre de riesgo.") # Forzar salto a except

            ef_max_sharpe = copy.deepcopy(ef)
            # Añadir objetivo L2 regularización puede ayudar a diversificar pesos
            ef_max_sharpe.add_objective(objective_functions.L2_reg, gamma=0.1) # Gamma pequeño
            msr_weights_raw = ef_max_sharpe.max_sharpe(risk_free_rate=risk_free_rate)
            msr_weights = {k: v for k, v in msr_weights_raw.items() if abs(v) > 1e-4} # Limpiar
            msr_performance = ef_max_sharpe.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            results['msr_weights'] = msr_weights
            results['msr_performance'] = {'expected_return': msr_performance[0], 'annual_volatility': msr_performance[1], 'sharpe_ratio': msr_performance[2]}
        except ValueError as e_msr: # Capturar ValueError (incluye el de risk-free rate)
             # Ya se mostró advertencia si es por risk-free rate
             if "optimization may fail" in str(e_msr):
                  st.warning(f"Optimización de máximo Sharpe puede haber fallado o ser inestable: {e_msr}")
             elif "exceeding the risk-free rate" not in str(e_msr): # No mostrar doble warning
                  st.warning(f"No se pudo calcular la cartera de máximo Sharpe: {e_msr}")
             results['msr_weights'] = None
             results['msr_performance'] = None
        except Exception as e_msr_other:
             st.warning(f"Error inesperado en cálculo Max Sharpe: {e_msr_other}")
             results['msr_weights'] = None
             results['msr_performance'] = None


        # 4. Generar puntos de la Frontera Eficiente para plotear
        try:
            ef_frontier = copy.deepcopy(ef) # Usar la versión solo largos
            n_samples = 100
            # Calcular rango de retornos objetivo (desde MVP hasta max retorno individual)
            # Asegurarse que mvp_performance no es None
            min_ret_frontier = results.get('mvp_performance')['expected_return'] if results.get('mvp_performance') else mu.min()
            max_ret_frontier = mu.max() # Max retorno esperado individual como límite superior razonable

            # Asegurar que min < max
            if min_ret_frontier >= max_ret_frontier:
                 st.warning("Rango de retornos para frontera inválido (min >= max). No se puede plotear.")
                 raise ValueError("Rango inválido para frontera")

            target_returns = np.linspace(min_ret_frontier, max_ret_frontier, n_samples)

            frontier_volatility = []
            frontier_sharpe = []

            # Calcular volatilidad mínima para cada nivel de retorno objetivo
            for target_ret in target_returns:
                 try:
                      ef_point = copy.deepcopy(ef_frontier) # Copiar para cada punto
                      # Usar efficient_return para encontrar la cartera con ese retorno
                      ef_point.efficient_return(target_return=target_ret)
                      perf = ef_point.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
                      frontier_volatility.append(perf[1]) # Volatilidad
                      frontier_sharpe.append(perf[2]) # Sharpe
                 except ValueError: # Si el retorno objetivo no es alcanzable
                      frontier_volatility.append(np.nan)
                      frontier_sharpe.append(np.nan)


            frontier_df = pd.DataFrame({
                'Return': target_returns, # Usar retornos objetivo como eje Y
                'Volatility': frontier_volatility,
                'Sharpe': frontier_sharpe
            }).dropna().sort_values(by='Volatility') # Ordenar por volatilidad

            results['frontier_df'] = frontier_df

        except Exception as e_frontier:
            st.warning(f"No se pudieron generar puntos de la frontera eficiente: {e_frontier}")
            results['frontier_df'] = None


        return (results.get('mvp_weights'), results.get('msr_weights'),
                results.get('mvp_performance'), results.get('msr_performance'),
                results.get('frontier_df'))

    except Exception as e:
        st.error(f"Error durante la optimización de cartera: {e}")
        return None, None, None, None, None


# --- Interfaz de Streamlit ---
st.title("💡 Backtester Quant v4.4 (Optimización)")
st.markdown("""
Sube tu archivo CSV con los precios históricos de tus fondos **y tu benchmark** para analizar y optimizar tu cartera.
**Formato esperado:** 1ª col Fecha, siguientes ISINs/Tickers y Benchmark; sep CSV (',' o ';'); decimal (',' o '.').
""")

# --- Barra Lateral ---
st.sidebar.header("Configuración del Backtest")
uploaded_file = st.sidebar.file_uploader("1. Carga tu archivo CSV (con Benchmark)", type=["csv"])

# --- Estado de Sesión ---
default_session_state = {
    'data': None,
    'weights': {},
    'run_results': None,
    'last_uploaded_id': None,
    'benchmark_col': None, # Columna seleccionada como benchmark
    'rolling_window': 60,
    'risk_free_rate': 0.0
}
for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Carga y Selección de Benchmark ---
benchmark_col = None # Variable local
if uploaded_file is not None:
    uploaded_file_id = uploaded_file.name + str(uploaded_file.size)
    if st.session_state.last_uploaded_id != uploaded_file_id:
        st.session_state['data'] = load_data(uploaded_file)
        st.session_state.last_uploaded_id = uploaded_file_id
        st.session_state['run_results'] = None # Resetear resultados
        st.session_state['benchmark_col'] = None # Resetear selección benchmark
        if st.session_state['data'] is not None:
            # Inicializar pesos solo para las columnas del archivo cargado
            # Excluir posible columna benchmark de la inicialización de pesos
            temp_cols = st.session_state['data'].columns.tolist()
            # Intentar adivinar benchmark para inicializar pesos correctamente
            common_benchmarks = ['^GSPC', 'SPY', 'IBEX', '^IBEX', 'benchmark', 'indice', 'Benchmark', 'Index']
            guessed_bm = None
            for col in temp_cols:
                 # Comprobar si el nombre de la columna (o en minúsculas) está en la lista
                 if col in common_benchmarks or col.lower() in [bm.lower() for bm in common_benchmarks]:
                    guessed_bm = col
                    break
            st.session_state['weights'] = {fund: 0.0 for fund in temp_cols if fund != guessed_bm}
        else:
             st.session_state['weights'] = {}

data = st.session_state['data']

if data is not None:
    st.sidebar.success(f"Archivo '{uploaded_file.name}' cargado.")
    st.sidebar.markdown("---")

    # Selección de Benchmark
    available_columns = data.columns.tolist()
    # Intentar preseleccionar benchmark si existe un nombre común
    common_benchmarks = ['^GSPC', 'SPY', 'IBEX', '^IBEX', 'benchmark', 'indice', 'Benchmark', 'Index']
    detected_benchmark_index = 0 # Por defecto la primera columna si no se detecta
    # Si ya hay una selección guardada, usarla
    saved_benchmark = st.session_state.get('benchmark_col')
    if saved_benchmark and saved_benchmark in available_columns:
        detected_benchmark_index = available_columns.index(saved_benchmark)
    else: # Si no hay guardada, intentar detectar
        for i, col in enumerate(available_columns):
            # Comprobar si el nombre de la columna (o en minúsculas) está en la lista
            if col in common_benchmarks or col.lower() in [bm.lower() for bm in common_benchmarks]:
                detected_benchmark_index = i
                break

    # Guardar selección en session state para persistencia
    st.session_state['benchmark_col'] = st.sidebar.selectbox(
        "2. Selecciona la Columna Benchmark",
        options=available_columns,
        index=detected_benchmark_index,
        key='benchmark_selector' # Usar clave para que el estado se guarde
    )
    benchmark_col = st.session_state['benchmark_col'] # Asignar a variable local
    st.sidebar.info(f"Columna '{benchmark_col}' será usada como benchmark.")
    st.sidebar.markdown("---")

    # --- Resto de Controles ---
    min_date = data.index.min().date()
    max_date = data.index.max().date()
    start_date = st.sidebar.date_input("3. Fecha de Inicio", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("4. Fecha de Fin", max_date, min_value=start_date, max_value=max_date)
    st.sidebar.markdown("---")
    initial_investment = st.sidebar.number_input("5. Inversión Inicial (€)", min_value=1, value=10000, step=100)
    st.sidebar.markdown("---")
    rebalance_freq = st.sidebar.selectbox("6. Frecuencia de Rebalanceo", ['Mensual', 'Trimestral', 'Anual', 'No Rebalancear'])
    st.sidebar.markdown("---")
    # Ajustar max_value dinámicamente basado en datos filtrados por fecha si es posible
    try:
        # Calcular longitud después de filtrar por fecha
        filtered_len = len(data.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)])
        # Asegurar que max_window sea al menos 20 y menor que la longitud
        max_window = max(20, filtered_len - 1) if filtered_len > 20 else 20
    except Exception: # Fallback si hay error en filtrado de fechas
        max_window = 252*3
    # Asegurar que el valor por defecto no exceda el máximo calculado
    default_window = min(st.session_state.get('rolling_window', 60), max_window) if max_window >= 20 else 20
    st.session_state['rolling_window'] = st.sidebar.number_input("7. Ventana Análisis Rodante (días)", min_value=20, max_value=max_window, value=default_window, step=10)
    st.session_state['risk_free_rate'] = st.sidebar.number_input("8. Tasa Libre de Riesgo Anual (%)", min_value=-5.0, max_value=20.0, value=st.session_state.get('risk_free_rate', 0.0) * 100 , step=0.1, format="%.2f") / 100.0

    st.sidebar.markdown("---")
    st.sidebar.subheader("9. Asignación de Pesos (%) [Excluye Benchmark]")

    # --- Formulario de Pesos y Ejecución ---
    with st.sidebar.form(key='weights_form'):
        weights_input = {}
        # Excluir columna benchmark de la asignación de pesos
        asset_columns = [col for col in data.columns if col != benchmark_col]
        if not asset_columns:
             st.sidebar.warning("No quedan columnas de activos después de seleccionar el benchmark.")

        for fund in asset_columns:
            # Cargar peso guardado si existe y corresponde a un activo actual, si no, 0.0
            default_weight = st.session_state['weights'].get(fund, 0.0) * 100 if fund in st.session_state['weights'] else 0.0
            weights_input[fund] = st.number_input(f"{fund}", min_value=0.0, max_value=100.0, value=default_weight, step=1.0, format="%.2f") / 100.0

        submitted = st.form_submit_button("🚀 Ejecutar Análisis Completo")

        if submitted:
            # Guardar pesos introducidos (solo para activos)
            st.session_state['weights'] = weights_input

            # Validar y normalizar pesos (solo de activos)
            total_input_weight_sum = sum(weights_input.values())
            weights_norm = {}
            if not np.isclose(total_input_weight_sum, 1.0, atol=0.01):
                 st.sidebar.warning(f"Los pesos de los activos suman {total_input_weight_sum*100:.2f}%. Deben sumar 100%. Se normalizarán.")
                 if total_input_weight_sum > 1e-6: # Evitar división por cero si suma es muy cercana a 0
                      weights_norm = {k: v / total_input_weight_sum for k, v in weights_input.items()}
                 else:
                      st.sidebar.error("La suma de los pesos de los activos es 0%. Asigna pesos.")
            else:
                 weights_norm = weights_input # Usar pesos tal cual

            active_weights = {k: v for k, v in weights_norm.items() if v > 1e-6} # Considerar pesos muy pequeños como 0

            if not active_weights:
                st.error("No hay fondos con peso > 0 asignado.")
                st.session_state['run_results'] = None
            else:
                funds_in_portfolio = list(active_weights.keys())
                # Filtrar datos para el rango y columnas necesarias (activos + benchmark)
                cols_needed = funds_in_portfolio + ([benchmark_col] if benchmark_col else [])
                # Asegurar que las columnas existen antes de filtrar
                cols_to_filter = [col for col in cols_needed if col in data.columns]
                data_filtered = data.loc[pd.to_datetime(start_date):pd.to_datetime(end_date), cols_to_filter].copy()

                # Limpiar NaNs DESPUÉS de filtrar columnas y fechas
                data_filtered.ffill(inplace=True)
                data_filtered.bfill(inplace=True)
                # Eliminar filas donde falte algún dato ESENCIAL (ej, precio activo o benchmark si se usa)
                essential_cols = funds_in_portfolio + ([benchmark_col] if benchmark_col in data_filtered.columns else [])
                data_filtered.dropna(subset=essential_cols, how='any', inplace=True)


                if data_filtered.empty or data_filtered.shape[0] < 2:
                     st.error("No hay suficientes datos comunes para activos (y benchmark si aplica) en el rango seleccionado después de limpiar NaNs.")
                     st.session_state['run_results'] = None
                elif benchmark_col and benchmark_col not in data_filtered.columns:
                     st.error(f"La columna benchmark '{benchmark_col}' no se encontró después de filtrar. Verifica el nombre.")
                     st.session_state['run_results'] = None
                else:
                    st.spinner("Realizando cálculos avanzados y optimización...")
                    # Separar datos de activos y benchmark (ya limpios y alineados por fecha)
                    asset_data_final = data_filtered[funds_in_portfolio]
                    benchmark_data_final = data_filtered[benchmark_col] if benchmark_col in data_filtered.columns else None

                    # --- Ejecución Análisis ---
                    # 1. Backtest Básico (solo con datos de activos)
                    portfolio_value, portfolio_returns = run_backtest(
                        asset_data_final,
                        active_weights,
                        initial_investment,
                        asset_data_final.index.min(),
                        asset_data_final.index.max(),
                        rebalance_freq
                    )

                    # Inicializar diccionario de resultados
                    results_dict = {'run_successful': False}

                    if portfolio_value is not None and not portfolio_value.empty:
                        # 2. Métricas Cartera y Fondos
                        portfolio_metrics = calculate_metrics(portfolio_value, portfolio_returns)
                        portfolio_metrics['Sortino Ratio'] = calculate_sortino_ratio(portfolio_returns, required_return=st.session_state['risk_free_rate'])

                        individual_metrics = calculate_individual_metrics(asset_data_final)
                        asset_returns = calculate_returns(asset_data_final) # Retornos de activos para análisis posteriores

                        # 3. Benchmark y Métricas Relativas (CONDICIONAL)
                        benchmark_returns = None
                        benchmark_metrics = {}
                        individual_asset_benchmark_metrics = pd.DataFrame(index=funds_in_portfolio, columns=['Beta', 'Alpha (anual)'])
                        rolling_beta_portfolio = None

                        if benchmark_data_final is not None and asset_returns is not None: # Necesitamos retornos de activos también
                            benchmark_returns = calculate_returns(benchmark_data_final)
                            if benchmark_returns is not None and not benchmark_returns.empty:
                                 common_idx_returns = portfolio_returns.index.intersection(benchmark_returns.index)
                                 # Asegurar suficientes datos para rolling beta
                                 if len(common_idx_returns) >= st.session_state['rolling_window']:
                                     portfolio_returns_aligned = portfolio_returns.loc[common_idx_returns]
                                     benchmark_returns_aligned = benchmark_returns.loc[common_idx_returns]

                                     benchmark_metrics = calculate_benchmark_metrics(portfolio_returns_aligned, benchmark_returns_aligned, st.session_state['risk_free_rate'])
                                     portfolio_metrics.update(benchmark_metrics)

                                     temp_asset_bm_metrics = {}
                                     for fund in funds_in_portfolio:
                                         if fund in asset_returns.columns:
                                             fund_ret_aligned = asset_returns[fund].loc[common_idx_returns]
                                             if len(fund_ret_aligned) >= 2:
                                                 metrics = calculate_benchmark_metrics(fund_ret_aligned, benchmark_returns_aligned, st.session_state['risk_free_rate'])
                                                 temp_asset_bm_metrics[fund] = {'Beta': metrics.get('Beta'), 'Alpha (anual)': metrics.get('Alpha (anual)')}
                                             else: temp_asset_bm_metrics[fund] = {'Beta': np.nan, 'Alpha (anual)': np.nan}
                                         else: temp_asset_bm_metrics[fund] = {'Beta': np.nan, 'Alpha (anual)': np.nan}
                                     individual_asset_benchmark_metrics.update(pd.DataFrame(temp_asset_bm_metrics).T)

                                     rolling_beta_portfolio = calculate_rolling_beta(portfolio_returns_aligned, benchmark_returns_aligned, st.session_state['rolling_window'])
                                 else: st.warning("Datos comunes insuficientes entre cartera y benchmark para métricas relativas/rodantes.")
                            else: benchmark_returns = None

                        # 4. Análisis Normalizado para Gráfico
                        normalized_individual_prices = pd.DataFrame()
                        # Usar asset_data_final alineado con el índice de portfolio_value
                        asset_data_norm_source = asset_data_final.loc[portfolio_value.index]
                        if not asset_data_norm_source.empty and (asset_data_norm_source.iloc[0] > 1e-9).all(): # Evitar división por cero
                             normalized_individual_prices = asset_data_norm_source / asset_data_norm_source.iloc[0] * 100
                        else: st.warning("No se pudieron normalizar precios individuales.")

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
                                 # else: st.warning(f"No se pudo normalizar benchmark '{benchmark_col}' (valor inicial inválido).") # Opcional

                        plot_data_normalized = pd.concat([normalized_portfolio_value, normalized_individual_prices], axis=1)
                        if normalized_benchmark is not None:
                             plot_data_normalized = pd.concat([plot_data_normalized, normalized_benchmark], axis=1)

                        # 5. Análisis Correlación
                        corr_matrix = calculate_correlation_matrix(asset_returns)
                        avg_rolling_corr, pair_rolling_corr = calculate_rolling_correlation(asset_returns, window=st.session_state['rolling_window'])

                        # 6. Análisis Riesgo
                        risk_contribution = calculate_risk_contribution(asset_returns, active_weights)
                        diversification_ratio = calculate_diversification_ratio(asset_returns, active_weights)
                        portfolio_metrics['Diversification Ratio'] = diversification_ratio

                        # 7. Análisis Rodante Cartera
                        rolling_vol, rolling_sharpe, rolling_sortino = calculate_rolling_metrics(
                            portfolio_returns, st.session_state['rolling_window'], st.session_state['risk_free_rate'])

                        # 8. Optimización de Cartera
                        mvp_weights, msr_weights, mvp_performance, msr_performance, frontier_df = optimize_portfolio(
                            asset_data_final, risk_free_rate=st.session_state['risk_free_rate'])

                        # Calcular rendimiento/riesgo cartera actual para plot optimización
                        current_portfolio_performance_opt = None
                        if active_weights and not asset_data_final.empty:
                            try:
                                # Usar los mismos inputs que optimize_portfolio
                                prices_cleaned_opt = asset_data_final.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='any')
                                if not prices_cleaned_opt.empty and prices_cleaned_opt.shape[1] >=1 : # Necesita al menos 1 activo
                                     mu_current = expected_returns.mean_historical_return(prices_cleaned_opt, frequency=252)
                                     S_current = risk_models.CovarianceShrinkage(prices_cleaned_opt, frequency=252).ledoit_wolf()

                                     # Asegurar que los pesos coinciden con las columnas limpiadas
                                     weights_array = np.array([active_weights.get(asset, 0) for asset in prices_cleaned_opt.columns])
                                     # Normalizar pesos por si se quitaron columnas en prices_cleaned_opt
                                     weights_sum = weights_array.sum()
                                     weights_array_norm = weights_array / weights_sum if weights_sum > 1e-9 else weights_array


                                     current_ret = pypfopt.objective_functions.portfolio_return(weights_array_norm, mu_current, negative=False)
                                     # CORREGIDO: Calcular volatilidad manualmente
                                     current_variance = weights_array_norm.T @ S_current @ weights_array_norm
                                     current_vol = np.sqrt(current_variance) if current_variance >= 0 else np.nan

                                     current_sharpe = (current_ret - st.session_state['risk_free_rate']) / current_vol if pd.notna(current_vol) and current_vol > 1e-9 else np.nan
                                     current_portfolio_performance_opt = {
                                        'expected_return': current_ret, 'annual_volatility': current_vol, 'sharpe_ratio': current_sharpe
                                     }
                            except Exception as e_curr:
                                st.warning(f"No se pudo calcular rendimiento/riesgo de cartera actual para optimización: {e_curr}")


                        # Guardar TODOS los resultados
                        results_dict = {
                            'portfolio_value': portfolio_value,
                            'portfolio_metrics': portfolio_metrics,
                            'plot_data_normalized': plot_data_normalized,
                            'individual_metrics': individual_metrics,
                            'individual_asset_benchmark_metrics': individual_asset_benchmark_metrics,
                            'asset_returns': asset_returns,
                            'benchmark_returns': benchmark_returns,
                            'corr_matrix': corr_matrix,
                            'avg_rolling_corr': avg_rolling_corr,
                            'pair_rolling_corr': pair_rolling_corr,
                            'risk_contribution': risk_contribution,
                            'rolling_vol': rolling_vol,
                            'rolling_sharpe': rolling_sharpe,
                            'rolling_sortino': rolling_sortino,
                            'rolling_beta_portfolio': rolling_beta_portfolio,
                            'weights': active_weights,
                            'rolling_window_used': st.session_state['rolling_window'],
                            'benchmark_col_used': benchmark_col if benchmark_data_final is not None and benchmark_returns is not None else None,
                            'mvp_weights': mvp_weights,
                            'msr_weights': msr_weights,
                            'mvp_performance': mvp_performance,
                            'msr_performance': msr_performance,
                            'frontier_df': frontier_df,
                            'current_portfolio_performance_opt': current_portfolio_performance_opt,
                            'run_successful': True # Marcar como exitoso
                        }
                        st.session_state['run_results'] = results_dict
                        st.sidebar.success("Análisis y optimización completados.")

                    else: # Si portfolio_value es None o vacío
                        st.error("La simulación del backtest no produjo resultados válidos.")
                        st.session_state['run_results'] = {'run_successful': False}


# --- Mostrar Resultados (Pestañas) ---
# Comprobar si hubo una ejecución y si fue exitosa
if 'run_results' in st.session_state and st.session_state['run_results'] is not None and st.session_state['run_results'].get('run_successful', False):
    # --- Extracción Segura de Resultados ---
    results = st.session_state['run_results']
    benchmark_col_used = results.get('benchmark_col_used')
    portfolio_value = results.get('portfolio_value')
    portfolio_metrics = results.get('portfolio_metrics', {})
    plot_data_normalized = results.get('plot_data_normalized')
    individual_metrics = results.get('individual_metrics', pd.DataFrame())
    individual_asset_benchmark_metrics = results.get('individual_asset_benchmark_metrics', pd.DataFrame())
    asset_returns = results.get('asset_returns')
    benchmark_returns = results.get('benchmark_returns')
    corr_matrix = results.get('corr_matrix')
    avg_rolling_corr = results.get('avg_rolling_corr')
    pair_rolling_corr = results.get('pair_rolling_corr')
    risk_contribution = results.get('risk_contribution', pd.Series(dtype=float))
    rolling_vol = results.get('rolling_vol')
    rolling_sharpe = results.get('rolling_sharpe')
    rolling_sortino = results.get('rolling_sortino')
    rolling_beta_portfolio = results.get('rolling_beta_portfolio')
    used_weights = results.get('weights', {})
    rolling_window_used = results.get('rolling_window_used', 0)
    # Resultados de optimización
    mvp_weights = results.get('mvp_weights')
    msr_weights = results.get('msr_weights')
    mvp_performance = results.get('mvp_performance')
    msr_performance = results.get('msr_performance')
    frontier_df = results.get('frontier_df')
    current_portfolio_performance_opt = results.get('current_portfolio_performance_opt')


    # --- Definición de Pestañas ---
    tabs_titles = ["📊 Visión General", "🔗 Correlación", "🧩 Activos y Riesgo", "🔬 Optimización"] # Añadir Optimización
    # Insertar pestaña benchmark solo si se usó y hay resultados válidos
    if benchmark_col_used and benchmark_returns is not None:
        tabs_titles.insert(1, f"🆚 vs {benchmark_col_used}")
        tabs = st.tabs(tabs_titles)
        tab1, tab_bm, tab_corr, tab_risk, tab_opt = tabs[0], tabs[1], tabs[2], tabs[3], tabs[4]
    else:
        # Si no hay benchmark válido, no mostrar esa pestaña
        tabs = st.tabs(tabs_titles)
        tab1, tab_corr, tab_risk, tab_opt = tabs[0], tabs[1], tabs[2], tabs[3]
        tab_bm = None # Marcar que no existe

    # --- Pestaña 1: Visión General ---
    with tab1:
        st.header("Visión General de la Cartera")
        st.subheader("Métricas Principales (Cartera Total)")
        # Formatear métricas para mostrar
        col1, col2, col3 = st.columns(3)
        try: locale.setlocale(locale.LC_ALL, 'es_ES.UTF-8')
        except locale.Error: locale.setlocale(locale.LC_ALL, '') # Fallback

        col1.metric("Rentabilidad Total", f"{portfolio_metrics.get('Rentabilidad Total', np.nan):.2%}")
        col2.metric("Rentab. Anual (CAGR)", f"{portfolio_metrics.get('Rentabilidad Anualizada (CAGR)', np.nan):.2%}")
        col3.metric("Volatilidad Anualizada", f"{portfolio_metrics.get('Volatilidad Anualizada', np.nan):.2%}")

        col1b, col2b, col3b = st.columns(3)
        col1b.metric("Máximo Drawdown", f"{portfolio_metrics.get('Máximo Drawdown', np.nan):.2%}")
        col2b.metric("Ratio de Sharpe", f"{portfolio_metrics.get('Ratio de Sharpe', np.nan):.2f}") # Nombre genérico
        col3b.metric("Ratio Sortino", f"{portfolio_metrics.get('Sortino Ratio', np.nan):.2f}")

        col1c, col2c, col3c = st.columns(3) # Fila adicional si es necesario
        col1c.metric("Ratio Diversificación", f"{portfolio_metrics.get('Diversification Ratio', np.nan):.2f}", help="Volatilidad media ponderada / Volatilidad Cartera (más alto = mejor diversificación)")

        st.markdown("---")
        st.subheader("Evolución Normalizada (Base 100)")
        if plot_data_normalized is not None and not plot_data_normalized.empty:
            # Eliminar columnas que sean completamente NaN antes de plotear
            plot_data_to_show = plot_data_normalized.dropna(axis=1, how='all')
            if not plot_data_to_show.empty:
                fig_perf = px.line(plot_data_to_show, x=plot_data_to_show.index, y=plot_data_to_show.columns,
                                   title="Cartera Total vs. Activos Individuales" + (f" vs {benchmark_col_used}" if benchmark_col_used else ""),
                                   labels={'value': 'Valor Normalizado (Base 100)', 'variable': 'Activo'})
                fig_perf.update_layout(xaxis_title="Fecha", yaxis_title="Valor Normalizado", legend_title_text='Activos')
                st.plotly_chart(fig_perf, use_container_width=True)
            else:
                 st.warning("No hay datos válidos para mostrar en el gráfico de evolución (posiblemente por NaNs).")
        else:
            st.warning("No hay datos para el gráfico de evolución.")
        st.markdown("---")

        st.subheader(f"Análisis Rodante de la Cartera (Ventana: {rolling_window_used} días)")
        col_roll1, col_roll2, col_roll3 = st.columns(3)
        with col_roll1:
            if rolling_vol is not None and not rolling_vol.empty:
                fig_roll_vol = px.line(rolling_vol, x=rolling_vol.index, y=rolling_vol.name, title="Volatilidad Anualizada Rodante", labels={'value': 'Volatilidad Anual.'})
                fig_roll_vol.update_layout(showlegend=False, yaxis_title="Volatilidad Anualizada")
                st.plotly_chart(fig_roll_vol, use_container_width=True)
            else: st.warning("Datos insuficientes para volatilidad rodante.")
        with col_roll2:
            if rolling_sharpe is not None and not rolling_sharpe.empty:
                fig_roll_sharpe = px.line(rolling_sharpe, x=rolling_sharpe.index, y=rolling_sharpe.name, title="Ratio de Sharpe Rodante", labels={'value': 'Sharpe Ratio'})
                fig_roll_sharpe.update_layout(showlegend=False, yaxis_title="Ratio de Sharpe")
                st.plotly_chart(fig_roll_sharpe, use_container_width=True)
            else: st.warning("Datos insuficientes para Sharpe rodante.")
        with col_roll3:
            if rolling_sortino is not None and not rolling_sortino.empty:
                 fig_roll_sortino = px.line(rolling_sortino, x=rolling_sortino.index, y=rolling_sortino.name, title="Ratio de Sortino Rodante", labels={'value': 'Sortino Ratio'})
                 fig_roll_sortino.update_layout(showlegend=False, yaxis_title="Ratio de Sortino")
                 st.plotly_chart(fig_roll_sortino, use_container_width=True)
            else: st.warning("Datos insuficientes para Sortino rodante.")

    # --- Pestaña Benchmark (Condicional) ---
    if tab_bm: # Solo ejecutar si la pestaña existe
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
                  fig_roll_beta = px.line(rolling_beta_portfolio, x=rolling_beta_portfolio.index, y=rolling_beta_portfolio.name, title="Beta Rodante de la Cartera", labels={'value': 'Beta'})
                  fig_roll_beta.update_layout(showlegend=False, yaxis_title="Beta")
                  st.plotly_chart(fig_roll_beta, use_container_width=True)
            else: st.warning("Datos insuficientes para Beta rodante.")

    # --- Pestaña Correlación ---
    with tab_corr:
        st.header("Análisis de Correlación entre Activos")
        st.subheader("Matriz de Correlación (Período Completo)")
        # Asegurarse que corr_matrix no es None y no está vacío
        if corr_matrix is not None and not corr_matrix.empty:
            # Evitar plotear si la matriz es 1x1 (un solo activo)
            if corr_matrix.shape[0] > 1 and corr_matrix.shape[1] > 1:
                fig_heatmap, ax = plt.subplots(figsize=(max(6, corr_matrix.shape[1]*0.8), max(5, corr_matrix.shape[0]*0.7)))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax, vmin=-1, vmax=1, annot_kws={"size": 8})
                ax.set_title('Correlación entre Activos')
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.yticks(rotation=0, fontsize=8)
                plt.tight_layout()
                st.pyplot(fig_heatmap)
                st.caption("Valores cercanos a +1: alta correlación positiva. Cercanos a -1: alta correlación negativa. Cercanos a 0: baja correlación lineal.")
            elif corr_matrix.shape[0] == 1:
                 st.info("Solo hay un activo en la cartera, no se puede calcular la matriz de correlación.")
            else: # Caso vacío ya cubierto arriba, pero por si acaso
                 st.warning("No se pudo calcular la matriz de correlación (datos insuficientes).")
        else: st.warning("No se pudo calcular la matriz de correlación.")
        st.markdown("---")

        st.subheader(f"Correlación Rodante (Ventana: {rolling_window_used} días)")
        if avg_rolling_corr is not None and not avg_rolling_corr.empty:
            fig_roll_corr_avg = px.line(avg_rolling_corr, x=avg_rolling_corr.index, y=avg_rolling_corr.name, title="Correlación Promedio Rodante entre Activos", labels={'value': 'Correlación Promedio'})
            fig_roll_corr_avg.update_layout(showlegend=False, yaxis_title="Correlación Promedio")
            st.plotly_chart(fig_roll_corr_avg, use_container_width=True)

            # Selección de Pares
            if asset_returns is not None and asset_returns.shape[1] >= 2:
                asset_list = asset_returns.columns.tolist()
                pair_options = [(asset_list[i], asset_list[j]) for i in range(len(asset_list)) for j in range(i + 1, len(asset_list))]
                if pair_options:
                    selected_pairs = st.multiselect(
                        "Selecciona pares para ver su correlación rodante específica:",
                        options=pair_options,
                        format_func=lambda pair: f"{pair[0]} vs {pair[1]}"
                        )
                    if selected_pairs:
                        # Recalcular solo para los pares seleccionados
                        _ , specific_pair_rolling_corr = calculate_rolling_correlation(asset_returns, window=rolling_window_used, pair_list=selected_pairs)
                        if specific_pair_rolling_corr is not None and not specific_pair_rolling_corr.empty:
                            fig_roll_corr_pairs = px.line(specific_pair_rolling_corr, x=specific_pair_rolling_corr.index, y=specific_pair_rolling_corr.columns, title="Correlación Rodante para Pares Seleccionados", labels={'value': 'Correlación', 'variable': 'Par de Activos'})
                            fig_roll_corr_pairs.update_layout(yaxis_title="Correlación", legend_title_text='Pares')
                            st.plotly_chart(fig_roll_corr_pairs, use_container_width=True)
                        else: st.warning("No se pudieron calcular las correlaciones para los pares seleccionados.")
        else: st.warning("Datos insuficientes para correlación rodante.")

    # --- Pestaña Activos y Riesgo ---
    with tab_risk:
        st.header("Análisis de Activos Individuales y Riesgo")
        st.subheader("Posicionamiento Riesgo/Retorno (Activos Individuales y Cartera)")
        # Comprobar que individual_metrics y portfolio_metrics existen y no son None
        if individual_metrics is not None and not individual_metrics.empty and portfolio_metrics:
             scatter_data = individual_metrics.T.copy()
             # CORREGIDO: Usar .get() para acceder a columnas y evitar KeyError
             # Usar nombres de métricas consistentes
             scatter_data['Rentab. Anual (CAGR)'] = pd.to_numeric(scatter_data.get('Rentabilidad Anualizada (CAGR)'), errors='coerce')
             scatter_data['Volatilidad Anualizada'] = pd.to_numeric(scatter_data.get('Volatilidad Anualizada'), errors='coerce')
             # CORREGIDO: Comprobar si las columnas existen antes de dropear NaNs
             cols_to_check = ['Rentab. Anual (CAGR)', 'Volatilidad Anualizada']
             valid_cols = [col for col in cols_to_check if col in scatter_data.columns]
             if len(valid_cols) == len(cols_to_check): # Solo dropear si ambas columnas existen
                 scatter_data.dropna(subset=valid_cols, inplace=True)
             else:
                 st.warning("Faltan columnas CAGR o Volatilidad en métricas individuales para gráfico Riesgo/Retorno.")
                 scatter_data = pd.DataFrame() # Vaciar si faltan columnas clave

             scatter_data['Tipo'] = 'Activo Individual'

             # Añadir datos de la cartera total
             portfolio_scatter = pd.DataFrame({
                  'Rentab. Anual (CAGR)': [portfolio_metrics.get('Rentabilidad Anualizada (CAGR)')],
                  'Volatilidad Anualizada': [portfolio_metrics.get('Volatilidad Anualizada')]
             }, index=['Cartera Total'])
             portfolio_scatter['Rentab. Anual (CAGR)'] = pd.to_numeric(portfolio_scatter['Rentab. Anual (CAGR)'], errors='coerce')
             portfolio_scatter['Volatilidad Anualizada'] = pd.to_numeric(portfolio_scatter['Volatilidad Anualizada'], errors='coerce')
             portfolio_scatter.dropna(inplace=True)
             portfolio_scatter['Tipo'] = 'Cartera Total'

             # CORREGIDO: Comprobar que scatter_data_final no esté vacío antes de plotear
             if not scatter_data.empty or not portfolio_scatter.empty:
                 scatter_data_final = pd.concat([scatter_data, portfolio_scatter])
                 # Asegurarse que las columnas necesarias existen después de concatenar
                 if not scatter_data_final.empty and all(col in scatter_data_final.columns for col in ['Volatilidad Anualizada', 'Rentab. Anual (CAGR)']):
                     fig_scatter = px.scatter(scatter_data_final,
                                               x='Volatilidad Anualizada', y='Rentab. Anual (CAGR)',
                                               text=scatter_data_final.index, color='Tipo', hover_name=scatter_data_final.index,
                                               title="Rentabilidad Anualizada vs. Volatilidad Anualizada")
                     fig_scatter.update_traces(textposition='top center')
                     # CORREGIDO: Usar formato correcto para ejes de porcentaje
                     fig_scatter.update_layout(xaxis_title="Volatilidad Anualizada", yaxis_title="Rentabilidad Anualizada (CAGR)",
                                              xaxis_tickformat=".1%", yaxis_tickformat=".1%")
                     st.plotly_chart(fig_scatter, use_container_width=True)
                 else:
                      st.warning("No hay datos válidos para generar el gráfico Riesgo/Retorno después de procesar.")
             else: st.warning("No hay datos válidos para generar el gráfico Riesgo/Retorno.")
        else: st.warning("Faltan métricas para el gráfico Riesgo/Retorno.")
        st.markdown("---")

        st.subheader("Contribución de Cada Activo a la Volatilidad de la Cartera")
        if risk_contribution is not None and not risk_contribution.empty:
            risk_contribution_pct = risk_contribution * 100
            risk_contribution_pct = risk_contribution_pct.reset_index()
            risk_contribution_pct.columns = ['Activo', 'Contribución al Riesgo (%)']
            fig_risk = px.bar(risk_contribution_pct.sort_values(by='Contribución al Riesgo (%)', ascending=False),
                              x='Activo', y='Contribución al Riesgo (%)',
                              title='Contribución Porcentual al Riesgo Total (Volatilidad)',
                              text='Contribución al Riesgo (%)')
            fig_risk.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig_risk.update_layout(yaxis_title="Contribución (%)", xaxis_title="Activo")
            st.plotly_chart(fig_risk, use_container_width=True)
            st.caption("Muestra qué porcentaje de la volatilidad total es atribuible a cada activo, considerando peso, volatilidad individual y correlación.")
        else: st.warning("No se pudo calcular la contribución al riesgo.")
        st.markdown("---")

        st.subheader("Ranking Avanzado de Activos en la Cartera")
        if individual_metrics is not None and not individual_metrics.empty:
            ranking_df = individual_metrics.T.copy() # Activos como filas
            if risk_contribution is not None and not risk_contribution.empty:
                 # Asegurar que el índice coincida para la asignación
                 risk_contribution.index.name = ranking_df.index.name # Usar el mismo nombre de índice
                 ranking_df = ranking_df.join(risk_contribution.rename('Contribución Riesgo (%)') * 100, how='left')

            # Añadir Beta/Alpha individual si existen y no están vacíos
            if benchmark_col_used and individual_asset_benchmark_metrics is not None and not individual_asset_benchmark_metrics.empty:
                 # Asegurarse que el índice coincide antes de unir
                 ranking_df = ranking_df.join(individual_asset_benchmark_metrics[['Beta', 'Alpha (anual)']], how='left')

            # Añadir Sortino Ratio individual (ya debería estar)
            if 'Sortino Ratio' not in ranking_df.columns:
                 # Si no está, calcularlo aquí podría ser una opción, pero es mejor asegurar que viene de individual_metrics
                 pass

            # Ordenar (ejemplo: por Sharpe)
            if 'Ratio de Sharpe' in ranking_df.columns:
                 ranking_df['Ratio de Sharpe'] = pd.to_numeric(ranking_df['Ratio de Sharpe'], errors='coerce')
                 ranking_df.sort_values(by='Ratio de Sharpe', ascending=False, inplace=True, na_position='last')

            # Formatear columnas para display
            ranking_display = ranking_df.copy()
            format_percent = lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
            format_float = lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
            format_percent_risk = lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"

            # Ajustar nombres de columnas para formato
            cols_percent = ['Rentabilidad Total', 'Rentabilidad Anualizada (CAGR)', 'Volatilidad Anualizada', 'Máximo Drawdown', 'Alpha (anual)']
            cols_float = ['Ratio de Sharpe', 'Sortino Ratio', 'Beta']
            cols_percent_risk = ['Contribución Riesgo (%)']

            for col in cols_percent:
                 if col in ranking_display.columns: ranking_display[col] = ranking_display[col].map(format_percent)
            for col in cols_float:
                 if col in ranking_display.columns: ranking_display[col] = ranking_display[col].map(format_float)
            for col in cols_percent_risk:
                 if col in ranking_display.columns: ranking_display[col] = ranking_display[col].map(format_percent_risk)

            st.dataframe(ranking_display)
            st.caption("Compara activos por métricas clave y su contribución al riesgo/retorno relativo.")
        else: st.warning("Faltan datos para el ranking avanzado.")

    # --- NUEVA Pestaña Optimización ---
    with tab_opt:
        st.header("🔬 Optimización de Cartera (Frontera Eficiente)")
        st.markdown("""
        Esta sección calcula la Frontera Eficiente basada en los **retornos históricos y volatilidades/correlaciones** de los activos seleccionados.
        Muestra las carteras que ofrecen el mejor perfil riesgo/retorno *según los datos pasados*.
        **Nota:** La optimización se basa en datos históricos y no garantiza resultados futuros. Usa métodos simples de estimación (media histórica, covarianza muestral). Se asumen posiciones largas únicamente (sin cortos).
        """)

        # Comprobar si hay datos para la frontera y las carteras óptimas
        if frontier_df is not None and not frontier_df.empty:
            # Crear gráfico de la frontera eficiente
            fig_frontier = go.Figure()

            # Añadir puntos de la frontera
            fig_frontier.add_trace(go.Scatter(
                x=frontier_df['Volatility'],
                y=frontier_df['Return'],
                mode='lines',
                name='Frontera Eficiente (Solo Largos)',
                line=dict(color='blue', width=2)
            ))

            # Marcar Cartera de Mínima Varianza (MVP)
            if mvp_performance:
                fig_frontier.add_trace(go.Scatter(
                    x=[mvp_performance['annual_volatility']],
                    y=[mvp_performance['expected_return']],
                    mode='markers+text', # Añadir texto
                    marker=dict(color='green', size=12, symbol='star'),
                    text="MVP", textposition="bottom center",
                    name=f"Min Varianza (Sharpe: {mvp_performance['sharpe_ratio']:.2f})"
                ))

            # Marcar Cartera de Máximo Sharpe (MSR)
            if msr_performance:
                fig_frontier.add_trace(go.Scatter(
                    x=[msr_performance['annual_volatility']],
                    y=[msr_performance['expected_return']],
                    mode='markers+text', # Añadir texto
                    marker=dict(color='red', size=12, symbol='star'),
                    text="MSR", textposition="bottom center",
                    name=f"Max Sharpe (Sharpe: {msr_performance['sharpe_ratio']:.2f})"
                ))

            # Marcar Cartera Actual del Usuario
            if current_portfolio_performance_opt:
                 fig_frontier.add_trace(go.Scatter(
                    x=[current_portfolio_performance_opt['annual_volatility']],
                    y=[current_portfolio_performance_opt['expected_return']],
                    mode='markers+text', # Añadir texto
                    marker=dict(color='orange', size=12, symbol='circle'),
                    text="Actual", textposition="bottom center",
                    name=f"Tu Cartera Actual (Sharpe: {current_portfolio_performance_opt['sharpe_ratio']:.2f})"
                ))

            fig_frontier.update_layout(
                title='Frontera Eficiente y Carteras Óptimas',
                xaxis_title='Volatilidad Anualizada (Riesgo)',
                yaxis_title='Rentabilidad Esperada Anualizada',
                xaxis_tickformat=".1%",
                yaxis_tickformat=".1%",
                legend_title_text='Carteras',
                height=500 # Ajustar altura
            )
            st.plotly_chart(fig_frontier, use_container_width=True)

            st.markdown("---")
            st.subheader("Pesos Sugeridos por la Optimización")

            col_opt1, col_opt2 = st.columns(2)

            with col_opt1:
                st.markdown("**Cartera Mínima Varianza (MVP)**")
                if mvp_weights:
                    mvp_weights_df = pd.DataFrame.from_dict(mvp_weights, orient='index', columns=['Peso Sugerido'])
                    mvp_weights_df.index.name = 'Activo'
                    mvp_weights_df['Peso Sugerido'] = mvp_weights_df['Peso Sugerido'].map('{:.2%}'.format)
                    st.dataframe(mvp_weights_df.sort_values(by='Peso Sugerido', ascending=False)) # Ordenar
                    if mvp_performance: # Comprobar si existe antes de mostrar
                         st.caption(f"Rend. Esperado: {mvp_performance['expected_return']:.2%}, Volatilidad: {mvp_performance['annual_volatility']:.2%}")
                else:
                    st.warning("No se pudo calcular.")

            with col_opt2:
                st.markdown("**Cartera Máximo Sharpe (MSR)**")
                if msr_weights:
                    msr_weights_df = pd.DataFrame.from_dict(msr_weights, orient='index', columns=['Peso Sugerido'])
                    msr_weights_df.index.name = 'Activo'
                    msr_weights_df['Peso Sugerido'] = msr_weights_df['Peso Sugerido'].map('{:.2%}'.format)
                    st.dataframe(msr_weights_df.sort_values(by='Peso Sugerido', ascending=False)) # Ordenar
                    if msr_performance: # Comprobar si existe antes de mostrar
                         st.caption(f"Rend. Esperado: {msr_performance['expected_return']:.2%}, Volatilidad: {msr_performance['annual_volatility']:.2%}")

                else:
                    st.warning("No se pudo calcular (posiblemente ningún activo supera la tasa libre de riesgo).")

        else:
            st.warning("No se pudo calcular la frontera eficiente. Revisa los datos de entrada y el rango de fechas (se necesitan al menos 2 activos con datos suficientes).")


# --- Mensajes Finales / Info ---
# Mostrar mensaje inicial si no hay resultados todavía
elif 'run_results' not in st.session_state or st.session_state['run_results'] is None:
     if data is not None:
          st.info("Configura los parámetros en la barra lateral y haz clic en 'Ejecutar Análisis Completo'.")
     elif uploaded_file is not None:
          # Error ya mostrado durante la carga si falló
          pass
     else:
          st.info("Por favor, carga un archivo CSV usando la barra lateral para comenzar.")
# Mostrar mensaje si la ejecución falló
elif not st.session_state['run_results'].get('run_successful', False):
     st.error("La ejecución del análisis falló. Revisa los mensajes de error anteriores o los parámetros de entrada.")


st.sidebar.markdown("---")
st.sidebar.markdown("Backtester Quant v4.4")
st.sidebar.markdown("Dios Familia y Cojones")