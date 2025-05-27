# src/utils.py
import pandas as pd
import numpy as np
from io import StringIO
import empyrical # Para métricas avanzadas

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
                        import streamlit as st # Import local to avoid circular dependency if not needed elsewhere
                        st.error(f"Error convirtiendo columna '{col}' a numérico: {e_conv}. Verifica el formato.")
                        return None
            return data
        except Exception as e:
            import streamlit as st # Import local
            st.error(f"Error crítico al procesar el archivo CSV: {e}")
            st.error("Verifica el formato: 1ª col Fecha, siguientes ISINs/Benchmark; sep CSV (',' o ';'); decimal (',' o '.').")
            return None
    return None

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
    if isinstance(returns, np.ndarray):
        if returns.size == 0:
            returns = pd.Series(dtype=float)
        else:
            returns = pd.Series(returns)

    if returns is None or returns.empty or returns.isnull().all():
        return np.nan

    returns_cleaned = pd.to_numeric(returns, errors='coerce').dropna()
    if returns_cleaned.empty or not np.all(np.isfinite(returns_cleaned)):
        return np.nan
    if len(returns_cleaned) < 2:
        return np.nan

    try:
        sortino = empyrical.sortino_ratio(returns_cleaned,
                                          required_return=daily_required_return_for_empyrical,
                                          annualization=252)
        return sortino if np.isfinite(sortino) else np.nan
    except Exception:
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