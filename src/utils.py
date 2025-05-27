# src/utils.py
import pandas as pd
import numpy as np
from io import StringIO
import empyrical
import streamlit as st # Importado aquí porque es necesario para st.error/warning en load_data/load_dynamic_weights

def load_data(uploaded_file):
    """Carga y procesa el archivo CSV subido por el usuario (precios)."""
    if uploaded_file is None:
        return None
    try:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        
        # Intentar detectar separador y decimal, y también especificar un formato de fecha común
        # Se prioriza YYYY-MM-DD para compatibilidad
        date_fmts = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y'] # Añadir otros formatos si son comunes en tus datos
        
        data = None
        for sep_char in [';', ',']:
            for dec_char in [',', '.']:
                for date_fmt in date_fmts:
                    try:
                        stringio.seek(0) # Siempre reiniciar el puntero del archivo
                        data = pd.read_csv(stringio, sep=sep_char, decimal=dec_char, parse_dates=[0], index_col=0, date_format=date_fmt)
                        if not data.empty and isinstance(data.index, pd.DatetimeIndex):
                            # Éxito en la lectura con este formato, salir de los bucles
                            break
                        else:
                            data = None # Resetear si no es un éxito completo (ej. parse_dates falla silenciosamente)
                    except (pd.errors.ParserError, ValueError, IndexError, KeyError):
                        data = None # Continuar probando
                if data is not None: break # Salir de los bucles de decimal y separador si ya encontramos el formato

        if data is None:
            # Si no se encontró un formato directo, intentar con infer_datetime_format
            stringio.seek(0)
            try:
                # Probar formatos comunes de nuevo, pero con infer_datetime_format si la columna es 'object'
                # Esto es más lento pero más robusto si el formato de fecha es variable
                for sep_char in [';', ',']:
                    for dec_char in [',', '.']:
                        stringio.seek(0)
                        temp_data = pd.read_csv(stringio, sep=sep_char, decimal=dec_char, index_col=0, header=0) # Leer sin parse_dates inicial
                        if not temp_data.empty:
                            try:
                                temp_data.index = pd.to_datetime(temp_data.index, infer_datetime_format=True)
                                data = temp_data
                                break # Salir si tiene éxito
                            except Exception:
                                pass # No es un formato de fecha inferible
                    if data is not None: break

            except Exception:
                pass # Falló la lectura general

        if data is None:
            st.error("No se pudo cargar el archivo CSV de precios. Revisa el formato: Fecha en 1ª col, separador (',' o ';'), decimal (',' o '.').")
            return None

        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index, infer_datetime_format=True)
            except Exception as e:
                st.error(f"Error al convertir la columna de fechas a formato de fecha y hora: {e}. Asegúrate de que la primera columna es una fecha válida.")
                return None

        data.dropna(axis=0, how='all', inplace=True)
        data.sort_index(inplace=True)
        data.ffill(inplace=True)
        data.bfill(inplace=True)

        for col in data.columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                try:
                    if data[col].dtype == 'object':
                        # Intentar limpiar y convertir números con coma como decimal
                        cleaned_col = data[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
                        data[col] = pd.to_numeric(cleaned_col)
                    else:
                        data[col] = pd.to_numeric(data[col])
                except (ValueError, AttributeError, TypeError) as e_conv:
                    st.error(f"Error convirtiendo columna '{col}' a numérico: {e_conv}. Verifica el formato.")
                    return None
        # Asegurarse de que no hay columnas completamente NaN después de la conversión
        data.dropna(axis=1, how='all', inplace=True)
        if data.empty:
            st.error("El archivo de precios no contiene datos válidos después de la limpieza.")
            return None
        return data
    except Exception as e:
        st.error(f"Error crítico al procesar el archivo CSV de precios: {e}")
        st.error("Verifica el formato: 1ª col Fecha, siguientes ISINs/Benchmark; sep CSV (',' o ';'); decimal (',' o '.').")
        return None


def load_dynamic_weights(uploaded_weights_file):
    """Carga y procesa el archivo de pesos dinámicos."""
    if uploaded_weights_file is None:
        return None

    try:
        stringio = StringIO(uploaded_weights_file.getvalue().decode("utf-8"))
        
        # Intentar cargar con diferentes separadores, decimales y formatos de fecha
        date_fmts = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']
        weights_df = None

        for sep_char in [';', ',']:
            for dec_char in [',', '.']:
                for date_fmt in date_fmts:
                    try:
                        stringio.seek(0)
                        temp_df = pd.read_csv(stringio, sep=sep_char, decimal=dec_char, parse_dates=[0], index_col=0, date_format=date_fmt)
                        if not temp_df.empty and isinstance(temp_df.index, pd.DatetimeIndex):
                            weights_df = temp_df
                            break
                        else:
                            temp_df = None # Reset si no es un éxito completo
                    except (pd.errors.ParserError, ValueError, IndexError, KeyError):
                        temp_df = None # Continuar probando
                if weights_df is not None: break

        if weights_df is None:
            # Fallback a infer_datetime_format si no se encontró un formato explícito
            stringio.seek(0)
            try:
                for sep_char in [';', ',']:
                    for dec_char in [',', '.']:
                        stringio.seek(0)
                        temp_df = pd.read_csv(stringio, sep=sep_char, decimal=dec_char, index_col=0, header=0)
                        if not temp_df.empty:
                            try:
                                temp_df.index = pd.to_datetime(temp_df.index, infer_datetime_format=True)
                                weights_df = temp_df
                                break
                            except Exception:
                                pass
                    if weights_df is not None: break
            except Exception:
                pass


        if weights_df is None:
            st.error("No se pudo cargar el archivo de pesos. Revisa el formato: Fecha en 1ª col, separador (',' o ';'), decimal (',' o '.').")
            return None

        if not isinstance(weights_df.index, pd.DatetimeIndex):
            try:
                weights_df.index = pd.to_datetime(weights_df.index, infer_datetime_format=True)
            except Exception as e:
                st.error(f"Error al convertir la columna de fechas en el archivo de pesos: {e}. Asegúrate de que la primera columna es una fecha válida.")
                return None

        weights_df.dropna(axis=0, how='all', inplace=True) # Eliminar filas completamente vacías
        weights_df.sort_index(inplace=True)

        # Convertir todas las columnas de pesos a numérico y normalizar
        for col in weights_df.columns:
            if not pd.api.types.is_numeric_dtype(weights_df[col]):
                try:
                    if weights_df[col].dtype == 'object':
                        cleaned_col = weights_df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
                        weights_df[col] = pd.to_numeric(cleaned_col)
                    else:
                        weights_df[col] = pd.to_numeric(weights_df[col])
                except (ValueError, AttributeError, TypeError) as e_conv:
                    st.warning(f"Advertencia: La columna '{col}' en el archivo de pesos no pudo ser convertida a numérico. Se omitirá. Error: {e_conv}")
                    weights_df.drop(columns=[col], inplace=True)
                    continue
            # Asegurar que los pesos son positivos o cero
            weights_df[col] = weights_df[col].apply(lambda x: max(0.0, x) if pd.notna(x) else 0.0)

        # Normalizar pesos en cada fila
        normalized_weights = pd.DataFrame(index=weights_df.index, columns=weights_df.columns, dtype=float)
        for idx, row in weights_df.iterrows():
            total_sum = row.sum()
            if total_sum > 1e-6: # Evitar división por cero
                if np.isclose(total_sum, 1.0) or np.isclose(total_sum, 100.0): # Si son porcentajes o ya están en decimal
                     if np.isclose(total_sum, 100.0): # Si son porcentajes, convertir a decimal
                         normalized_weights.loc[idx] = row / 100.0
                     else: # Si ya están normalizados a 1
                         normalized_weights.loc[idx] = row
                else: # Si no suman 1 ni 100, se asume que necesitan normalización
                    normalized_weights.loc[idx] = row / total_sum
            else:
                normalized_weights.loc[idx] = 0.0 # Si la suma es cero, los pesos son cero

        normalized_weights.ffill(inplace=True) # Propagar últimos pesos válidos
        normalized_weights.bfill(inplace=True) # Rellenar hacia atrás si los primeros son NaN

        if normalized_weights.empty:
            st.error("El archivo de pesos no contiene datos válidos después de la limpieza y normalización.")
            return None

        return normalized_weights
    except Exception as e:
        st.error(f"Error crítico al procesar el archivo de pesos: {e}")
        st.error("Verifica el formato: 1ª col Fecha, siguientes ISINs/Benchmark; sep CSV (',' o ';'); decimal (',' o '.').")
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
    except Exception: # Fallback manual si empyrical falla por algún caso borde
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