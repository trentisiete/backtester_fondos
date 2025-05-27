import pandas as pd
import numpy as np
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.cloud import bigquery
from google.auth.exceptions import DefaultCredentialsError
import ast # To convert string dictionary to dictionary
import re # For regular expressions
from datetime import datetime, timedelta # Import datetime and timedelta explicitly
from dateutil.relativedelta import relativedelta
import time # For time.sleep
import os # To check for file existence
from pathlib import Path # For path operations

# --- Configuration ---
ARCHIVO_CREDENCIALES_BQ = 'alvaro-martinezgamo_key.json'
PROJECT_ID_BQ = 'ironia-data'
ARCHIVO_CARTERAS_CSV = 'carteras_liga_ironia.csv'
USER_SP500_CSV_PATH = "sp500.csv" # Path to your local S&P500 CSV

def cargar_credenciales_bigquery(ruta_archivo_json_credenciales):
    """
    Loads Google Cloud credentials from a service account JSON file.
    """
    try:
        credentials = service_account.Credentials.from_service_account_file(
            ruta_archivo_json_credenciales,
            scopes=["https://www.googleapis.com/auth/bigquery"]
        )
        credentials.refresh(Request())
        print("Credenciales de BigQuery cargadas y token refrescado con éxito.")
        return credentials
    except DefaultCredentialsError as e:
        print(f"Error de credenciales: {e}")
        return None
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de credenciales en '{ruta_archivo_json_credenciales}'.")
        return None
    except Exception as e:
        print(f"Se produjo un error al cargar las credenciales: {e}")
        return None

def parse_complex_isins_string(text_data):
    """
    Parses the complex-formatted ISIN string using regular expressions.
    """
    pattern = re.compile(r"(?:[^,{}]*?):\s*\(([^,]+?),\s*([\d\.]+?%)\)")
    if text_data.startswith("{") and text_data.endswith("}"):
        text_data = text_data[1:-1]
    matches = pattern.findall(text_data)
    if not matches: return None
    parsed_dict = {}
    for isin_str, pct_str in matches:
        isin_str, pct_str = isin_str.strip(), pct_str.strip()
        try:
            parsed_dict[isin_str] = float(pct_str.replace('%', '')) / 100.0
        except ValueError:
            print(f"Aviso: No se pudo convertir porcentaje '{pct_str}' para ISIN '{isin_str}'. Se omitirá.")
    return parsed_dict

def cargar_y_seleccionar_cartera(ruta_csv_carteras):
    """
    Loads portfolios from a CSV file and allows the user to select one.
    """
    try:
        df_carteras = pd.read_csv(ruta_csv_carteras, dtype={'ISINs_Porcentajes': str})
    except FileNotFoundError:
        print(f"Error: No se encontró '{ruta_csv_carteras}'.")
        return None, None
    except Exception as e:
        print(f"Error al leer CSV de carteras: {e}")
        return None, None

    if df_carteras.empty or not all(c in df_carteras.columns for c in ['NombreCartera', 'ISINs_Porcentajes']):
        print("Error: CSV de carteras vacío o columnas faltantes.")
        return None, None

    print("\nCarteras disponibles:")
    for i, r in df_carteras.iterrows(): print(f"{i+1}. {r['NombreCartera']}")

    while True:
        try:
            sel = int(input("Selecciona el número de la cartera: "))
            if 1 <= sel <= len(df_carteras):
                cartera = df_carteras.iloc[sel - 1]
                nombre_c = cartera['NombreCartera']
                isins_str = cartera['ISINs_Porcentajes']
                if pd.isna(isins_str) or not isins_str.strip():
                    print(f"Error: 'ISINs_Porcentajes' vacío para '{nombre_c}'.")
                    return None, None

                raw_dict = parse_complex_isins_string(isins_str)
                if raw_dict is None:
                    try:
                        raw_dict = ast.literal_eval(isins_str)
                        if not isinstance(raw_dict, dict): raise ValueError("El string no es un diccionario Python válido")
                    except (ValueError, SyntaxError) as e:
                        print(f"Error procesando ISINs para '{nombre_c}': {e}")
                        return None, None

                final_dict = {}
                placeholders = ["isin no encontrado", "not_found", "placeholder"]
                for k, v in raw_dict.items():
                    if not (isinstance(k, str) and isinstance(v, (int, float))): continue
                    k_up = k.strip().upper()
                    if not any(p.upper() in k_up for p in placeholders) and len(k_up) == 12 and k_up.isalnum():
                        final_dict[k_up] = v
                    # else: print(f"Aviso: ISIN/Placeholder '{k}' omitido.") # Opcional: más verbosidad

                if not final_dict:
                    print(f"Error: No hay ISINs válidos para '{nombre_c}'.")
                    return None, None
                return nombre_c, final_dict
            else: print("Selección no válida.")
        except ValueError: print("Entrada no válida.")

def obtener_precios_fondos_bigquery(nombre_cartera, isins_dict, project_id, credentials):
    """
    Obtains fund prices (NAV) from BigQuery.
    """
    if not isins_dict: return None
    lista_isins_sql = ",".join([f"'{i}'" for i in isins_dict.keys()])
    # Start date is 3 years from today.
    fecha_inicio_str = (datetime.now() - relativedelta(years=3)).strftime('%Y-%m-%d')


    sql = f"""SELECT * FROM (
                  SELECT date, UPPER(isin) as isin, nav FROM `{project_id}.ironia.navs`
                  WHERE date BETWEEN '{fecha_inicio_str}' AND CURRENT_DATE() AND UPPER(isin) IN ({lista_isins_sql})
                ) PIVOT (MAX(nav) FOR isin IN ({lista_isins_sql})) ORDER BY date ASC"""
    print(f"\nExecuting query in BigQuery for '{nombre_cartera}'...")
    try:
        client = bigquery.Client(project=project_id, credentials=credentials)
        df = client.query(sql).to_dataframe()
        if df.empty: print(f"No se encontraron datos en BigQuery para '{nombre_cartera}'."); return None
        print("Datos de fondos obtenidos de BigQuery.")
        df.columns = [c.upper() if c.lower() != 'date' else 'date' for c in df.columns]
        return df
    except Exception as e:
        print(f"Error en consulta BigQuery: {e}"); return None

def _parse_price_string_robust(price_val):
    """
    Converts a price value (possibly string) to float robustly.
    Handles formats like "1.234,56" (European) and "1234.56" (US) or "1234,56" (European without thousands).
    """
    if pd.isna(price_val):
        return np.nan

    price_str = str(price_val).strip()

    # If there are commas and periods, assume European format: "1.234,56"
    if ',' in price_str and '.' in price_str:
        # Remove periods (thousands separator)
        price_str = price_str.replace('.', '')
        # Replace comma (decimal) by period
        price_str = price_str.replace(',', '.')
    # If there are only commas, assume European decimal format: "1234,56"
    elif ',' in price_str:
        price_str = price_str.replace(',', '.')
    # If only periods (or none, or it's already a number), pd.to_numeric should handle it
    # e.g.: "1234.56" or "1234"

    return pd.to_numeric(price_str, errors='coerce')


def _parse_user_sp500_csv(filepath):
    """Reads the S&P500 CSV provided by the user robustly."""
    try:
        print(f"Intentando cargar datos del S&P 500 desde el archivo local del usuario: {filepath}")
        
        # Try reading with comma (,) as separator first, matching the user's example
        try:
            df = pd.read_csv(filepath, sep=',', dtype={'Precio Ajustado': str, 'Fecha': str})
        except Exception:
            # Fallback to semicolon if comma fails (less likely based on user's example)
            df = pd.read_csv(filepath, sep=';', dtype={'Precio Ajustado': str, 'Fecha': str})

        # Standardize column names after reading, regardless of original case/name
        if 'Fecha' in df.columns:
            date_col = 'Fecha'
        elif 'Date' in df.columns: # Common alternative for date column
            date_col = 'Date'
        else:
            print(f"Advertencia: {filepath} no contiene una columna de fecha reconocida ('Fecha' o 'Date').")
            return None

        if 'Precio Ajustado' in df.columns:
            price_col = 'Precio Ajustado'
        elif 'Adj Close' in df.columns: # Common alternative for adjusted close
            price_col = 'Adj Close'
        elif 'Close' in df.columns: # Common alternative for close
            price_col = 'Close'
        else:
            print(f"Advertencia: {filepath} no contiene una columna de precio reconocida ('Precio Ajustado', 'Adj Close' o 'Close').")
            return None
        
        # Rename columns to standardized names for internal use
        df = df.rename(columns={date_col: 'Fecha', price_col: 'Precio Ajustado'}).copy()


        df['Fecha'] = pd.to_datetime(df['Fecha'], format='%Y-%m-%d', errors='coerce').dt.date

        # Apply the robust parsing function
        df['SP500_Close'] = df['Precio Ajustado'].apply(_parse_price_string_robust)

        df.dropna(subset=['Fecha', 'SP500_Close'], inplace=True)
        if df.empty:
            print(f"{filepath} está vacío o no contiene datos válidos después del procesamiento.")
            return None

        print(f"Cargados {len(df)} registros válidos desde {filepath}.")
        return df[['Fecha', 'SP500_Close']]
    except FileNotFoundError:
        print(f"Archivo S&P 500 del usuario no encontrado en: {filepath}")
        return None
    except Exception as e:
        print(f"Error al leer o procesar el CSV del usuario {filepath}: {e}")
        return None

def obtener_y_procesar_sp500(start_date_req, end_date_req):
    """
    Obtains S&P500 data from the user's local CSV.
    start_date_req and end_date_req must be datetime.date objects.
    """
    # 1. Load from user CSV
    df_sp500_local = _parse_user_sp500_csv(USER_SP500_CSV_PATH)

    if df_sp500_local is None or df_sp500_local.empty:
        print("No se pudieron obtener datos del S&P 500 del CSV local del usuario.")
        return None

    # Filter to the exact requested range
    final_df_sp500 = df_sp500_local[
        (df_sp500_local['Fecha'] >= start_date_req) &
        (df_sp500_local['Fecha'] <= end_date_req)
    ].copy()

    if final_df_sp500.empty:
        print(f"No se encontraron datos del S&P 500 para el rango específico: {start_date_req.strftime('%Y-%m-%d')} a {end_date_req.strftime('%Y-%m-%d')} en el CSV local.")
        return None

    print(f"S&P 500 procesado. {len(final_df_sp500)} registros para el rango solicitado.")
    return final_df_sp500

def procesar_y_guardar_datos(precios_df_fondos, nombre_cartera):
    """
    Processes the fund prices DataFrame, joins it with S&P 500, and saves it to a CSV file.
    """
    if precios_df_fondos is None or precios_df_fondos.empty:
        print(f"No hay datos de fondos para procesar para '{nombre_cartera}'.")
        return

    if 'date' not in precios_df_fondos.columns:
        print("Advertencia: Columna 'date' no encontrada en datos de fondos.")
        return

    precios_df_fondos['date'] = pd.to_datetime(precios_df_fondos['date'])
    precios_df_fondos = precios_df_fondos.set_index('date')

    df_limpio = precios_df_fondos.dropna(axis=1, how='all').dropna(axis=0, how='any')
    if df_limpio.empty:
        print(f"Todos los datos de fondos eliminados tras limpieza para '{nombre_cartera}'."); return

    df_para_guardar = df_limpio.reset_index()
    df_para_guardar['date'] = pd.to_datetime(df_para_guardar['date']).dt.normalize().dt.date
    df_para_guardar.rename(columns={'date': 'Fecha'}, inplace=True)

    if not df_para_guardar.empty and 'Fecha' in df_para_guardar.columns:
        min_fecha, max_fecha = df_para_guardar['Fecha'].min(), df_para_guardar['Fecha'].max()
        
        # Obtener datos del S&P 500 solo del CSV local del usuario
        df_sp500 = obtener_y_procesar_sp500(min_fecha, max_fecha)

        if df_sp500 is not None and not df_sp500.empty:
            print("Uniendo datos de fondos con S&P 500...")
            df_para_guardar = pd.merge(df_para_guardar, df_sp500, on='Fecha', how='left')
            df_para_guardar['SP500_Close'] = pd.to_numeric(df_para_guardar['SP500_Close'], errors='coerce')
            df_para_guardar.sort_values(by='Fecha', inplace=True)
            # Use the recommended syntax for ffill and bfill to avoid FutureWarning
            df_para_guardar['SP500_Close'] = df_para_guardar['SP500_Close'].ffill()
            df_para_guardar['SP500_Close'] = df_para_guardar['SP500_Close'].bfill() # For NaNs at the beginning
            print("Unión y relleno S&P 500 completado.")
        else:
            print("No se pudieron obtener datos del S&P 500 para el rango solicitado del CSV local. Se guardará sin ellos.")
            df_para_guardar['SP500_Close'] = np.nan
    else:
        # If df_para_guardar is empty or has no 'Fecha', add an empty SP500_Close column
        df_para_guardar['SP500_Close'] = np.nan

    # Convert 'Fecha' column to string in YYYY-MM-DD format for CSV
    # Only if the column exists and is not completely NaN
    if 'Fecha' in df_para_guardar.columns:
        df_para_guardar['Fecha'] = df_para_guardar['Fecha'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) and hasattr(x, 'strftime') else '')


    nombre_archivo_csv = f"precios_fondos_{nombre_cartera.replace(' ', '_').replace(',', '')}.csv"
    try:
        print(f"Guardando datos combinados en: {nombre_archivo_csv}")
        df_para_guardar.to_csv(
            nombre_archivo_csv,
            sep=';',
            decimal=',',
            index=False,
            encoding='utf-8-sig',
            float_format='%.2f' # Ensure two-decimal format for all floats
        )
        print(f"Datos guardados con éxito en '{nombre_archivo_csv}'.")
    except Exception as e:
        print(f"Error al guardar el archivo CSV final: {e}")

def main():
    print("Iniciando script...")
    # Ensure the directory for the S&P500 CSV exists if it's not in the current directory
    Path(USER_SP500_CSV_PATH).parent.mkdir(parents=True, exist_ok=True)

    credenciales = cargar_credenciales_bigquery(ARCHIVO_CREDENCIALES_BQ)
    if not credenciales: return

    nombre_c, isins_d = cargar_y_seleccionar_cartera(ARCHIVO_CARTERAS_CSV)
    if not nombre_c or not isins_d: return

    print(f"\nProcesando cartera: '{nombre_c}' con ISINs: {isins_d}")
    df_fondos = obtener_precios_fondos_bigquery(nombre_c, isins_d, PROJECT_ID_BQ, credenciales)

    if df_fondos is not None:
        procesar_y_guardar_datos(df_fondos, nombre_c)
    else:
        print(f"No se obtuvieron datos de fondos para '{nombre_c}'.")
    print("\nScript finalizado.")

if __name__ == "__main__":
    main()
