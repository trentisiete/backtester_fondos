import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import time
import os
from pathlib import Path

class SP500DataDownloader:
    """
    Clase para descargar datos históricos del S&P 500 desde varias fuentes
    de API y web scraping, y guardarlos en un archivo CSV.
    """
    def __init__(self, csv_filename="sp500.csv", years=5):
        """
        Inicializa el descargador de datos.

        Args:
            csv_filename (str): Nombre del archivo CSV donde se guardarán los datos.
            years (int): Número de años de datos históricos a intentar descargar.
        """
        self.csv_filename = csv_filename
        self.years = years
        # Asegura que el directorio para el archivo CSV exista
        self.base_path = Path(csv_filename).parent
        self.base_path.mkdir(parents=True, exist_ok=True)

    def get_alpha_vantage_data(self, api_key, symbol="^GSPC"):
        """
        Obtiene datos usando Alpha Vantage API.
        Requiere una clave gratuita. Registrarse en: https://www.alphavantage.co/support/#api-key

        Args:
            api_key (str): Tu clave de API de Alpha Vantage.
            symbol (str): Símbolo del activo a descargar (por defecto "^GSPC" para S&P 500).

        Returns:
            pd.DataFrame or None: DataFrame con 'Fecha' y 'Precio Ajustado', o None si falla.
        """
        # Alpha Vantage usa formato especial para índices
        if symbol == "^GSPC":
            symbol = "SPX"  # Alpha Vantage usa SPX para S&P 500

        url = f"https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': symbol,
            'outputsize': 'full', # 'full' para obtener el historial completo disponible
            'apikey': api_key
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()

            if 'Time Series (Daily)' in data:
                time_series = data['Time Series (Daily)']
                df_data = []

                for date, values in time_series.items():
                    df_data.append({
                        'Fecha': pd.to_datetime(date),
                        'Precio Ajustado': float(values.get('5. adjusted close', values.get('4. close', 0)))
                    })

                df = pd.DataFrame(df_data)
                df = df.sort_values('Fecha').reset_index(drop=True)

                # Filtrar por el número de años solicitado
                min_date = datetime.now() - timedelta(days=self.years * 365)
                df = df[df['Fecha'] >= min_date].copy()

                return df
            else:
                print(f"Error en Alpha Vantage: {data.get('Note', data)}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Error de conexión con Alpha Vantage: {e}")
            return None
        except json.JSONDecodeError:
            print("Error decodificando la respuesta JSON de Alpha Vantage.")
            return None
        except Exception as e:
            print(f"Error inesperado con Alpha Vantage: {e}")
            return None

    def get_tiingo_data(self, api_key, symbol="^GSPC"):
        """
        Obtiene datos usando Tiingo API.
        Requiere una clave gratuita. Registrarse en: https://api.tiingo.com/

        Args:
            api_key (str): Tu clave de API de Tiingo.
            symbol (str): Símbolo del activo a descargar (por defecto "^GSPC" para S&P 500).

        Returns:
            pd.DataFrame or None: DataFrame con 'Fecha' y 'Precio Ajustado', o None si falla.
        """
        headers = {'Content-Type': 'application/json'}

        # Tiingo usa formato especial para índices
        if symbol == "^GSPC":
            symbol = "SPX"  # Tiingo también usa SPX para S&P 500

        # Calcular fechas basándose en self.years
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=self.years * 365)).strftime('%Y-%m-%d')

        url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
        params = {
            'startDate': start_date,
            'endDate': end_date,
            'token': api_key,
            'format': 'json'
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                df_data = []

                for item in data:
                    df_data.append({
                        'Fecha': pd.to_datetime(item['date']),
                        'Precio Ajustado': float(item.get('adjClose', item.get('close', 0)))
                    })

                df = pd.DataFrame(df_data)
                df = df.sort_values('Fecha').reset_index(drop=True)
                return df
            else:
                print(f"Error en Tiingo: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Error de conexión con Tiingo: {e}")
            return None
        except json.JSONDecodeError:
            print("Error decodificando la respuesta JSON de Tiingo.")
            return None
        except Exception as e:
            print(f"Error inesperado con Tiingo: {e}")
            return None

    def get_marketstack_data(self, api_key, symbol="^GSPC"):
        """
        Obtiene datos usando Marketstack API.
        Requiere una clave gratuita. Registrarse en: https://marketstack.com/

        Args:
            api_key (str): Tu clave de API de Marketstack.
            symbol (str): Símbolo del activo a descargar (por defecto "^GSPC" para S&P 500).

        Returns:
            pd.DataFrame or None: DataFrame con 'Fecha' y 'Precio Ajustado', o None si falla.
        """
        # Marketstack usa formato diferente para índices
        if symbol == "^GSPC":
            symbol = "SPX.INDX"  # Marketstack usa este formato para S&P 500

        url = "http://api.marketstack.com/v1/eod"
        params = {
            'access_key': api_key,
            'symbols': symbol,
            'limit': self.years * 252, # Estimación de días de trading por año (252)
            'sort': 'DESC'
        }

        try:
            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()

                if 'data' in data and data['data']:
                    df_data = []

                    for item in data['data']:
                        df_data.append({
                            'Fecha': pd.to_datetime(item['date']),
                            'Precio Ajustado': float(item.get('adj_close', item.get('close', 0)))
                        })

                    df = pd.DataFrame(df_data)
                    df = df.sort_values('Fecha').reset_index(drop=True)

                    # Filtrar por el número de años solicitado
                    min_date = datetime.now() - timedelta(days=self.years * 365)
                    df = df[df['Fecha'] >= min_date].copy()

                    return df
                else:
                    print(f"Error en Marketstack: {data.get('error', data)}")
                    return None
            else:
                print(f"Error en Marketstack: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Error de conexión con Marketstack: {e}")
            return None
        except json.JSONDecodeError:
            print("Error decodificando la respuesta JSON de Marketstack.")
            return None
        except Exception as e:
            print(f"Error inesperado con Marketstack: {e}")
            return None

    def get_polygon_data(self, api_key, symbol="^GSPC"):
        """
        Obtiene datos usando Polygon.io API.
        Requiere una clave gratuita. Registrarse en: https://polygon.io/

        Args:
            api_key (str): Tu clave de API de Polygon.io.
            symbol (str): Símbolo del activo a descargar (por defecto "^GSPC" para S&P 500).

        Returns:
            pd.DataFrame or None: DataFrame con 'Fecha' y 'Precio Ajustado', o None si falla.
        """
        # Polygon usa formato I: para índices
        if symbol == "^GSPC":
            symbol = "I:SPX"  # Polygon usa este formato para S&P 500

        # Calcular fechas basándose en self.years
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=self.years * 365)).strftime('%Y-%m-%d')

        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'apikey': api_key
        }

        try:
            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()

                if 'results' in data and data['results']:
                    df_data = []
                    for item in data['results']:
                        date = datetime.fromtimestamp(item['t'] / 1000) # Polygon devuelve timestamp en milisegundos
                        df_data.append({
                            'Fecha': date,
                            'Precio Ajustado': float(item['c'])  # 'c' es el precio de cierre
                        })

                    df = pd.DataFrame(df_data)
                    df = df.sort_values('Fecha').reset_index(drop=True)
                    return df
                else:
                    print(f"Error en Polygon: No se encontraron resultados o {data.get('error', data)}")
                    return None
            else:
                print(f"Error en Polygon: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Error de conexión con Polygon: {e}")
            return None
        except json.JSONDecodeError:
            print("Error decodificando la respuesta JSON de Polygon.")
            return None
        except Exception as e:
            print(f"Error inesperado con Polygon: {e}")
            return None

    def get_fred_data(self):
        """
        Obtiene datos del S&P 500 desde TwelveData API (gratuito, sin API key).
        Este método reemplaza la lógica original de FRED con TwelveData como primera opción gratuita.

        Returns:
            pd.DataFrame or None: DataFrame con 'Fecha' y 'Precio Ajustado', o None si falla.
        """
        try:
            # Intentar con TwelveData API (gratuito, sin API key para uso básico)
            url = "https://api.twelvedata.com/time_series"
            params = {
                'symbol': 'SPX',
                'interval': '1day',
                'outputsize': self.years * 252, # Ajustar el outputsize a los años solicitados
                'format': 'JSON'
            }

            print("Intentando obtener datos del S&P 500 desde TwelveData...")
            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()

                if 'values' in data and data['values']:
                    df_data = []
                    for item in data['values']:
                        if item.get('close') and item.get('close') != 'null':
                            df_data.append({
                                'Fecha': pd.to_datetime(item['datetime']),
                                'Precio Ajustado': float(item['close'])
                            })

                    if df_data:
                        df = pd.DataFrame(df_data)
                        df = df.sort_values('Fecha').reset_index(drop=True)
                        print("Datos obtenidos exitosamente de TwelveData.")
                        return df
                    else:
                        print("TwelveData no devolvió valores de cierre válidos.")
                else:
                    print(f"Error en TwelveData: {data.get('message', 'No se encontraron datos.')}")
            else:
                print(f"Error en TwelveData: {response.status_code} - {response.text}")

            # Si TwelveData falla o no devuelve datos, intentar el método de respaldo
            print("TwelveData falló o no devolvió datos. Intentando método de respaldo (web scraping)...")
            return self.get_investing_data()

        except requests.exceptions.RequestException as e:
            print(f"Error de conexión con TwelveData: {e}")
            print("Intentando método de respaldo (web scraping)...")
            return self.get_investing_data()
        except json.JSONDecodeError:
            print("Error decodificando la respuesta JSON de TwelveData.")
            print("Intentando método de respaldo (web scraping)...")
            return self.get_investing_data()
        except Exception as e:
            print(f"Error inesperado con TwelveData: {e}")
            print("Intentando método de respaldo (web scraping)...")
            return self.get_investing_data()

    def get_investing_data(self):
        """
        Método de respaldo usando web scraping básico (STOOQ o Yahoo Finance).
        Este método es menos fiable y puede fallar debido a cambios en las estructuras de las páginas web.

        Returns:
            pd.DataFrame or None: DataFrame con 'Fecha' y 'Precio Ajustado', o None si falla.
        """
        try:
            urls_to_try = [
                # STOOQ - fuente europea con datos históricos del S&P 500
                "https://stooq.com/q/d/l/?s=^spx&i=d",
                # Backup con Yahoo Finance directo
                "https://query1.finance.yahoo.com/v7/finance/download/%5EGSPC"
            ]

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            for url in urls_to_try:
                print(f"Intentando obtener datos de: {url}...")
                try:
                    if "yahoo" in url:
                        # Para Yahoo Finance, usar parámetros específicos para el rango de años solicitado
                        params = {
                            'period1': int((datetime.now() - timedelta(days=self.years * 365)).timestamp()),
                            'period2': int(datetime.now().timestamp()),
                            'interval': '1d',
                            'events': 'history',
                            'includeAdjustedClose': 'true'
                        }
                        response = requests.get(url, params=params, headers=headers, timeout=30)
                    else:
                        # Para STOOQ, no necesita parámetros especiales, se filtrará después de la descarga
                        response = requests.get(url, headers=headers, timeout=30)

                    if response.status_code == 200 and len(response.text) > 100:
                        # Leer CSV directamente desde la respuesta
                        from io import StringIO
                        df = pd.read_csv(StringIO(response.text))

                        # Verificar y renombrar columnas según la fuente
                        date_col = None
                        if 'Date' in df.columns:
                            date_col = 'Date'
                        elif 'date' in df.columns:
                            date_col = 'date'
                        else:
                            print(f"No se encontró columna de fecha en {url}. Saltando.")
                            continue

                        price_col = None
                        if 'Adj Close' in df.columns:
                            price_col = 'Adj Close'
                        elif 'Close' in df.columns:
                            price_col = 'Close'
                        elif 'close' in df.columns:
                            price_col = 'close'
                        else:
                            print(f"No se encontró columna de precio en {url}. Saltando.")
                            continue

                        # Renombrar columnas al formato requerido
                        df = df.rename(columns={
                            date_col: 'Fecha',
                            price_col: 'Precio Ajustado'
                        })

                        # Seleccionar solo las columnas necesarias
                        df = df[['Fecha', 'Precio Ajustado']].copy()
                        df['Fecha'] = pd.to_datetime(df['Fecha'])
                        df['Precio Ajustado'] = pd.to_numeric(df['Precio Ajustado'], errors='coerce')
                        df = df.dropna()  # Eliminar valores nulos que puedan surgir de la conversión

                        df = df.sort_values('Fecha').reset_index(drop=True)

                        # Filtrar por el número de años solicitado para STOOQ
                        min_date = datetime.now() - timedelta(days=self.years * 365)
                        df = df[df['Fecha'] >= min_date].copy()


                        if len(df) > 100:  # Verificar que tenemos suficientes datos
                            print(f"Datos obtenidos exitosamente de {url}.")
                            return df
                        else:
                            print(f"Pocos datos obtenidos de {url} ({len(df)} registros). Intentando otra fuente.")

                except requests.exceptions.RequestException as e:
                    print(f"Error de conexión con {url}: {e}")
                except Exception as e:
                    print(f"Error procesando datos de {url}: {e}")
                time.sleep(1) # Esperar antes del siguiente intento

            print("No se pudieron obtener datos de ninguna URL de respaldo.")
            return None

        except Exception as e:
            print(f"Error general con método de respaldo: {e}")
            return None

    def update_data(self, api_keys=None):
        """
        Actualiza los datos del S&P 500 probando diferentes fuentes.

        Args:
            api_keys (dict, optional): Diccionario con las claves de API.
                                       Ej: {'alpha_vantage': 'tu_clave', 'tiingo': 'tu_clave'}.
                                       Si es None o vacío, solo intentará fuentes gratuitas.

        Returns:
            bool: True si los datos se actualizaron exitosamente, False en caso contrario.
        """
        # Primero intentar con TwelveData API (gratuito, sin API key)
        print("🔄 Intentando con TwelveData API (fuente gratuita)...")
        df = self.get_fred_data()  # Esta función ahora maneja TwelveData y el scraping de respaldo

        if df is not None and not df.empty:
            df.to_csv(self.csv_filename, index=False)
            print(f"✅ Datos actualizados exitosamente usando TwelveData API o respaldo web scraping.")
            print(f"📁 Archivo guardado: {self.csv_filename}")
            print(f"📊 Registros: {len(df)}")
            print(f"📅 Rango: {df['Fecha'].min().strftime('%Y-%m-%d')} a {df['Fecha'].max().strftime('%Y-%m-%d')}")
            return True

        if api_keys is None or not any(api_keys.values()):
            print("⚠️ TwelveData/Respaldo no disponible y no se proporcionaron claves de API válidas.")
            print("Puedes obtener claves gratuitas en:")
            print("- Alpha Vantage: https://www.alphavantage.co/support/#api-key")
            print("- Tiingo: https://api.tiingo.com/")
            print("- Marketstack: https://marketstack.com/")
            print("- Polygon: https://polygon.io/")
            return False

        # Si las fuentes gratuitas fallan, intentar con las APIs proporcionadas
        sources = [
            ('Alpha Vantage', self.get_alpha_vantage_data, api_keys.get('alpha_vantage')),
            ('Tiingo', self.get_tiingo_data, api_keys.get('tiingo')),
            ('Marketstack', self.get_marketstack_data, api_keys.get('marketstack')),
            ('Polygon', self.get_polygon_data, api_keys.get('polygon'))
        ]

        for source_name, method, api_key in sources:
            if api_key: # Solo intentar si hay una clave de API para esta fuente
                print(f"🔄 Intentando con {source_name} (usando clave de API)...")
                df = method(api_key)

                if df is not None and not df.empty:
                    df.to_csv(self.csv_filename, index=False)
                    print(f"✅ Datos actualizados exitosamente usando {source_name}")
                    print(f"📁 Archivo guardado: {self.csv_filename}")
                    print(f"📊 Registros: {len(df)}")
                    print(f"📅 Rango: {df['Fecha'].min().strftime('%Y-%m-%d')} a {df['Fecha'].max().strftime('%Y-%m-%d')}")
                    return True
                else:
                    print(f"❌ {source_name} no devolvió datos válidos o hubo un error.")
                    time.sleep(1)  # Esperar antes del siguiente intento para evitar bloqueos de API

        print("❌ No se pudieron obtener datos de ninguna fuente configurada.")
        return False

    def load_existing_data(self):
        """
        Carga datos existentes si el archivo CSV ya existe.

        Returns:
            pd.DataFrame or None: DataFrame con los datos cargados, o None si el archivo no existe o hay un error.
        """
        if os.path.exists(self.csv_filename):
            try:
                df = pd.read_csv(self.csv_filename)
                df['Fecha'] = pd.to_datetime(df['Fecha'])
                return df
            except Exception as e:
                print(f"Error cargando datos existentes desde {self.csv_filename}: {e}")
                return None
        print(f"El archivo {self.csv_filename} no existe. No hay datos para cargar.")
        return None

    def show_summary(self):
        """
        Muestra un resumen de los datos actuales del S&P 500 cargados desde el CSV.
        """
        df = self.load_existing_data()
        if df is not None and not df.empty:
            print("\n📈 RESUMEN DE DATOS S&P 500")
            print("=" * 40)
            print(f"Archivo: {self.csv_filename}")
            print(f"Registros: {len(df)}")
            print(f"Fecha inicial: {df['Fecha'].min().strftime('%Y-%m-%d')}")
            print(f"Fecha final: {df['Fecha'].max().strftime('%Y-%m-%d')}")
            print(f"Precio más reciente: ${df['Precio Ajustado'].iloc[-1]:.2f}")
            print(f"Precio mínimo: ${df['Precio Ajustado'].min():.2f}")
            print(f"Precio máximo: ${df['Precio Ajustado'].max():.2f}")

            # Mostrar últimos 5 registros
            print("\n📋 Últimos 5 registros:")
            print(df.tail().to_string(index=False))
        else:
            print("\nNo hay datos disponibles para mostrar el resumen.")

def main():
    """
    Función principal para ejecutar el descargador de datos del S&P 500.
    Permite al usuario actualizar los datos y ver un resumen.
    """
    # CONFIGURAR TUS CLAVES DE API AQUÍ
    # Reemplaza 'None' con tus claves reales. Si no tienes una, déjala como 'None'.
    api_keys = {
        'alpha_vantage': "31YBGCSQ0IT2U71W",
        'tiingo': None,
        'marketstack': "3ed87746f4ef4ef385fe16b916ed1983",
        'polygon': "zpapgPZgyKHplBHuDkX_d2tMFfjFHdlN"
    }

    print("🚀 ACTUALIZADOR DE DATOS S&P 500")
    print("=" * 50)

    # Preguntar al usuario cuántos años de datos desea
    years_input = input("¿Cuántos años de datos históricos deseas descargar? (Ej: 1, 2, 5, 10. Por defecto: 5 años): ").strip()
    try:
        years_to_fetch = int(years_input) if years_input else 5
        if years_to_fetch <= 0:
            print("El número de años debe ser un entero positivo. Usando 5 años por defecto.")
            years_to_fetch = 5
    except ValueError:
        print("Entrada inválida. Usando 5 años por defecto.")
        years_to_fetch = 5

    downloader = SP500DataDownloader("sp500.csv", years=years_to_fetch)

    # Mostrar datos actuales si existen
    downloader.show_summary()

    # Preguntar si actualizar
    respuesta = input("\n¿Deseas actualizar los datos? (s/n): ").lower().strip()

    if respuesta in ['s', 'si', 'sí', 'y', 'yes']:
        if not any(api_keys.values()):
            print("\n💡 No se han configurado claves de API. Se intentarán solo fuentes gratuitas y de respaldo.")
        else:
            print("\n💡 Se han configurado claves de API. Se intentarán las fuentes configuradas si las gratuitas fallan.")

        print("\n🔄 Iniciando descarga...")
        success = downloader.update_data(api_keys)
        if success:
            downloader.show_summary()
        else:
            print("\nLa actualización de datos no fue exitosa.")
    else:
        print("Operación de actualización cancelada.")
    print("=" * 50)
    print("Fin del programa.")

if __name__ == "__main__":
    main()
