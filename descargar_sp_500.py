# Primero, asegúrate de tener instaladas las librerías necesarias:
# pip install yfinance pandas

import yfinance as yf
import pandas as pd
import datetime

# --- Parámetros ---
ticker_symbol = "^GSPC" # Símbolo para S&P 500 en Yahoo Finance

# Timestamps de la URL proporcionada
# period1=1588550400 -> 2020-05-04 00:00:00 UTC
# period2=1746397638 -> 2025-05-04 00:37:18 UTC
period1_ts = 1588550400
period2_ts = 1746397638

# Convertir timestamps a formato de fecha YYYY-MM-DD
start_date = pd.to_datetime(period1_ts, unit='s').strftime('%Y-%m-%d')
# Para incluir el día 2025-05-04, ponemos como fecha final 2025-05-05 (exclusivo en yfinance)
end_date_limit = pd.to_datetime(period2_ts, unit='s') + datetime.timedelta(days=1)
end_date = end_date_limit.strftime('%Y-%m-%d')
end_date_display = pd.to_datetime(period2_ts, unit='s').strftime('%Y-%m-%d') # Para el nombre del archivo


# Nombre para el archivo CSV de salida
output_csv_filename = f"sp500_precio_ajustado_{start_date}_a_{end_date_display}.csv"

print(f"Descargando datos históricos para {ticker_symbol}...")
print(f"Período: Desde {start_date} hasta {end_date_display}")

try:
    # Crear un objeto Ticker
    ticker = yf.Ticker(ticker_symbol)

    # Descargar el historial de precios
    hist = ticker.history(start=start_date, end=end_date)

    if hist.empty:
        print(f"No se encontraron datos para {ticker_symbol} en el período especificado.")
    else:
        print(f"Datos descargados exitosamente: {len(hist)} registros encontrados.")

        # Resetear el índice para que 'Date' se convierta en una columna
        hist.reset_index(inplace=True)

        # Asegurarse de que la columna 'Date' está en formato YYYY-MM-DD
        hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None).dt.strftime('%Y-%m-%d')

        # Seleccionar las columnas deseadas: 'Date' y 'Close' (que es el ajustado)
        df_final = hist[['Date', 'Close']].copy()
        df_final.rename(columns={'Date': 'Fecha', 'Close': 'Precio Ajustado'}, inplace=True)

        # --- AJUSTE AQUÍ ---
        # Guardar el DataFrame resultante en un archivo CSV con formato de 2 decimales
        print(f"Guardando datos en {output_csv_filename} con 2 decimales...")
        df_final.to_csv(
            output_csv_filename,
            index=False,          # No incluir el índice del DataFrame en el CSV
            encoding='utf-8-sig', # Codificación compatible con Excel
            sep=';',              # Separador de punto y coma
            decimal=',',           # Coma como separador decimal
            float_format='%.2f'   # <--- LÍNEA AÑADIDA/MODIFICADA
            )
        print(f"Archivo CSV guardado exitosamente como: {output_csv_filename}")

except Exception as e:
    print(f"\nOcurrió un error durante la descarga o procesamiento:")
    print(e)
    print("\nPosibles causas:")
    print("- No tienes conexión a internet.")
    print("- El símbolo del ticker ('{ticker_symbol}') no es válido o ha cambiado.")
    print("- Problema temporal con los servidores de Yahoo Finance.")
    print("- Asegúrate de tener las librerías 'yfinance' y 'pandas' instaladas ('pip install yfinance pandas').")