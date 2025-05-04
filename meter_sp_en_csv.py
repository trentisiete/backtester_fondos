import pandas as pd
import argparse
import os
import numpy as np # Para manejar NaN si es necesario

# --- FUNCIÓN limpiar_precio_sp500 YA NO ES NECESARIA PARA ESTE FORMATO ---
# Se puede eliminar o dejar comentada si prevés usarla para otros formatos
# def limpiar_precio_sp500(precio_str):
#     """Limpia el formato de precio 'X.XXX,XX' a un float."""
#     # ... (código anterior) ...

def unir_csv_con_sp500(sp500_path, otro_path):
    """
    Une dos archivos CSV basándose en la fecha. Añade el precio de cierre del
    S&P 500 al segundo archivo, rellenando fechas sin correspondencia con el
    valor anterior. ADAPTADO para leer el formato específico del nuevo CSV de SP500.
    """
    print(f"--- Iniciando proceso ---")
    print(f"Archivo S&P 500: {sp500_path}")
    print(f"Archivo Otro:    {otro_path}")

    # --- Leer y procesar archivo S&P 500 (MODIFICADO) ---
    try:
        print("Leyendo archivo S&P 500...")
        # === CAMBIOS AQUÍ ===
        df_sp500 = pd.read_csv(
            sp500_path,
            sep=';',       # Cambiado de ',' a ';'
            decimal=','    # Añadido para reconocer la coma decimal en 'Precio Ajustado'
            # Ya no se necesita dtype={'Último': str}
        )
        print(f"Leídas {len(df_sp500)} filas del S&P 500.")

        # Validar columnas necesarias (MODIFICADO)
        # Buscamos 'Fecha' y 'Precio Ajustado'
        if 'Fecha' not in df_sp500.columns or 'Precio Ajustado' not in df_sp500.columns:
            # Mensaje de error actualizado
            print("Error: El archivo S&P 500 debe contener las columnas 'Fecha' y 'Precio Ajustado'.")
            return

        # Convertir Fecha a datetime (formato YYYY-MM-DD) (MODIFICADO)
        df_sp500['Fecha'] = pd.to_datetime(df_sp500['Fecha'], format='%Y-%m-%d', errors='coerce') # Cambiado el formato

        # --- YA NO SE NECESITA LIMPIEZA MANUAL DEL PRECIO ---
        # pd.read_csv con decimal=',' debería haberlo leído correctamente como float.
        # # Limpiar y convertir columna 'Último'
        # df_sp500['SP500_Close'] = df_sp500['Último'].apply(limpiar_precio_sp500)

        # Renombrar 'Precio Ajustado' a 'SP500_Close' para consistencia interna (MODIFICADO)
        df_sp500.rename(columns={'Precio Ajustado': 'SP500_Close'}, inplace=True)

        # Eliminar filas donde la fecha o el precio (ahora SP500_Close) no se pudieron convertir
        original_len = len(df_sp500)
        # Asegurarse que SP500_Close sea numérico por si acaso
        df_sp500['SP500_Close'] = pd.to_numeric(df_sp500['SP500_Close'], errors='coerce')
        df_sp500.dropna(subset=['Fecha', 'SP500_Close'], inplace=True)
        if len(df_sp500) < original_len:
            print(f"Advertencia: Se eliminaron {original_len - len(df_sp500)} filas del S&P 500 debido a errores de formato en Fecha o Precio Ajustado.")

        # Seleccionar las columnas necesarias ('Fecha' y la renombrada 'SP500_Close')
        # Esta parte no necesita cambio porque ya hemos renombrado la columna
        df_sp500 = df_sp500[['Fecha', 'SP500_Close']].copy()

        # Ordenar por fecha
        df_sp500.sort_values(by='Fecha', inplace=True)

        # Eliminar duplicados de fecha
        df_sp500.drop_duplicates(subset='Fecha', keep='last', inplace=True)

        print("Archivo S&P 500 procesado.")

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo S&P 500 en la ruta: {sp500_path}")
        return
    except Exception as e:
        # Imprimir un error más detallado puede ayudar a depurar
        import traceback
        print(f"Error inesperado al leer o procesar el archivo S&P 500:")
        print(traceback.format_exc()) # Muestra la traza completa del error
        return

    # --- Leer y procesar el otro archivo CSV (SIN CAMBIOS) ---
    # Esta parte asume que el 'otro_path' sigue teniendo el formato esperado
    try:
        print("Leyendo el otro archivo CSV...")
        df_otro = pd.read_csv(
            otro_path,
            sep=';',
            decimal=','
        )
        print(f"Leídas {len(df_otro)} filas del otro archivo.")
        if 'Fecha' not in df_otro.columns:
             print("Error: El otro archivo CSV debe contener la columna 'Fecha'.")
             return
        df_otro['Fecha'] = pd.to_datetime(df_otro['Fecha'], format='%Y-%m-%d', errors='coerce')
        original_len_otro = len(df_otro)
        df_otro.dropna(subset=['Fecha'], inplace=True)
        if len(df_otro) < original_len_otro:
             print(f"Advertencia: Se eliminaron {original_len_otro - len(df_otro)} filas del otro archivo debido a errores en el formato de Fecha.")
        df_otro.sort_values(by='Fecha', inplace=True)
        print("Otro archivo CSV procesado.")
    except FileNotFoundError:
        print(f"Error: No se encontró el otro archivo CSV en la ruta: {otro_path}")
        return
    except Exception as e:
        import traceback
        print(f"Error inesperado al leer o procesar el otro archivo CSV:")
        print(traceback.format_exc())
        return

    # --- Unir, Rellenar y Guardar (SIN CAMBIOS) ---
    print("Uniendo los archivos por fecha...")
    df_final = pd.merge(df_otro, df_sp500, on='Fecha', how='left')
    print("Unión completada.")

    fechas_sin_match_directo = df_final[df_final['SP500_Close'].isna()]['Fecha'].dt.strftime('%Y-%m-%d').tolist()
    if fechas_sin_match_directo:
        print("\n--- Fechas en el segundo archivo SIN coincidencia directa en S&P 500 ---")
        for i in range(0, len(fechas_sin_match_directo), 5):
             print(", ".join(fechas_sin_match_directo[i:i+5]))
        print("Se rellenarán con el valor anterior disponible del S&P 500.\n")
    else:
        print("\nTodas las fechas del segundo archivo encontraron una coincidencia directa en el S&P 500.\n")

    df_final.sort_values(by='Fecha', inplace=True)
    print("Aplicando forward fill para rellenar valores faltantes de S&P 500...")
    df_final['SP500_Close'].fillna(method='ffill', inplace=True)
    nan_iniciales = df_final['SP500_Close'].isna().sum()
    if nan_iniciales > 0:
        print(f"ADVERTENCIA: {nan_iniciales} fila(s) al inicio del archivo no pudieron ser rellenadas (no hay datos anteriores de S&P 500).")
    print("Relleno completado.")

    df_final['Fecha'] = df_final['Fecha'].dt.strftime('%Y-%m-%d')
    base_sp500 = os.path.splitext(os.path.basename(sp500_path))[0]
    base_otro = os.path.splitext(os.path.basename(otro_path))[0]
    output_filename = f"{base_otro}_con_{base_sp500}.csv"
    print(f"Guardando resultado en: {output_filename}")
    try:
        df_final.to_csv(
            output_filename,
            sep=';',
            decimal=',',
            index=False,
            encoding='utf-8-sig'
        )
        print("--- ¡Proceso completado con éxito! ---")
    except Exception as e:
        print(f"Error al guardar el archivo de salida: {e}")

# --- Ejecución del Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Une un CSV con datos financieros con los precios de cierre del S&P 500 basados en la fecha.")
    # Actualizar ayuda para reflejar flexibilidad o el nuevo formato esperado
    parser.add_argument("sp500_file", help="Ruta al archivo CSV del S&P 500 (formato esperado: 'Fecha';'Precio Ajustado' con fecha YYYY-MM-DD y decimal con coma).")
    parser.add_argument("other_file", help="Ruta al otro archivo CSV (con columna 'Fecha' (YYYY-MM-DD) y separador ';').")
    args = parser.parse_args()
    unir_csv_con_sp500(args.sp500_file, args.other_file)