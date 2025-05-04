import pandas as pd
import argparse
import os

def seleccionar_columnas_isin(input_path, output_path, num_isins):
    """
    Lee un archivo CSV (con columna 'Fecha' y columnas ISIN), selecciona
    la columna 'Fecha' y las primeras 'num_isins' columnas ISIN,
    y guarda el resultado en un nuevo archivo CSV.
    """
    if num_isins <= 0:
        print("Error: El número de ISINs a seleccionar debe ser mayor que 0.")
        return

    print(f"Leyendo archivo de entrada: {input_path}")
    try:
        # Asumir formato consistente con salida de scripts anteriores: sep=';', decimal=','
        # Leer la primera línea para obtener cabeceras y detectar formato si es necesario
        try:
             # Intentar con formato esperado primero
             df = pd.read_csv(input_path, sep=';', decimal=',', header=0, nrows=0)
             sep=';'
             dec=','
        except:
             try: # Intentar con coma/punto
                 df = pd.read_csv(input_path, sep=',', decimal='.', header=0, nrows=0)
                 sep=','
                 dec='.'
                 print("Detectado formato CSV con separador ',' y decimal '.'.")
             except: # Fallback genérico
                 df = pd.read_csv(input_path, header=0, nrows=0)
                 sep=',' # Asumir coma si todo falla
                 dec='.'
                 print("Advertencia: No se pudo determinar claramente el formato CSV. Asumiendo sep=',' y decimal='.'")


        # Leer el archivo completo con el formato detectado/asumido
        df = pd.read_csv(input_path, sep=sep, decimal=dec, header=0)

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de entrada: {input_path}")
        return
    except Exception as e:
        print(f"Error al leer el archivo de entrada: {e}")
        return

    if df.empty:
        print("Error: El archivo de entrada está vacío o no se pudo leer correctamente.")
        return

    # Verificar que la primera columna sea 'Fecha' (o similar) - opcional pero buena práctica
    fecha_col_name = df.columns[0]
    if not ('fecha' in fecha_col_name.lower()):
         print(f"Advertencia: La primera columna se llama '{fecha_col_name}', no 'Fecha'. Se asumirá que es la columna de fechas.")


    # Calcular cuántas columnas ISIN hay disponibles
    available_isins = len(df.columns) - 1 # Restar 1 por la columna de Fecha

    if available_isins < num_isins:
        print(f"Error: Solicitaste seleccionar {num_isins} columnas ISIN, pero solo hay {available_isins} disponibles en el archivo.")
        print(f"Columnas ISIN disponibles: {list(df.columns[1:])}")
        return
    elif available_isins == 0:
         print(f"Error: El archivo no contiene columnas de ISIN (solo la columna '{fecha_col_name}').")
         return


    # Seleccionar las columnas deseadas
    # La primera columna (Fecha) + las siguientes 'num_isins' columnas
    cols_to_keep_indices = [0] + list(range(1, num_isins + 1))
    cols_to_keep_names = df.columns[cols_to_keep_indices].tolist()

    print(f"Seleccionando columnas: {cols_to_keep_names}")
    df_selected = df[cols_to_keep_names]

    print(f"Guardando archivo de salida con {num_isins} ISINs: {output_path}")
    try:
        # Guardar en CSV manteniendo el formato (sep=';', decimal=',')
        df_selected.to_csv(
            output_path,
            sep=';',
            decimal=',',
            index=False, # No escribir el índice de pandas
            encoding='utf-8-sig' # Buena compatibilidad con Excel
        )
        print("¡Archivo procesado y guardado con éxito!")
    except Exception as e:
        print(f"Error al guardar el archivo de salida: {e}")

# --- Ejecución del Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Selecciona la columna 'Fecha' y un número específico de las primeras columnas ISIN de un CSV.")
    parser.add_argument("input_file", help="Ruta al archivo CSV de entrada (con columna 'Fecha' y varios ISINs).")
    parser.add_argument("output_file", help="Ruta donde guardar el nuevo archivo CSV con los ISINs seleccionados.")
    parser.add_argument("num_isins", type=int, help="Número de las primeras columnas ISIN a seleccionar (después de la columna 'Fecha').")

    args = parser.parse_args()

    seleccionar_columnas_isin(args.input_file, args.output_file, args.num_isins)