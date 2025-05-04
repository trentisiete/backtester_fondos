import pandas as pd
import holidays
# import datetime # Ya no es estrictamente necesario con pd.Timestamp
import argparse
import os

def detectar_parametros_lectura(filepath):
    """Intenta detectar separador y decimal para CSV."""
    if not filepath.lower().endswith('.csv'):
        return None, None, None # No aplica para Excel

    encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1']
    separators_to_try = [';', ',']
    decimals_to_try = [',', '.']

    # --- (Código de detectar_parametros_lectura sin cambios) ---
    for enc in encodings_to_try:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                lines = [f.readline() for _ in range(min(5, os.path.getsize(filepath)))]
                if not lines: return ';', ',', 'utf-8'
                best_sep, best_dec = None, None
                max_sep_count = -1
                for sep in separators_to_try:
                    for dec in decimals_to_try:
                        if sep == dec: continue
                        sep_count = sum(line.count(sep) for line in lines)
                        dec_count = sum(line.count(dec) for line in lines)
                        if sep_count > dec_count and sep_count > 0 and sep_count > max_sep_count:
                            try:
                                pd.read_csv(filepath, sep=sep, decimal=dec, nrows=1, encoding=enc)
                                best_sep, best_dec = sep, dec
                                max_sep_count = sep_count
                            except Exception:
                                continue
                if best_sep and best_dec:
                    print(f"Detectado: sep='{best_sep}', decimal='{best_dec}', encoding='{enc}'")
                    return best_sep, best_dec, enc
                try:
                    pd.read_csv(filepath, sep=';', decimal=',', nrows=1, encoding=enc)
                    print(f"Detectado (fallback europeo): sep=';', decimal=',', encoding='{enc}'")
                    return ';', ',', enc
                except Exception: pass
                try:
                    pd.read_csv(filepath, sep=',', decimal='.', nrows=1, encoding=enc)
                    print(f"Detectado (fallback anglosajón): sep=',', decimal='.', encoding='{enc}'")
                    return ',', '.', enc
                except Exception: pass
        except UnicodeDecodeError: continue
        except Exception as e:
             print(f"Error inesperado durante detección con encoding '{enc}': {e}")
             continue
    print("Advertencia: No se pudo detectar automáticamente el formato del CSV. Usando ';' y ',' por defecto.")
    return ';', ',', 'utf-8'


def get_most_recent_business_day(country_code='ES'):
    """Encuentra el día laborable más reciente (hoy o anterior) para un país."""
    # --- (Código de get_most_recent_business_day sin cambios) ---
    today = pd.Timestamp('today').normalize()
    try:
        country_holidays = holidays.country_holidays(country_code, years=[today.year -1, today.year], observed=False)
    except KeyError:
         print(f"Advertencia: Código de país '{country_code}' no reconocido por holidays. No se excluirán festivos.")
         country_holidays = {}
    except Exception as e:
        print(f"Advertencia: No se pudieron obtener festivos para {country_code}: {e}. No se excluirán festivos.")
        country_holidays = {}
    current_date = today
    while True:
        if current_date.dayofweek >= 5 or current_date in country_holidays:
            current_date -= pd.Timedelta(days=1)
        else:
            return current_date.tz_localize(None) if current_date.tz else current_date

# --- MODIFICACIÓN EN ESTA FUNCIÓN ---
def añadir_columna_fecha(input_path, output_path):
    """
    Lee un archivo CSV o Excel, añade una columna 'Fecha' con días laborables
    (excluyendo fines de semana y festivos de España), asignando la fecha laborable
    MÁS ANTIGUA a la PRIMERA fila y la MÁS RECIENTE a la ÚLTIMA fila.
    Guarda un nuevo CSV.
    """
    print(f"Leyendo archivo de entrada: {input_path}")

    # --- (Código de lectura de archivo sin cambios) ---
    try:
        if input_path.lower().endswith('.csv'):
            sep, dec, encoding = detectar_parametros_lectura(input_path)
            if sep is None:
                 print("Error: No se pudieron detectar los parámetros de lectura para el CSV.")
                 return
            encoding = encoding if encoding else 'utf-8'
            df = pd.read_csv(input_path, sep=sep, decimal=dec, header=0, encoding=encoding)
        elif input_path.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(input_path, header=0)
        else:
            print("Error: Formato de archivo no soportado. Usa .csv, .xlsx o .xls.")
            return
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de entrada: {input_path}")
        return
    except Exception as e:
        print(f"Error al leer el archivo de entrada: {e}")
        return

    if df.empty:
        print("Advertencia: El archivo de entrada está vacío. Se creará un archivo de salida vacío.")
        df_final = pd.DataFrame(columns=['Fecha'])
        try:
             df_final.to_csv(output_path, sep=';', decimal=',', index=False, encoding='utf-8-sig')
             print("Archivo de salida vacío creado con éxito.")
        except Exception as e:
             print(f"Error al guardar el archivo de salida vacío: {e}")
        return

    num_rows = df.shape[0]
    print(f"Archivo leído. Número de filas de datos detectadas: {num_rows}")

    # --- Lógica de Fechas ---
    # 1. Determinar la fecha MÁS RECIENTE (irá en la ÚLTIMA fila)
    most_recent_date = get_most_recent_business_day('ES')
    print(f"El día laborable más reciente (asignado a la última fila) es: {most_recent_date.strftime('%Y-%m-%d')}")

    # 2. Determinar rango de años para obtener festivos (sin cambios)
    estimated_days_back = int((num_rows -1) * 1.5) if num_rows > 0 else 0
    estimated_oldest_date = most_recent_date - pd.Timedelta(days=estimated_days_back)
    start_year = estimated_oldest_date.year
    end_year = most_recent_date.year
    years_needed = range(start_year, end_year + 1)

    # 3. Obtener festivos nacionales de España para los años necesarios (sin cambios)
    try:
        es_holidays = holidays.country_holidays('ES', years=years_needed, observed=False)
        print(f"Festivos de España obtenidos para los años: {list(years_needed)}")
    except Exception as e:
        print(f"Advertencia: Error al obtener los festivos: {e}. Se continuará sin excluir festivos.")
        es_holidays = {}

    # 4. Generar rango de fechas laborables TERMINANDO en most_recent_date
    try:
        # Generamos las N fechas laborables que acaban en la fecha más reciente
        # bdate_range las devuelve ordenadas de MÁS ANTIGUA a MÁS RECIENTE.
        # Este es el orden que queremos asignar directamente.
        business_dates = pd.bdate_range(
            end=most_recent_date,
            periods=num_rows,
            freq='C', # Custom business day frequency
            holidays=list(es_holidays.keys())
        )
        print(f"Generado rango de fechas desde {business_dates.min().strftime('%Y-%m-%d')} hasta {business_dates.max().strftime('%Y-%m-%d')}.")

        # --- YA NO INVERTIMOS LA LISTA ---
        # La línea business_dates_descending = business_dates_ascending[::-1] ha sido eliminada.
        print("Fechas ordenadas: Más antigua primero (asignada a la primera fila).") # Mensaje actualizado


    except Exception as e:
        print(f"Error al generar el rango de fechas: {e}")
        return

    # Comprobamos la longitud de la lista original (ascendente)
    if len(business_dates) != num_rows:
          print(f"Advertencia: Se generaron {len(business_dates)} fechas pero se esperaban {num_rows}. Revisa la lógica o los datos.")
          print("Error: Discrepancia en número de fechas generadas y filas del archivo.")
          return

    # Crear DataFrame de fechas (en orden ascendente: antigua primero)
    # Usamos directamente business_dates (que está ordenada ascendentemente)
    df_dates = pd.DataFrame({'Fecha': business_dates})

    # Añadir la columna 'Fecha' al principio del DataFrame original (sin cambios)
    df.reset_index(drop=True, inplace=True)
    df_dates.reset_index(drop=True, inplace=True)
    df_final = pd.concat([df_dates, df], axis=1)

    # Formatear la columna de fecha a YYYY-MM-DD para el CSV de salida (sin cambios)
    df_final['Fecha'] = df_final['Fecha'].dt.strftime('%Y-%m-%d')

    print(f"Guardando archivo de salida con fechas: {output_path}")
    # --- (Código de guardar archivo sin cambios) ---
    try:
        df_final.to_csv(
            output_path,
            sep=';',
            decimal=',',
            index=False,
            encoding='utf-8-sig'
        )
        print("¡Archivo procesado y guardado con éxito!")
    except Exception as e:
        print(f"Error al guardar el archivo de salida: {e}")


# --- Ejecución del Script ---
if __name__ == "__main__":
    # Actualizar descripción para reflejar el orden de fechas
    parser = argparse.ArgumentParser(description="Añade una columna 'Fecha' con días laborables (L-V, sin festivos ES) a un archivo CSV/Excel. La PRIMERA fila corresponde a la fecha MÁS ANTIGUA del periodo generado y la última a la más reciente.")
    parser.add_argument("input_file", help="Ruta al archivo CSV o Excel de entrada (sin fechas).")
    parser.add_argument("output_file", help="Ruta donde guardar el nuevo archivo CSV con fechas.")
    args = parser.parse_args()
    añadir_columna_fecha(args.input_file, args.output_file)