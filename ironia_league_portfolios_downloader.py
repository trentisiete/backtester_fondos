#!/usr/bin/env python3
"""
Script para extraer información de carteras y fondos (tooltips de Highcharts)
desde la competición de Ironia Tech y volcar los resultados en un CSV.
Intenta extraer nombres de fondos del tooltip, sospechando que no están en el texto.

Requisitos:
    pip install playwright
    playwright install chromium

Uso:
    1. Asegúrate de tener 'auth.json'.
    2. Ejecuta: python nombre_del_script.py
    3. *** OBSERVA MANUALMENTE EL TOOLTIP EN EL NAVEGADOR ABIERTO ***
       Usa 'Inspeccionar Elemento' para ver la estructura HTML del tooltip
       y del segmento de barra (<rect>, <path>) sobre el que se hace hover.
       Busca dónde podría estar el nombre del fondo (texto, atributo title/aria-label/data-*).
"""
import asyncio
import csv
import re
import json # Para manejar mejor la salida de JS
from pathlib import Path
from playwright.async_api import async_playwright, Error as PlaywrightError, TimeoutError as PlaywrightTimeoutError

# --- Configuración ---
STATE_FILE = Path("auth.json")
PROMO_URL = "https://store.ironia.tech/competitions-promotioned"
OUTPUT_CSV = Path("carteras_fondos.csv")
OUTPUT_HTML = Path("pagina_competicion.html")
# *** MANTENER DELAY ALTO Y SLOW_MO PARA INSPECCIÓN MANUAL ***
HOVER_DELAY_S = 1.0 # Aumentado para dar tiempo a inspeccionar
COMPETITION_INDEX = 2
INTERACTIVE_POINT_SELECTOR = 'g.highcharts-tracker rect, g.highcharts-tracker path, .highcharts-point'
# *** PONER headless=False ES OBLIGATORIO PARA INSPECCIÓN MANUAL ***
HEADLESS_MODE = False
SLOW_MOTION_MS = 150 # Ralentizar para facilitar la inspección

async def fetch_and_extract_funds_to_csv():
    extracted_data = []

    try:
        async with async_playwright() as p:
            print("🚀 Iniciando Playwright...")
            browser = await p.chromium.launch(headless=HEADLESS_MODE, slow_mo=SLOW_MOTION_MS)

            # --- Gestión de autenticación ---
            context = None
            if STATE_FILE.exists():
                print(f"ℹ️ Cargando estado de autenticación desde: {STATE_FILE}")
                try:
                    context = await browser.new_context(storage_state=str(STATE_FILE))
                    print("✅ Estado de autenticación cargado correctamente")
                except Exception as auth_error:
                    print(f"⚠️ Error al cargar el estado de autenticación: {auth_error}")
                    context = None

            # Si no hay archivo de autenticación o hubo error al cargarlo, iniciar proceso de login manual
            if not context:
                print("🔑 Archivo de autenticación no encontrado o inválido. Iniciando proceso de login manual...")

                # Credenciales - puedes definirlas aquí o en una configuración externa
                email = "ALVARO.MARTINEZGAMO@ESTUDIANTE.UAM.ES"
                password = "JA69@#123web"

                # Crear contexto para el login
                context = await browser.new_context()
                login_page = await context.new_page()

                # Ir a la página de login
                print("➡️ Navegando a la página de login...")
                await login_page.goto("https://ui.id.ironia.tech/login/login_input?next=https%3A%2F%2Fauth.id.ironia.tech%2Foauth2%2Fauthorize%3Fresponse_type%3Dcode%26client_id%3D699239840089865%26redirect_uri%3Dhttps%253A%252F%252Fproxy.ironia.tech%252Foauth%252Fcallback%26x_method%3Dlogin",
                              wait_until="networkidle", timeout=60000)

                # Esperar a que la página cargue
                await login_page.wait_for_load_state("domcontentloaded")
                await asyncio.sleep(2)

                # Buscar el campo de email
                print("🔍 Buscando campo de email...")
                email_input = await login_page.query_selector("#identifier") or await login_page.query_selector("input[name='identifier']")

                # Si no encontramos el campo con selector específico, buscar alternativas
                if not email_input:
                    print("⚠️ Campo de email no encontrado con selector específico, buscando alternativas...")
                    email_input = await login_page.query_selector("input[type='text'], input[type='email']")

                # Buscar el campo de contraseña
                print("🔍 Buscando campo de contraseña...")
                password_input = await login_page.query_selector("#contact_password") or await login_page.query_selector("input[name='password']")

                # Si no encontramos el campo con selector específico, buscar alternativas
                if not password_input:
                    print("⚠️ Campo de contraseña no encontrado con selector específico, buscando alternativas...")
                    password_input = await login_page.query_selector("input[type='password']")

                # Verificar que se encontraron los campos
                if not email_input or not password_input:
                    print("❌ No se pudieron encontrar los campos de login. Abortando.")
                    return

                # Ingresar credenciales
                print("✍️ Ingresando credenciales...")
                await email_input.click()
                await email_input.fill("")  # Limpiar primero
                await email_input.type(email, delay=50)  # Escribir caracter por caracter

                await password_input.click()
                await password_input.fill("")  # Limpiar primero
                await password_input.type(password, delay=50)  # Escribir caracter por caracter

                # Buscar botón de login
                print("🔍 Buscando botón de login...")
                button_selectors = [
                    "button[type='submit']",
                    "input[type='submit']",
                    "button:has-text('Entrar')",
                    "button:has-text('Login')",
                    "button:has-text('Iniciar')",
                    "button:has-text('Acceder')",
                    "button.login",
                    "button.submit",
                    "input.login",
                    "input.submit"
                ]

                login_button = None
                for selector in button_selectors:
                    login_button = await login_page.query_selector(selector)
                    if login_button:
                        print(f"✅ Botón de login encontrado con selector: {selector}")
                        break

                # Hacer clic en el botón o presionar Enter
                if login_button:
                    print("🖱️ Haciendo clic en el botón de login...")
                    await login_button.click()
                else:
                    print("⚠️ Botón de login no encontrado, usando tecla Enter...")
                    await password_input.press("Enter")

                # Esperar a que se complete el login
                print("⏳ Esperando que se complete el proceso de login...")
                await login_page.wait_for_load_state("networkidle", timeout=30000)
                await asyncio.sleep(3)

                # Verificar si el login fue exitoso
                current_url = login_page.url
                print(f"ℹ️ URL actual después del intento de login: {current_url}")

                if "/login" in current_url:
                    print("❌ ERROR: Todavía en la página de login después del intento")
                    error_messages = await login_page.query_selector_all("text=error, text=inválid, .error, .alert")
                    for error in error_messages:
                        error_text = await error.inner_text()
                        print(f"❌ Mensaje de error encontrado: {error_text}")
                    return
                else:
                    print("✅ Login completado con éxito")

                # Guardar el estado de autenticación para futuros usos
                print("💾 Guardando estado de autenticación...")
                await context.storage_state(path=str(STATE_FILE))
                print(f"✅ Estado guardado en: {STATE_FILE}")

                # Cerrar página de login ya que usaremos una nueva para la navegación principal
                await login_page.close()

            # --- A partir de aquí, el código original sigue igual ---
            print(f"➡️ Navegando a: {PROMO_URL}")
            page = await context.new_page()

            # --- Navegación y esperas ---
            await page.goto(PROMO_URL, wait_until="networkidle", timeout=60000)
            await page.wait_for_selector("article.rounded", timeout=30000)
            competition_selector = f"article.rounded:nth-of-type({COMPETITION_INDEX}) button:has-text('Ver')"
            await page.locator(competition_selector).click()
            await page.wait_for_load_state("networkidle", timeout=90000)
            await page.wait_for_function("() => document.querySelectorAll('table tbody tr').length > 0", timeout=60000)
            await page.wait_for_selector(f"table td:nth-child(4) {INTERACTIVE_POINT_SELECTOR}", timeout=60000)
            print("✅ Tabla y elementos de gráfico encontrados.")

            # --- Guardar HTML --- POR AHORA NO NECESARIO
            # try:
                # print("📄 Guardando HTML para análisis offline...")
                # html_content = await page.content()
                # OUTPUT_HTML.write_text(html_content, encoding='utf-8')
                # print(f"📄 HTML guardado en: {OUTPUT_HTML.resolve()}")
                # print(f"‼️ Revisa este HTML si el script falla. Selector actual: '{INTERACTIVE_POINT_SELECTOR}'")
            # except Exception as html_err:
                # print(f"⚠️ No se pudo guardar el HTML: {html_err}")

            # --- Resto del código original ---
            rows_locator = page.locator("table tbody tr")
            count = await rows_locator.count()
            print(f"📊 Encontradas {count} filas. Procesando...")
            print("====== ¡¡ATENCIÓN!! ======")
            print("El navegador se está ejecutando. Cuando el script haga 'hover' sobre")
            print("un segmento de gráfico, el tooltip aparecerá.")
            print("USA EL INSPECTOR DE ELEMENTOS DEL NAVEGADOR (Click derecho -> Inspeccionar)")
            print("SOBRE ESE TOOLTIP para ver su estructura HTML y dónde está el nombre.")
            print("Busca texto oculto, atributos 'title', 'aria-label', 'data-*', etc.")
            print("===========================")
            await asyncio.sleep(5) # Pausa para leer el mensaje


            for i in range(count):
                row = rows_locator.nth(i)
                print(f"--- Procesando fila {i+1}/{count} ---")
                sharpe = await row.locator("td:nth-child(2)").inner_text()
                cartera = await row.locator("td:nth-child(3)").inner_text()
                print(f"  -> Cartera: '{cartera.strip()}', Sharpe: '{sharpe.strip()}'")

                chart_cell_locator = row.locator("td:nth-child(4)")
                interactive_points_locator = chart_cell_locator.locator(INTERACTIVE_POINT_SELECTOR)
                points_count = await interactive_points_locator.count()
                print(f"  -> Encontrados {points_count} puntos/segmentos interactivos.")

                fondos_de_esta_fila = set()

                if points_count > 0:
                    for j in range(points_count):
                        point_locator = interactive_points_locator.nth(j)
                        print(f"    -> Procesando punto/segmento {j+1}/{points_count}...")
                        # *** PON UN BREAKPOINT AQUÍ EN TU DEBUGGER SI QUIERES PAUSAR ANTES DEL HOVER ***
                        try:
                            await point_locator.scroll_into_view_if_needed(timeout=5000)
                            print(f"      -> Hovering... OBSERVA EL NAVEGADOR AHORA.")
                            await point_locator.hover(timeout=5000, force=True) # force=True como último recurso
                            print(f"      -> Esperando {HOVER_DELAY_S}s para que aparezca tooltip...")
                            await asyncio.sleep(HOVER_DELAY_S)

                            # *** Nueva estrategia de extracción JS: Buscar en texto Y atributos ***
                            tooltip_data = await page.evaluate(r"""
                                () => {
                                    const tooltip = document.querySelector('.highcharts-label.highcharts-tooltip:not([visibility="hidden"])');
                                    if (!tooltip) return null;

                                    const data = {
                                        innerText: tooltip.innerText || tooltip.textContent || '',
                                        innerHTML: tooltip.innerHTML || '', // Para inspección posterior si es necesario
                                        attributes: {}
                                    };

                                    // Intentar buscar atributos comunes en el tooltip o sus hijos inmediatos
                                    const elementsToSearch = [tooltip, ...tooltip.querySelectorAll('tspan, text, span, div, strong')];
                                    for (const el of elementsToSearch) {
                                         const title = el.getAttribute('title');
                                         const ariaLabel = el.getAttribute('aria-label');
                                         if (title && !data.attributes.title) data.attributes.title = title;
                                         if (ariaLabel && !data.attributes.ariaLabel) data.attributes.ariaLabel = ariaLabel;
                                         // Añadir más data-* si los ves en la inspección manual
                                    }

                                    return data;
                                }
                            """)

                            print(f"      -> Raw Tooltip Data Extracted (JS):")
                            # Usar json.dumps para una visualización más clara del objeto
                            print(json.dumps(tooltip_data, indent=2, ensure_ascii=False))

                            possible_names = set()

                            if tooltip_data:
                                # 1. Buscar en atributos extraídos
                                for key, value in tooltip_data.get('attributes', {}).items():
                                    if value and isinstance(value, str) and len(value) > 1:
                                         print(f"        -> Found in attribute '{key}': '{value}'")
                                         possible_names.add(value.strip())

                                # 2. Procesar innerText como antes, por si acaso
                                text_content = tooltip_data.get('innerText', '')
                                if text_content:
                                    lines = text_content.split('\n')
                                    for line in lines:
                                        line = line.strip()
                                        if not line: continue

                                        match = re.match(r"^\s*(.*?)\s*(?::\s*[\d.,]+\s*%?)?\s*$", line)
                                        nombre_fondo = ""
                                        if match:
                                            nombre_fondo = match.group(1).strip()
                                            nombre_fondo = re.sub(r"^\s*●\s*", "", nombre_fondo).strip()
                                        else:
                                            nombre_fondo = re.sub(r"^\s*●\s*", "", line).strip()

                                        # Validar (ignorar solo números/% y bala)
                                        if nombre_fondo and nombre_fondo != '●' and not re.fullmatch(r"[\d.,]+\s*%?", nombre_fondo):
                                             print(f"        -> Found in innerText line: '{nombre_fondo}' (from line: '{line}')")
                                             possible_names.add(nombre_fondo)

                            if not possible_names:
                                print(f"      -> WARN: No se pudo identificar un nombre de fondo ni en atributos ni en texto.")
                            else:
                                fondos_de_esta_fila.update(possible_names) # Añadir todos los nombres encontrados a la colección de la fila


                        except PlaywrightTimeoutError:
                             print(f"      -> TIMEOUT al interactuar con el punto {j+1}.")
                        except Exception as point_err:
                            print(f"      -> ERROR durante procesamiento del punto {j+1}: {point_err}")
                            # Considerar añadir un pequeño sleep aquí si hay errores consecutivos rápidos
                            await asyncio.sleep(0.2)


                    # --- Mouse move out ---
                    try:
                        # Mover a un punto neutral para intentar ocultar tooltip
                        chart_bb = await chart_cell_locator.bounding_box()
                        if chart_bb:
                             await page.mouse.move(chart_bb['x'] - 10, chart_bb['y'] - 10) # Fuera y a la izquierda/arriba
                        else:
                             await page.mouse.move(0,0)
                        await asyncio.sleep(0.1)
                    except Exception: pass

                else: # No se encontraron puntos interactivos
                    print("  -> WARN: No se encontraron puntos interactivos. Verifique HTML y selector.")
                    fondos_de_esta_fila.add('ERROR: No se encontraron puntos interactivos')

                # --- Procesamiento final de la fila ---
                fondos_finales = sorted(list(fondos_de_esta_fila))
                print(f"  -> Fondos finales extraídos para la fila: {fondos_finales}")
                extracted_data.append({
                    'sharpe': sharpe.strip(),
                    'cartera': cartera.strip(),
                    'fondos': fondos_finales
                })
                print(f"--- Fin fila {i+1}/{count} ---")
                # Pausa opcional entre filas si es necesario para observar/evitar bloqueos
                # await asyncio.sleep(0.5)


            # 6) Volcar a CSV
            print(f"\n💾 Guardando datos en: {OUTPUT_CSV.resolve()}")
            # ... (CSV writing logic remains the same) ...
            with OUTPUT_CSV.open('w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['Sharpe', 'Cartera', 'Fondos'])
                for item in extracted_data:
                    fondos_str = ';'.join(item['fondos'])
                    writer.writerow([item['sharpe'], item['cartera'], fondos_str])

            print(f"✅ ¡Proceso completado! CSV generado en: {OUTPUT_CSV.resolve()}")
            print("🛑 Presiona Enter en la consola para cerrar el navegador...")
            input() # Espera a que el usuario presione Enter antes de cerrar
            await browser.close()
            print("🔒 Navegador cerrado.")

    # --- Error Handling ---
    except PlaywrightTimeoutError as e:
        print(f"❌ TIMEOUT general esperando algún elemento o navegación: {e}")
    except PlaywrightError as e:
        print(f"❌ Error fatal de Playwright: {e}")
    except Exception as e:
        print(f"❌ Ocurrió un error inesperado: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("--- Script finalizado ---")


if __name__ == '__main__':
    print("--- Iniciando script de extracción de fondos ---")
    asyncio.run(fetch_and_extract_funds_to_csv())