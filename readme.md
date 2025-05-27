# 🚀 Backtester Quant v5.0 (Extendido)

¡Bienvenido a la versión 5.0 de tu **Backtester Quant**! Esta potente herramienta te permite **analizar, optimizar y gestionar el riesgo de tus carteras de inversión**. Simula el rendimiento de tus estrategias con datos históricos, realiza análisis de riesgo avanzados, optimiza la asignación de activos y proyecta escenarios futuros.

Esta versión introduce una funcionalidad vital para la gestión activa de carteras: la capacidad de definir **pesos dinámicos que cambian a lo largo del tiempo**, permitiendo simular estrategias de "timing" o de ajuste de asignación de activos.

---

## 🌟 Características Principales

* **Carga de Datos Flexible**: Importa tus precios históricos de fondos y benchmarks desde archivos CSV.
* **Gestión de Pesos de Cartera**:
    * **Manual Fija**: Asigna pesos fijos a tus activos para todo el período de backtest.
    * **Dinámica por Archivo (¡NUEVO!)**: Define cómo cambian los pesos de tu cartera en fechas específicas a lo largo del tiempo, simulando una gestión activa.
* **Backtesting Robusto**: Simula el rendimiento de tu cartera con opciones de rebalanceo periódico y costes de transacción.
* **Análisis de Rendimiento Detallado**: Calcula métricas clave como **Rentabilidad Total**, **Rentabilidad Anualizada (CAGR)**, **Volatilidad**, **Ratios de Sharpe y Sortino**, **Máximo Drawdown**, y más.
* **Comparativa con Benchmark**: Evalúa el desempeño de tu cartera frente a un índice de referencia con métricas como **Alpha**, **Beta**, **Tracking Error** e **Information Ratio**.
* **Análisis Rodante**: Visualiza la evolución del riesgo y la rentabilidad de tu cartera a lo largo del tiempo con ventanas móviles.
* **Análisis de Correlación**: Entiende la relación entre tus activos con matrices de correlación y correlaciones rodantes.
* **Contribución al Riesgo**: Identifica cómo cada activo contribuye a la volatilidad total de tu cartera, revelando el poder de la diversificación.
* **Análisis de Riesgo Avanzado (VaR y ES)**: Cuantifica las pérdidas potenciales máximas de tu cartera utilizando métodos históricos y paramétricos.
* **Pruebas de Estrés y Escenarios Históricos**: Evalúa la resiliencia de tu cartera ante shocks de mercado hipotéticos o durante periodos de crisis pasadas.
* **Simulación de Montecarlo**: Proyecta la evolución futura de tu cartera bajo diferentes escenarios probabilísticos.
* **Optimización de Cartera (Frontera Eficiente)**: Encuentra la asignación de pesos óptima para minimizar el riesgo o maximizar el retorno ajustado al riesgo, con la posibilidad de aplicar restricciones a la Renta Fija y al Monetario.
* **Visualizaciones Interactivas**: Gráficos intuitivos y dinámicos para una mejor comprensión de los resultados.

---

## 📚 Explicación Detallada de Funcionalidades y Métricas

### 1. Carga de Datos (Precios Históricos y Benchmark)

La aplicación comienza con la carga de tus datos históricos de precios.

* **Formato Esperado**:
    * Un archivo CSV (o Excel) donde la primera columna sea la **Fecha**.
    * Las columnas siguientes deben ser los **ISINs/Tickers** de tus fondos y el Benchmark que desees analizar.
    * **Separador CSV**: Puede ser coma (`,`) o punto y coma (`;`).
    * **Separador Decimal**: Puede ser coma (`,`) o punto (`.`). La aplicación intentará detectarlos automáticamente.
* **Importancia**: La calidad y el rango de tus datos históricos son cruciales. Para análisis avanzados (optimización, Montecarlo, escenarios históricos), se recomienda encarecidamente cargar datos con una extensión temporal considerable (varios años).

### 2. Definición de la Cartera y Pesos

Aquí defines cómo se compone tu cartera.

* **Selección de Columna Benchmark**: Elige cuál de las columnas de tu archivo de precios representa el índice de referencia con el que quieres comparar tu cartera (ej., S&P 500, IBEX 35).
* **Activos para Restricciones de Optimización (Renta Fija / Monetario)**: Puedes etiquetar ciertos activos como "Renta Fija" o "Monetarios". Estas etiquetas se utilizarán únicamente en la sección de Optimización para aplicar restricciones (ej., máximo 9% en Renta Fija, máximo 1% en Monetario). **Importante**: Estas restricciones NO se aplican si introduces los pesos manualmente o a través de un archivo de pesos dinámicos; en esos casos, la aplicación simplemente simulará la cartera tal como la definas.
* **Asignación de Pesos (%)**:
    * **Método Manual (Fijos)**:
        * Introduces un porcentaje para cada activo.
        * Los pesos se normalizarán automáticamente para que sumen 100%.
        * Estos pesos se mantendrán fijos durante todo el backtest y la cartera se rebalanceará a ellos según la frecuencia de rebalanceo elegida.
    * **Método por Archivo (Dinámicos) - ¡NUEVO y VITAL!**:
        * Permite que los pesos de tu cartera cambien en fechas específicas a lo largo del tiempo.
        * **Formato del Archivo de Pesos**:
            * Un archivo CSV (o Excel).
            * Primera columna: **Fecha** (ej., AAAA-MM-DD).
            * Columnas siguientes: Los **ISINs/Tickers** de tus activos (deben coincidir exactamente con los nombres de tu archivo de precios).
            * Valores: Los pesos para cada activo en esa fecha (pueden ser decimales que sumen 1, o porcentajes que sumen 100; la aplicación los normalizará).
        * **Ejemplo**:
            ```
            Fecha;FondoA;FondoB;FondoC
            2025-01-01;0,33;0,33;0,34
            2025-01-10;0,10;0,10;0,80
            2025-01-20;0,01;0,09;0,90
            ```
        * **Cómo se aplican los pesos dinámicos**: Los pesos definidos para una fecha específica se aplican a partir de ese día (inclusive) y permanecen activos hasta el día anterior a la siguiente fecha especificada en el archivo. Si no hay más fechas en el archivo, los últimos pesos se mantienen hasta el final del backtest.
        * **Rebalanceo con pesos dinámicos**: Si eliges una frecuencia de rebalanceo (Mensual, Trimestral, Anual), la cartera se rebalanceará a los pesos activos en ese momento en cada fecha de rebalanceo. Si los pesos cambian por una entrada en el archivo, se realiza un ajuste de la cartera a esos nuevos pesos.
        * **Advertencias de Restricciones**: Si los pesos de tu archivo dinámico exceden las restricciones de Renta Fija/Monetario que definiste (9%/1%), la aplicación te mostrará una advertencia, pero no impedirá la simulación.

### 3. Parámetros del Backtest

Controla cómo se simula tu cartera.

* **Fecha de Inicio / Fin**: Define el período de tiempo para tu simulación. Asegúrate de que tus datos de precios cubran este rango.
* **Inversión Inicial (€)**: El capital con el que comienza la simulación.
* **Frecuencia de Rebalanceo**:
    * **No Rebalancear**: Los pesos solo cambian si usas un archivo de pesos dinámicos y hay una nueva fecha en él. Si usas pesos manuales, la cartera se "desvía" de esos pesos a medida que los precios cambian.
    * **Mensual / Trimestral / Anual**: La cartera se ajusta a los pesos objetivo (manuales o los activos del archivo dinámico) en el primer día de cada período de rebalanceo.
* **Coste Transacción (pb por operación)**: Un porcentaje (en puntos básicos, pb) que se deduce del valor de la cartera cada vez que se realiza una operación de compra/venta (al inicio, y en cada rebalanceo/cambio de pesos).

### 4. Parámetros de Análisis

Configuraciones para las métricas y gráficos.

* **Ventana Análisis Rodante (días)**: El número de días que se utilizan para calcular métricas como la volatilidad o la correlación rodante. Una ventana de 60 días, por ejemplo, significa que cada punto en el gráfico rodante representa los últimos 60 días de datos.
* **Tasa Libre de Riesgo Anual (%)**: La tasa de retorno de una inversión "sin riesgo" (ej., bonos del tesoro a corto plazo). Se utiliza para calcular métricas ajustadas al riesgo como el Ratio de Sharpe o Alpha.

### 5. Análisis de Rendimiento y Riesgo (Visión General)

Esta sección te proporciona una visión general del desempeño de tu cartera.

* **Métricas Principales (Cartera Total)**:
    * **Rentabilidad Total**: El porcentaje de crecimiento acumulado de tu cartera desde el inicio hasta el fin del backtest.
        $$Rentabilidad\ Total = \left( \frac{Valor\ Final - Valor\ Inicial}{Valor\ Inicial} \right) \times 100$$
    * **Rentabilidad Anualizada (CAGR - Compound Annual Growth Rate)**: La tasa de crecimiento promedio anual compuesta de tu cartera durante el período del backtest. Es esencial para comparar el rendimiento de inversiones de diferente duración, ya que normaliza el retorno a una base anual.
        $$CAGR = \left[ \left( \frac{Valor\ Final}{Valor\ Inicial} \right)^{\frac{1}{Número\ de\ Años}} - 1 \right] \times 100$$
        Donde "Número de Años" es la duración del período de inversión en años (puede ser decimal).
    * **Volatilidad Anualizada**: Mide la dispersión de los retornos de tu cartera, indicando cuánto fluctúa su valor. Es una medida del riesgo. Se calcula como la desviación estándar de los retornos diarios, anualizada.
        $$Volatilidad\ Anualizada = Desviación\ Estándar\ de\ Retornos\ Diarios \times \sqrt{252}$$
        (Se asumen 252 días hábiles en un año de trading).
    * **Máximo Drawdown**: La mayor caída porcentual desde un pico (valor máximo alcanzado) hasta un valle (valor mínimo posterior) antes de que se alcance un nuevo pico. Mide la peor pérdida potencial que habrías experimentado si hubieras invertido en el pico y vendido en el valle.
        $$Drawdown_t = \frac{Valor\ Cartera_t - Pico\ Anterior_t}{Pico\ Anterior_t}$$       $$Máximo\ Drawdown = \min(Drawdown_t)$$
    * **Ratio de Sharpe**: Mide el exceso de retorno de la cartera por unidad de riesgo (volatilidad). Cuanto mayor, mejor es el retorno ajustado al riesgo.
        $$Ratio\ de\ Sharpe = \frac{Rentabilidad\ Anualizada\ Cartera - Tasa\ Libre\ de\ Riesgo\ Anual}{Volatilidad\ Anualizada\ Cartera}$$
    * **Ratio Sortino**: Similar al Sharpe, pero solo penaliza la volatilidad a la baja (retornos negativos o por debajo de un umbral). Es útil para inversores que solo se preocupan por el riesgo de pérdidas.
        $$Ratio\ Sortino = \frac{Rentabilidad\ Anualizada\ Cartera - Tasa\ Mínima\ Aceptable\ Anual}{Desviación\ a\ la\ Baja\ Anualizada}$$
        La desviación a la baja anualizada se calcula como la desviación estándar de los retornos que caen por debajo de la tasa mínima aceptable, anualizada.
    * **Ratio de Diversificación**: Mide cuánto la diversificación ha reducido la volatilidad de tu cartera. Un valor mayor que 1 indica una buena diversificación, ya que la volatilidad promedio ponderada de los activos individuales es mayor que la volatilidad de la cartera combinada.
        $$Ratio\ de\ Diversificación = \frac{\sum_{i=1}^{N}(w_i \times \sigma_i)}{\sigma_p}$$
        Donde $w_i$ es el peso del activo $i$, $\sigma_i$ es la volatilidad anualizada del activo $i$, y $\sigma_p$ es la volatilidad anualizada de la cartera.
    * **Costes Totales Transacción (€ y %)**: El monto total de dinero gastado en comisiones por las operaciones de compra/venta y rebalanceo, tanto en valor absoluto como en porcentaje de la inversión inicial.
* **Evolución Normalizada (Base 100)**:
    * Gráfico que muestra cómo habría evolucionado una inversión inicial de 100€ (o 100 unidades) en tu cartera, en cada activo individual y en el benchmark.
    * Fórmula:
        $$Valor\ Normalizado_t = \left( \frac{Valor\ Actual_t}{Valor\ Inicial} \right) \times 100$$
    * **Utilidad**: Permite comparar visualmente el rendimiento relativo de diferentes activos y de la cartera total, independientemente de sus valores iniciales. Todas las líneas comienzan en el mismo punto (100).
* **Análisis Rodante de la Cartera**:
    * Gráficos que muestran la evolución de la **Volatilidad**, el **Ratio de Sharpe** y el **Ratio de Sortino** de tu cartera calculados sobre una "ventana móvil" (ej., los últimos 60 días).
    * **Utilidad**: Revela cómo el riesgo y el rendimiento ajustado al riesgo de tu cartera han cambiado a lo largo del tiempo, permitiendo identificar periodos de mayor/menor riesgo o eficiencia.

### 6. Comparativa vs Benchmark

Evalúa el rendimiento de tu cartera en relación con el benchmark seleccionado.

* **Métricas Relativas al Benchmark**:
    * **Beta ($\beta$)**: Mide la sensibilidad de tu cartera a los movimientos del benchmark. Una Beta de 1 significa que la cartera se mueve en línea con el benchmark; >1 es más volátil que el benchmark; <1 es menos volátil.
        $$\beta = \frac{Covarianza(Retornos\ Cartera, Retornos\ Benchmark)}{Varianza(Retornos\ Benchmark)}$$
    * **Alpha ($\alpha$) Anual**: El retorno "excedente" de tu cartera en comparación con lo que se esperaría dado su Beta y el retorno del benchmark (según el Modelo de Precios de Activos de Capital - CAPM). Un Alpha positivo indica que la cartera ha superado al benchmark después de ajustar por el riesgo sistemático.
        $$\alpha = Retorno\ Cartera - [Tasa\ Libre\ de\ Riesgo + \beta \times (Retorno\ Benchmark - Tasa\ Libre\ de\ Riesgo)]$$
    * **Tracking Error Anual**: Mide la desviación estándar de la diferencia de retornos entre tu cartera y el benchmark. Cuanto menor, más de cerca sigue tu cartera al benchmark.
        $$Tracking\ Error = Desviación\ Estándar(Retornos\ Cartera - Retornos\ Benchmark)$$
    * **Information Ratio**: Mide el Alpha generado por unidad de Tracking Error. Cuanto mayor, mejor es la habilidad del gestor para generar retornos excedentes de manera consistente.
        $$Information\ Ratio = \frac{Alpha}{Tracking\ Error}$$
* **Beta Rodante de la Cartera**: Gráfico que muestra cómo la Beta de tu cartera ha cambiado a lo largo del tiempo, revelando si tu cartera se ha vuelto más o menos sensible a los movimientos del mercado.

### 7. Análisis de Correlación entre Activos

Entiende cómo se mueven tus activos entre sí.

* **Matriz de Correlación (Período Completo)**:
    * Un mapa de calor que muestra el coeficiente de correlación entre cada par de activos de tu cartera. El coeficiente de correlación ($\rho$) entre dos activos A y B se calcula como:
        $$\rho_{A,B} = \frac{Covarianza(R_A, R_B)}{\sigma_A \times \sigma_B}$$
        Donde $R_A, R_B$ son los retornos de los activos A y B, y $\sigma_A, \sigma_B$ son sus desviaciones estándar.
    * **Valores**: Van de -1 (correlación perfectamente negativa, se mueven en direcciones opuestas) a +1 (correlación perfectamente positiva, se mueven en la misma dirección). 0 indica ausencia de correlación lineal.
    * **Utilidad**: Ayuda a identificar oportunidades de diversificación (buscando activos con baja o negativa correlación).
* **Correlación Rodante**:
    * **Correlación Promedio Rodante**: Gráfico que muestra la correlación promedio entre todos los pares de activos de tu cartera a lo largo del tiempo.
    * **Correlación Rodante (Pares Seleccionados)**: Puedes elegir pares específicos de activos para ver cómo su correlación ha evolucionado.
    * **Utilidad**: Las correlaciones no son estáticas. Esta gráfica te permite ver si tus activos se han vuelto más o menos correlacionados en diferentes entornos de mercado.

### 8. Análisis de Activos Individuales y Riesgo de Cartera

Profundiza en el rendimiento de cada componente y su impacto en el riesgo total.

* **Posicionamiento Riesgo/Retorno (Activos vs Cartera)**:
    * Gráfico de dispersión donde cada punto es un activo individual o la cartera total, con su volatilidad en el eje X y su rentabilidad en el eje Y.
    * **Utilidad**: Permite comparar visualmente la eficiencia de cada activo y ver cómo la combinación en la cartera total se posiciona en términos de riesgo y retorno. Idealmente, la cartera total debería estar "mejor" (más a la izquierda y/o más arriba) que la mayoría de sus componentes individuales, gracias a la diversificación.
* **Contribución de Activos a la Volatilidad de Cartera**:
    * Gráfico de barras que muestra el porcentaje de la volatilidad total de la cartera que es atribuible a cada activo. La contribución marginal al riesgo (MCTR) de un activo $i$ a la volatilidad de la cartera $\sigma_p$ es:
        $$MCTR_i = \frac{\partial \sigma_p}{\partial w_i} = \frac{(\Sigma w)_i}{\sigma_p}$$
        Donde $\Sigma$ es la matriz de covarianza de los retornos de los activos y $w$ es el vector de pesos de la cartera. La contribución porcentual al riesgo de cada activo es entonces:
        $$Contribución\ Porcentual_i = \frac{w_i \times MCTR_i}{\sigma_p}$$
    * **Importancia de la Contribución Negativa**: Un activo puede tener una contribución al riesgo negativa. Esto es un indicador muy positivo de diversificación. Significa que, debido a su baja o negativa correlación con otros activos, este activo está reduciendo activamente la volatilidad general de la cartera, actuando como un "amortiguador" en momentos de mercado desfavorables para otros activos. La suma de todas las contribuciones al riesgo (positivas y negativas) es igual a la varianza total de la cartera.
* **Ranking Avanzado de Activos**: Tabla que resume las métricas clave (Rentabilidad, Volatilidad, Sharpe, Sortino, Beta, Alpha, Contribución al Riesgo) para cada activo individual, permitiendo una comparación rápida.

### 9. Optimización de Cartera (Frontera Eficiente)

Encuentra la asignación de pesos ideal para tus objetivos.

* **Frontera Eficiente**:
    * Basada en la Teoría Moderna de Carteras (MPT) de Harry Markowitz, la frontera eficiente representa el conjunto de carteras óptimas que ofrecen la máxima rentabilidad esperada para un nivel de riesgo dado, o el mínimo riesgo para un nivel de rentabilidad esperada dado.
    * El objetivo general de la optimización es encontrar un vector de pesos $w$ que:
        * Minimice la Varianza de la Cartera: $\sigma_p^2 = w^T \Sigma w$
        * O Maximice el Retorno Esperado de la Cartera: $R_p = w^T \mu$
        Sujeto a restricciones como $\sum w_i = 1$ (la suma de pesos es 100%) y $0 \le w_i \le 1$ (no hay ventas en corto).
    * **Utilidad**: Te ayuda a visualizar el "mejor" trade-off riesgo-retorno posible con tus activos.
* **Cartera Mínima Varianza (MVP - Minimum Volatility Portfolio)**:
    * La cartera en la frontera eficiente que tiene el riesgo más bajo posible. Su objetivo es minimizar $\sigma_p^2$.
    * Se muestran sus pesos sugeridos y su rendimiento.
* **Cartera Máximo Sharpe (MSR - Maximum Sharpe Ratio Portfolio)**:
    * La cartera en la frontera eficiente que ofrece la mejor rentabilidad ajustada al riesgo (el punto con la mayor pendiente desde la tasa libre de riesgo). Su objetivo es maximizar el Ratio de Sharpe.
    * Se muestran sus pesos sugeridos y su rendimiento.
* **Tu Cartera Actual**: Tu cartera simulada se grafica en la frontera para que veas cómo se compara con las carteras óptimas.
* **Importancia de Datos**: La optimización requiere una cantidad significativa de datos históricos (al menos 20 días, pero idealmente varios años) para calcular matrices de covarianza fiables. Si tu período de backtest es muy corto, la optimización no se calculará.

### 10. Pruebas de Estrés y Análisis de Escenarios

Evalúa la resiliencia de tu cartera ante eventos adversos.

* **Shock Hipotético de Mercado**:
    * Permite simular el impacto de una caída (o subida) porcentual instantánea en el valor de tu cartera.
    * Cálculo: $Valor\ Post-Shock = Último\ Valor\ Cartera \times (1 + Porcentaje\ Shock/100)$
    * **Utilidad**: Te ayuda a entender la pérdida absoluta que sufrirías si el mercado experimentara un shock repentino de la magnitud que definas.
* **Análisis de Escenario Histórico**:
    * Simula cómo se habría comportado tu cartera actual (con sus pesos actuales) si hubiera existido durante un período de crisis histórica predefinido (ej., Crisis Financiera Global 2008, Burbuja .com, COVID-19 Crash).
    * **Utilidad**: Es una prueba de estrés basada en eventos reales. Te muestra la rentabilidad total, el drawdown máximo y la volatilidad que tu cartera (con su composición actual) habría experimentado en esos momentos difíciles.
    * **Requisito**: Para que funcione, tu archivo de precios principal debe contener datos que cubran las fechas de esos escenarios históricos. Si tu archivo solo tiene datos recientes (ej., desde 2024), no podrá simular un escenario de 2008.

### 11. Simulación de Montecarlo (Proyección Futura)

Proyecta posibles caminos futuros para tu cartera.

* **Propósito**: Estima un rango de posibles valores futuros para tu cartera basándose en la volatilidad y el retorno históricos.
* **Cómo funciona**: Genera miles de "caminos" aleatorios para el valor de tu cartera. Se asume que los retornos diarios de la cartera siguen una distribución normal con la media ($\mu$) y la desviación estándar ($\sigma$) de tus retornos históricos. Para cada día de la proyección y cada simulación, se genera un retorno aleatorio:
    $$R_t \sim N(\mu, \sigma)$$
    El valor de la cartera se actualiza iterativamente:
    $$P_t = P_{t-1} \times (1 + R_t)$$
* **Resultados**:
    * **Gráfico de Distribución**: Muestra la media de las simulaciones, y percentiles (ej., 5% y 95%) para visualizar el rango probable de resultados.
    * **Resumen Estadístico**: Proporciona el valor medio, la mediana, los percentiles (ej., peor 5%, mejor 5%), y los valores mínimo y máximo alcanzados en las simulaciones.
* **Importancia de Datos**: Requiere una cantidad significativa de retornos históricos (idealmente varios años) para que la media y desviación estándar sean representativas y la simulación sea fiable. Si el período de backtest es muy corto, la simulación podría ser engañosa o no ejecutarse.

---

## 🛠️ Cómo Usar la Aplicación

**Prerrequisitos**:

* Asegúrate de tener Python instalado (versión 3.8 o superior recomendada).
* Instala las librerías necesarias (si aún no lo has hecho):
    ```bash
    pip install streamlit pandas numpy empyrical pypfopt plotly matplotlib seaborn
    ```

**Estructura de Archivos**:

* Asegúrate de que tus archivos Python (`app.py`, `src/analysis.py`, `src/constants.py`, `src/models.py`, `src/optimization.py`, `src/simulation.py`, `src/utils.py`, `src/visualization.py`) estén organizados en la estructura de carpetas correcta (`app.py` en la raíz, y el resto dentro de una carpeta `src`).

**Ejecutar la Aplicación**:

1.  Abre tu terminal o línea de comandos.
2.  Navega hasta el directorio donde se encuentra tu archivo `app.py`.
3.  Ejecuta el siguiente comando:
    ```bash
    streamlit run app.py
    ```
    Esto abrirá la aplicación en tu navegador web.

**Flujo de Trabajo en la Aplicación**:

1.  **Carga tu archivo CSV (Precios y Benchmark)**: Sube tu archivo de precios históricos.
2.  **Selecciona la Columna Benchmark**: Elige el índice de referencia.
2a. **Activos para Restricciones de Optimización**: (Opcional) Marca los activos de Renta Fija y Monetarios.
3.  **Fecha de Inicio / 4. Fecha de Fin**: Define el período de tu backtest.
5.  **Inversión Inicial (€)**: Introduce el capital inicial.
6.  **Frecuencia de Rebalanceo**: Elige la periodicidad de rebalanceo.
    * **Coste Transacción (pb por operación)**: Introduce las comisiones.
7.  **Ventana Análisis Rodante (días)**: Define la ventana para los cálculos móviles.
8.  **Tasa Libre de Riesgo Anual (%)**: Introduce la tasa de referencia.
    * **Análisis de Riesgo Avanzado y Escenarios**: Configura los parámetros de VaR/ES, shock hipotético y selecciona un escenario histórico.
    * **Simulación Montecarlo**: Configura el número de simulaciones y días de proyección.
9.  **Asignación de Pesos (%)**:
    * Elige método de asignación de pesos: Selecciona "manual" o "archivo".
    * Si eliges "manual": Introduce los porcentajes para cada fondo.
    * Si eliges "archivo": Sube tu archivo CSV/Excel con los pesos dinámicos (ver formato en la sección 2).
10. **🚀 Ejecutar Análisis Completo**: Haz clic en este botón para iniciar todos los cálculos y ver los resultados en las diferentes pestañas.
11. **Explora las Pestañas**: Navega por las diferentes pestañas ("Visión General", "vs Benchmark", "Correlación", etc.) para ver los resultados detallados y las visualizaciones.

---

## ⚠️ Solución de Problemas Comunes

* **"The min_value, set to AAAA-MM-DD, shouldn't be larger than the max_value, set to AAAA-MM-DD."**
    * **Causa**: Las fechas de inicio/fin que Streamlit intenta establecer (o las que tú eliges) están fuera del rango de fechas de tus datos de precios cargados. Típicamente, la fecha de inicio es posterior a la última fecha de tus datos.
    * **Solución**: Asegúrate de que tu archivo CSV de precios (`data`) contenga datos que abarquen el período de backtest deseado. Si tus datos terminan en una fecha antigua, ajusta manualmente la "Fecha de Inicio" y "Fecha de Fin" en la barra lateral para que estén dentro del rango de tus datos.

* **"No se han proporcionado pesos manuales ni un archivo de pesos dinámicos."**
    * **Causa**: Después de presionar "Ejecutar Análisis Completo", la aplicación no pudo encontrar un conjunto de pesos válido (ni los manuales ni el archivo dinámico). Esto suele ocurrir si no se seleccionó ningún método de pesos, no se introdujeron pesos manuales, o el archivo de pesos dinámicos no se subió o no pudo ser procesado correctamente.
    * **Solución**:
        * Verifica que has seleccionado "manual" o "archivo" en el radio button de pesos.
        * Si es "manual", asegúrate de que al menos un fondo tenga un peso >0.
        * Si es "archivo", verifica que el archivo se subió correctamente (deberías ver un mensaje de éxito) y que su formato es el esperado.

* **"Insuficientes retornos históricos para una simulación de Montecarlo fiable." / "No hay suficientes filas de datos (...) para una optimización robusta." / "No se pudieron obtener datos para el escenario histórico..."**
    * **Causa**: Las funcionalidades avanzadas (Optimización, Montecarlo, Escenarios Históricos) requieren un mínimo de datos históricos (idealmente varios años) para realizar cálculos significativos y robustos. Tu período de backtest actual es demasiado corto.
    * **Solución**: Carga un archivo de precios que contenga muchos años de datos para tus activos y el benchmark. Luego, selecciona un rango de fechas de backtest más amplio (ej., 3-5 años o más) en la barra lateral.

* **"Benchmark 'X' no encontrado o sin datos válidos en el rango seleccionado."**
    * **Causa**: La columna que seleccionaste como benchmark no existe en tu archivo de precios, o existe pero no tiene datos válidos (solo NaNs) en el período de backtest seleccionado.
    * **Solución**: Revisa tu archivo de precios y el nombre de la columna del benchmark. Asegúrate de que contenga datos válidos para el período de backtest.

* **"Error crítico al procesar el archivo CSV..." / "Error convirtiendo columna 'X' a numérico..."**
    * **Causa**: Problemas con el formato de tu archivo CSV (separadores, decimales, fechas, caracteres no numéricos en columnas de precios).
    * **Solución**: Abre tu archivo CSV con un editor de texto plano y verifica que el formato sea consistente. Asegúrate de que la primera columna sea siempre la fecha, y que los números usen el mismo separador decimal en todas partes.

---

## 📁 Estructura del Código

El proyecto está organizado en módulos para una mejor legibilidad y mantenimiento:

* `app.py`: El archivo principal de Streamlit que orquesta la interfaz de usuario y llama a las funciones de los otros módulos.
* `src/`: Directorio que contiene todos los módulos de lógica de negocio.
    * `src/analysis.py`: Contiene funciones para calcular métricas de rendimiento (Sharpe, Sortino, Drawdown, etc.), correlaciones y contribución al riesgo.
    * `src/constants.py`: Define constantes globales como la tasa libre de riesgo por defecto y los escenarios históricos predefinidos.
    * `src/models.py`: Implementa modelos de riesgo (VaR, ES) y funciones para pruebas de estrés y simulación de Montecarlo.
    * `src/optimization.py`: Contiene la lógica para la optimización de carteras (Frontera Eficiente, MVP, MSR) utilizando PyPortfolioOpt.
    * `src/simulation.py`: Implementa la lógica central del backtest, incluyendo el manejo de pesos dinámicos, rebalanceo y costes de transacción.
    * `src/utils.py`: Funciones de utilidad para la carga y preprocesamiento de datos (precios y pesos dinámicos).
    * `src/visualization.py`: Contiene funciones para generar todos los gráficos utilizando Plotly y Matplotlib/Seaborn.

---

¡Esperamos que disfrutes usando el Backtester Quant v5.0 Extendido y que te sea de gran utilidad en tu análisis de inversiones!

Dios Familia y Cojones.