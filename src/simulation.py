# src/simulation.py

import pandas as pd
import numpy as np
import copy
import streamlit as st # Streamlit is needed for st.warning/st.error

def run_backtest(data, initial_investment, start_date, end_date, rebalance_freq, transaction_cost_bps=0.0, initial_manual_weights=None, dynamic_weights_df=None):
    """
    Ejecuta la simulación del backtesting para la cartera, incluyendo costes de transacción.
    Permite pesos fijos con rebalanceo periódico o pesos dinámicos desde un DataFrame.
    """
    start_date_ts = pd.to_datetime(start_date)
    end_date_ts = pd.to_datetime(end_date)

    # Validar que al menos un método de pesos está activo
    if dynamic_weights_df is None and (initial_manual_weights is None or not initial_manual_weights):
        st.error("No se han proporcionado pesos manuales ni un archivo de pesos dinámicos.")
        return None, None, 0.0

    # Determinar los fondos a considerar en el backtest
    # Si hay pesos dinámicos, los fondos son los del archivo de pesos.
    # Si son manuales, los fondos son los de initial_manual_weights.
    if dynamic_weights_df is not None and not dynamic_weights_df.empty:
        asset_columns = dynamic_weights_df.columns.tolist()
    else: # Usar pesos manuales
        asset_columns = list(initial_manual_weights.keys())

    # Asegurarse de que los activos seleccionados existen en los datos de precios
    funds_in_data = [fund for fund in asset_columns if fund in data.columns]
    if not funds_in_data:
        st.error("Ninguno de los activos definidos en los pesos (manuales o dinámicos) se encuentra en los datos de precios.")
        return None, None, 0.0
    
    # Filtrar datos de precios por el rango de fechas y los activos que realmente están en el portfolio
    data_in_range = data.loc[data.index.intersection(pd.date_range(start_date_ts, end_date_ts)), funds_in_data].copy()

    if data_in_range.empty:
        st.warning("No hay datos en el rango de fechas seleccionado para los activos del portfolio.")
        return None, None, 0.0

    prices = data_in_range.copy()

    # Rellenar NaNs para evitar interrupciones
    prices.ffill(inplace=True)
    prices.bfill(inplace=True)

    if prices.iloc[0].isnull().any():
        missing_funds_initial = prices.columns[prices.iloc[0].isnull()].tolist()
        st.warning(f"Faltan datos iniciales para: {', '.join(missing_funds_initial)} en {prices.index[0].date()}. Asegúrate de que los precios están disponibles desde el inicio.")
        # Ajustar la fecha de inicio del backtest a la primera fecha con datos completos para los activos seleccionados
        first_valid_date_for_assets = prices.dropna(axis=0, how='any').index.min()
        if pd.isna(first_valid_date_for_assets) or first_valid_date_for_assets > end_date_ts:
            st.error("No hay datos completos para los activos seleccionados en el rango. Ajusta el rango o revisa los datos.")
            return None, None, 0.0
        prices = prices.loc[first_valid_date_for_assets:].copy()
        start_date_ts = first_valid_date_for_assets
        st.warning(f"Backtest comenzará efectivamente en {prices.index[0].date()} debido a datos faltantes.")
        if prices.empty: return None, None, 0.0

    portfolio_value = pd.Series(index=prices.index, dtype=float)
    if prices.empty:
        st.error("No se pueden inicializar los valores de la cartera: datos de precios vacíos tras la preparación.")
        return None, None, 0.0

    # Inicialización del backtest
    current_weights_active = {} # Diccionario de pesos que se están usando actualmente
    total_transaction_costs_value = 0.0
    transaction_cost_rate = transaction_cost_bps / 10000.0 # Convertir bps a decimal

    last_rebalance_date = prices.index[0] # Fecha del último rebalanceo o inicio

    # Determinar los pesos iniciales a aplicar
    if dynamic_weights_df is not None and not dynamic_weights_df.empty:
        # Usar los pesos del archivo más cercanos a la fecha de inicio del backtest
        # Asegurar que dynamic_weights_df contiene las columnas de los fondos a considerar
        dynamic_weights_for_funds = dynamic_weights_df[funds_in_data]
        
        initial_weights_series = dynamic_weights_for_funds.asof(start_date_ts)
        if initial_weights_series is None or initial_weights_series.isnull().all():
            st.error(f"No hay pesos definidos en el archivo de pesos dinámicos para la fecha de inicio {start_date_ts.date()} o anteriores. No se puede iniciar el backtest.")
            return None, None, 0.0
        current_weights_active = initial_weights_series.dropna().to_dict()
        
    else: # Modo de pesos manuales
        current_weights_active = initial_manual_weights.copy()
        
    # Normalizar los pesos activos iniciales (por si el usuario puso porcentajes o no sumó 1)
    sum_active_weights = sum(current_weights_active.values())
    if not np.isclose(sum_active_weights, 1.0) and sum_active_weights > 1e-6:
        current_weights_active = {k: v / sum_active_weights for k, v in current_weights_active.items()}
    elif sum_active_weights <= 1e-6:
        st.error("Los pesos iniciales (manuales o dinámicos) suman cero o son inválidos. No se puede iniciar el backtest.")
        return None, None, 0.0

    # Calcular participaciones iniciales
    shares = {}
    initial_turnover_value = 0.0 # Valor que se está comprando/vendiendo en la inicialización
    
    # Calcular el valor total de la inversión inicial con los pesos normalizados
    total_initial_portfolio_value = initial_investment

    for fund, weight in current_weights_active.items():
        if fund in prices.columns:
            initial_price = prices[fund].iloc[0]
            if pd.notna(initial_price) and initial_price > 0:
                target_value_fund = total_initial_portfolio_value * weight
                shares[fund] = target_value_fund / initial_price
                initial_turnover_value += target_value_fund # Todo el valor invertido es turnover inicial
            else:
                shares[fund] = 0
                st.warning(f"Precio inicial inválido o cero para {fund} en {prices.index[0].date()}. No se asignó peso inicial.")
        else:
            shares[fund] = 0
            st.warning(f"El activo {fund} de los pesos no se encontró en los datos de precios. Se ignorará.")

    # Deducción del coste de transacción inicial
    cost_initial_alloc = initial_turnover_value * transaction_cost_rate
    total_transaction_costs_value += cost_initial_alloc
    
    # El valor inicial de la cartera se establece después de los costes
    portfolio_value.loc[prices.index[0]] = total_initial_portfolio_value - cost_initial_alloc

    rebalance_offset = {'Mensual': pd.DateOffset(months=1), 'Trimestral': pd.DateOffset(months=3),
                        'Anual': pd.DateOffset(years=1), 'No Rebalancear': None}
    offset = rebalance_offset[rebalance_freq]

    # Bucle principal del backtest
    for i in range(1, len(prices)):
        current_date = prices.index[i]
        prev_date = prices.index[i-1]

        # 1. Calcular el valor de la cartera al final del día (antes de cualquier rebalanceo del día actual)
        current_portfolio_value_before_rebal = 0.0
        missing_price_on_day = False
        for fund, num_shares in shares.items():
            current_price = prices.loc[current_date, fund] if current_date in prices.index and pd.notna(prices.loc[current_date, fund]) else np.nan
            
            if pd.notna(current_price):
                current_portfolio_value_before_rebal += num_shares * current_price
            else:
                # Si falta el precio, usar el último precio conocido.
                # Ya se hizo ffill/bfill al inicio, así que esto debería ser raro, pero por seguridad
                if prev_date in prices.index and pd.notna(prices.loc[prev_date, fund]):
                    current_portfolio_value_before_rebal += num_shares * prices.loc[prev_date, fund]
                else:
                    # Si no hay precio actual ni previo (grave), romper y usar el valor de cartera del día anterior
                    missing_price_on_day = True
                    break # Salir del bucle interno de fondos
        
        if missing_price_on_day:
            # Si un precio de un activo falló, todo el cálculo del día es incierto.
            # Mejor usar el valor de cartera del día anterior y no recalcular
            current_portfolio_value_before_rebal = portfolio_value.loc[prev_date] if prev_date in portfolio_value else 0.0
            st.warning(f"Usando valor de cartera previo para {current_date} debido a precio faltante/inválido para algunos activos.")
            
        portfolio_value.loc[current_date] = current_portfolio_value_before_rebal

        # 2. Lógica de Rebalanceo (tanto por frecuencia como por cambio en pesos dinámicos)
        perform_rebalance = False
        if offset and current_date >= last_rebalance_date + offset:
            perform_rebalance = True
        
        # Si hay pesos dinámicos, verificar si los pesos objetivo han cambiado para la fecha actual
        if dynamic_weights_df is not None and not dynamic_weights_df.empty:
            target_weights_for_date_series = dynamic_weights_for_funds.asof(current_date)
            if target_weights_for_date_series is not None and not target_weights_for_date_series.isnull().all():
                target_weights_for_date = target_weights_for_date_series.dropna().to_dict()
                
                # Check if target weights are different from current active weights
                # Convert to Series for easy comparison and drop any zero-weighted items for comparison purposes
                current_weights_series = pd.Series(current_weights_active).loc[lambda x: x > 1e-9].sort_index()
                target_weights_series = pd.Series(target_weights_for_date).loc[lambda x: x > 1e-9].sort_index()

                # Align indices before comparison
                common_assets = current_weights_series.index.intersection(target_weights_series.index)
                
                weights_differ = False
                if not current_weights_series.empty or not target_weights_series.empty:
                    if not current_weights_series.index.equals(target_weights_series.index): # Different assets or order
                        weights_differ = True
                    elif not np.allclose(current_weights_series, target_weights_series): # Same assets, different weights
                        weights_differ = True
                
                if weights_differ:
                    perform_rebalance = True
                    current_weights_active = target_weights_for_date # Actualizar los pesos activos para el rebalanceo
                    
                    # Log a warning if dynamic weights violate optimization constraints (just for user info)
                    fi_sum = sum(current_weights_active.get(f, 0) for f in st.session_state['fixed_income_selection'] if f in current_weights_active)
                    mm_sum = sum(current_weights_active.get(f, 0) for f in st.session_state['money_market_selection'] if f in current_weights_active)
                    if fi_sum > 0.09 + 1e-6: # Allow a small tolerance
                         st.warning(f"Advertencia: En {current_date.date()}, los pesos de Renta Fija del archivo ({fi_sum:.2%}) exceden el límite sugerido (9%).")
                    if mm_sum > 0.01 + 1e-6: # Allow a small tolerance
                         st.warning(f"Advertencia: En {current_date.date()}, los pesos de Monetario del archivo ({mm_sum:.2%}) exceden el límite sugerido (1%).")

        if perform_rebalance and pd.notna(current_portfolio_value_before_rebal) and current_portfolio_value_before_rebal > 0:
            turnover_rebalance = 0.0
            new_shares = {}

            # Calcular el valor actual de cada activo justo antes del rebalanceo
            current_asset_values = {}
            for fund, num_shares in shares.items():
                price_at_rebal = prices.loc[current_date, fund] if current_date in prices.index and pd.notna(prices.loc[current_date, fund]) else np.nan
                if pd.notna(price_at_rebal):
                    current_asset_values[fund] = num_shares * price_at_rebal
                else:
                    # Si no hay precio para un activo, no podemos calcular su valor actual para rebalanceo
                    # Mantener sus acciones actuales y advertir
                    new_shares[fund] = num_shares
                    st.warning(f"Precio inválido para {fund} en rebalanceo {current_date}. Este activo no se rebalanceó.")
                    current_asset_values[fund] = 0 # Considerar su valor como 0 para el cálculo de turnover si no se rebalancea
            
            # Recalcular el valor total de la cartera basado solo en los activos con precios válidos para el rebalanceo
            # Esto evita que una asignación incorrecta o un precio faltante arruinen el rebalanceo
            valid_rebal_portfolio_value = sum(current_asset_values.values())
            if valid_rebal_portfolio_value == 0:
                st.warning(f"Valor de cartera calculado en {current_date} para rebalanceo es cero. Omitiendo rebalanceo.")
                last_rebalance_date = current_date # Actualizar para evitar rebalanceo constante si hay problemas
                continue # Saltar el rebalanceo actual

            # Aplicar los nuevos pesos (ya sea de dynamic_weights_df o los initial_manual_weights)
            for fund, target_weight in current_weights_active.items():
                if fund in prices.columns:
                    price_at_rebal = prices.loc[current_date, fund] if current_date in prices.index and pd.notna(prices.loc[current_date, fund]) else np.nan
                    
                    if pd.notna(price_at_rebal) and price_at_rebal > 0:
                        target_value_fund = valid_rebal_portfolio_value * target_weight
                        new_shares[fund] = target_value_fund / price_at_rebal
                        # Turnover: abs(valor_objetivo - valor_actual_antes_de_rebalanceo)
                        turnover_rebalance += abs(target_value_fund - current_asset_values.get(fund, 0.0))
                    elif fund not in new_shares: # Si el activo no se pudo calcular antes y no está en new_shares
                        new_shares[fund] = shares.get(fund, 0) # Mantener participaciones si no se puede rebalancear
                elif fund not in new_shares: # Si el activo no está en los datos de precios en absoluto
                    new_shares[fund] = 0 # No se puede operar con él
                    st.warning(f"El activo {fund} no se encuentra en los datos de precios. No se pudo rebalancear.")


            shares = new_shares # Actualizar participaciones
            
            # El turnover total para el cálculo de costes es la suma de las diferencias de valor, pero solo se aplica una vez (compra o venta)
            cost_this_rebalance = (turnover_rebalance / 2.0) * transaction_cost_rate # Dividir por 2 porque el turnover suma ambos lados de la transacción

            total_transaction_costs_value += cost_this_rebalance
            portfolio_value.loc[current_date] -= cost_this_rebalance # Deducir costes del valor de la cartera

            last_rebalance_date = current_date # Actualizar la fecha del último rebalanceo

        elif perform_rebalance and (pd.isna(current_portfolio_value_before_rebal) or current_portfolio_value_before_rebal <= 0):
            st.warning(f"Valor de cartera inválido ({current_portfolio_value_before_rebal:.2f}€) en rebalanceo {current_date}. Omitido.")

    # Asegurar que el DataFrame de valor de cartera no tenga NaNs al final
    portfolio_value.ffill(inplace=True)
    portfolio_value.bfill(inplace=True)
    portfolio_value.dropna(inplace=True)

    portfolio_returns = portfolio_value.pct_change().dropna()
    return portfolio_value, portfolio_returns, total_transaction_costs_value