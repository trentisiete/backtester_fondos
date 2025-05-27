# src/simulation.py

import pandas as pd
import numpy as np
import copy
import streamlit as st # Streamlit is needed for st.warning/st.error

def run_backtest(data, weights_dict, initial_investment, start_date, end_date, rebalance_freq, transaction_cost_bps=0.0):
    """Ejecuta la simulación del backtesting para la cartera, incluyendo costes de transacción."""
    asset_columns = list(weights_dict.keys())
    start_date_ts = pd.to_datetime(start_date)
    end_date_ts = pd.to_datetime(end_date)

    data_in_range = data.loc[data.index.intersection(pd.date_range(start_date_ts, end_date_ts))]
    if data_in_range.empty:
        st.warning("No hay datos en el índice para el rango de fechas seleccionado.")
        return None, None, 0.0 # portfolio_value, portfolio_returns, total_transaction_costs

    prices = data_in_range[asset_columns].copy()

    if prices.empty or prices.isnull().all().all():
        st.warning("No hay datos válidos para los activos en el rango de fechas seleccionado.")
        return None, None, 0.0

    prices.ffill(inplace=True)
    prices.bfill(inplace=True)

    if prices.iloc[0].isnull().any():
        missing_funds = prices.columns[prices.iloc[0].isnull()].tolist()
        st.warning(f"Faltan datos iniciales para: {', '.join(missing_funds)} en {prices.index[0].date()}.")
        first_valid_date = prices.dropna(axis=0, how='any').index.min()
        if pd.isna(first_valid_date) or first_valid_date > end_date_ts:
            st.error("No hay datos comunes suficientes para iniciar el backtest en el rango seleccionado.")
            return None, None, 0.0
        prices = prices.loc[first_valid_date:].copy()
        st.warning(f"Backtest comenzará en {prices.index[0].date()}.")
        if prices.empty: return None, None, 0.0

    total_weight = sum(weights_dict.values())
    if not np.isclose(total_weight, 1.0) and total_weight != 0:
        weights_dict = {k: v / total_weight for k, v in weights_dict.items()}

    portfolio_value = pd.Series(index=prices.index, dtype=float)
    if not prices.empty:
        portfolio_value.loc[prices.index[0]] = initial_investment
    else:
        st.error("No se pueden inicializar los valores de la cartera.")
        return None, None, 0.0

    current_weights = weights_dict.copy()
    last_rebalance_date = prices.index[0]

    # Costes de transacción
    total_transaction_costs_value = 0.0
    transaction_cost_rate = transaction_cost_bps / 10000.0 # Convertir bps a decimal

    # Asignación inicial y cálculo de participaciones
    initial_alloc_value = {}
    shares = {}
    initial_turnover = 0 # Para el coste de la primera asignación

    for fund in current_weights:
        initial_price = prices[fund].iloc[0]
        if pd.notna(initial_price) and initial_price != 0:
            target_value_fund = initial_investment * current_weights[fund]
            initial_alloc_value[fund] = target_value_fund
            shares[fund] = target_value_fund / initial_price
            initial_turnover += target_value_fund # El "turnover" inicial es el valor total invertido
        else:
            shares[fund] = 0
            initial_alloc_value[fund] = 0
            st.warning(f"Precio inicial inválido para {fund} en {prices.index[0].date()}. Peso inicial no aplicado.")

    cost_initial_alloc = initial_turnover * transaction_cost_rate
    total_transaction_costs_value += cost_initial_alloc
    portfolio_value.loc[prices.index[0]] -= cost_initial_alloc # Deducir coste de la inversión inicial

    rebalance_offset = {'Mensual': pd.DateOffset(months=1), 'Trimestral': pd.DateOffset(months=3),
                        'Anual': pd.DateOffset(years=1), 'No Rebalancear': None}
    offset = rebalance_offset[rebalance_freq]

    for i in range(1, len(prices)):
        current_date = prices.index[i]
        prev_date = prices.index[i-1]

        # Calcular valor de cartera antes de rebalanceo
        current_portfolio_value_before_rebal = 0.0
        for fund in shares:
            current_price = prices.loc[current_date, fund] if current_date in prices.index and pd.notna(prices.loc[current_date, fund]) else \
                                prices.loc[prev_date, fund] if prev_date in prices.index and pd.notna(prices.loc[prev_date, fund]) else np.nan

            if pd.notna(current_price):
                current_portfolio_value_before_rebal += shares[fund] * current_price
            else: # Si no hay precio actual ni previo, usar el valor de cartera del día anterior
                current_portfolio_value_before_rebal = portfolio_value.loc[prev_date] if prev_date in portfolio_value else 0
                st.warning(f"Usando valor de cartera previo para {current_date} debido a precio faltante para {fund}")
                break

        if pd.isna(current_portfolio_value_before_rebal) or (current_portfolio_value_before_rebal == 0 and prev_date in portfolio_value and portfolio_value.loc[prev_date] != 0):
             current_portfolio_value_before_rebal = portfolio_value.loc[prev_date] if prev_date in portfolio_value and pd.notna(portfolio_value.loc[prev_date]) else 0


        portfolio_value.loc[current_date] = current_portfolio_value_before_rebal

        # Lógica de Rebalanceo
        if offset and current_date >= last_rebalance_date + offset:
            if pd.notna(current_portfolio_value_before_rebal) and current_portfolio_value_before_rebal > 0:
                turnover_rebalance = 0
                new_shares = {}

                # Calcular valor actual de cada activo antes de rebalancear
                current_asset_values = {}
                for fund in shares:
                    price_at_rebal = prices.loc[current_date, fund] if current_date in prices.index and pd.notna(prices.loc[current_date, fund]) else np.nan
                    if pd.notna(price_at_rebal):
                        current_asset_values[fund] = shares[fund] * price_at_rebal
                    else: # Si falta precio en rebalanceo, no se puede rebalancear ese activo
                        current_asset_values[fund] = 0
                        new_shares[fund] = shares[fund] # Mantener participaciones antiguas
                        st.warning(f"Precio inválido para {fund} en rebalanceo {current_date}. No se rebalanceó este activo.")

                for fund in weights_dict:
                    target_value_fund = current_portfolio_value_before_rebal * weights_dict[fund]
                    price_at_rebal = prices.loc[current_date, fund] if current_date in prices.index and pd.notna(prices.loc[current_date, fund]) else np.nan

                    if pd.notna(price_at_rebal) and price_at_rebal > 0:
                        new_shares[fund] = target_value_fund / price_at_rebal
                        # Calcular turnover (valor absoluto de la diferencia entre valor objetivo y valor actual)
                        # Solo se cuenta la mitad del turnover (compras o ventas)
                        turnover_rebalance += abs(target_value_fund - current_asset_values.get(fund, 0))
                    elif fund not in new_shares: # Si no se pudo calcular antes y no está en new_shares
                        new_shares[fund] = shares.get(fund, 0) # Mantener participaciones si no se puede rebalancear

                shares = new_shares # Actualizar participaciones

                cost_this_rebalance = (turnover_rebalance / 2) * transaction_cost_rate # Se divide por 2 porque el turnover cuenta compras y ventas
                total_transaction_costs_value += cost_this_rebalance
                portfolio_value.loc[current_date] -= cost_this_rebalance # Deducir costes del valor de la cartera

                last_rebalance_date = current_date
            else:
                st.warning(f"Valor de cartera inválido ({current_portfolio_value_before_rebal}) en rebalanceo {current_date}. Omitido.")

    portfolio_returns = portfolio_value.pct_change().dropna()
    return portfolio_value.dropna(), portfolio_returns, total_transaction_costs_value