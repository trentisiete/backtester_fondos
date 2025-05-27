# src/optimization.py

import pandas as pd
import numpy as np
import copy
import pypfopt
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions
import cvxpy as cp # Para definir restricciones
# import streamlit as st # REMOVE THIS LINE

def optimize_portfolio(prices, risk_free_rate=0.0, fixed_income_assets=None, money_market_assets=None):
    """
    Calcula la frontera eficiente y carteras óptimas (Min Vol, Max Sharpe)
    con restricciones opcionales para Renta Fija y Monetario.
    """
    if prices is None or prices.empty or prices.shape[1] < 2:
        # st.warning("Se necesitan al menos 2 activos con datos válidos para la optimización.") # REMOVED
        return None, None, None, None, None

    results = {}
    try:
        prices_cleaned = prices.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
        prices_cleaned = prices_cleaned.dropna(axis=1, how='all').dropna(axis=0, how='all').ffill().bfill()
        prices_cleaned = prices_cleaned.dropna(axis=1, how='any') # Asegurar que no hay columnas enteras NaN

        if prices_cleaned.shape[1] < 2:
            # st.warning(f"Menos de 2 activos con datos suficientes para optimización. Activos: {prices_cleaned.columns.tolist()}") # REMOVED
            return None, None, None, None, None
        if prices_cleaned.shape[0] < 20: # PyPortfolioOpt puede necesitar más datos
            # st.warning(f"No hay suficientes filas de datos ({prices_cleaned.shape[0]}) para una optimización robusta. Se requieren al menos 20.") # REMOVED
            return None, None, None, None, None

        mu = expected_returns.mean_historical_return(prices_cleaned, frequency=252)
        S = risk_models.CovarianceShrinkage(prices_cleaned, frequency=252).ledoit_wolf()

        ef = EfficientFrontier(mu, S, weight_bounds=(0, 1)) # Solo largos

        # Renta Fija (Máx 9%)
        if fixed_income_assets:
            fixed_income_in_model = [asset for asset in fixed_income_assets if asset in ef.tickers]
            if fixed_income_in_model:
                fi_indices = [ef.tickers.index(asset) for asset in fixed_income_in_model]
                if fi_indices:
                    ef.add_constraint(lambda w: cp.sum(w[fi_indices]) <= 0.09)

        # Monetario (Máx 1%)
        if money_market_assets:
            money_market_in_model = [asset for asset in money_market_assets if asset in ef.tickers]
            if money_market_in_model:
                mm_indices = [ef.tickers.index(asset) for asset in money_market_in_model]
                if mm_indices:
                    ef.add_constraint(lambda w: cp.sum(w[mm_indices]) <= 0.01)

        # MVP
        try:
            ef_min_vol = copy.deepcopy(ef)
            mvp_weights_raw = ef_min_vol.min_volatility()
            mvp_weights = {k: v for k, v in mvp_weights_raw.items() if abs(v) > 1e-5} # Umbral pequeño para limpieza
            mvp_performance = ef_min_vol.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            results['mvp_weights'] = mvp_weights
            results['mvp_performance'] = {'expected_return': mvp_performance[0], 'annual_volatility': mvp_performance[1], 'sharpe_ratio': mvp_performance[2]}
        except (ValueError, cp.error.SolverError) as e_mvp:
            if "infeasible" in str(e_mvp).lower() or "optimal_inaccurate" in str(e_mvp).lower():
                # st.warning(f"Optimización MVP infactible o imprecisa (posiblemente por restricciones): {e_mvp}") # REMOVED
                pass # handle message in app.py
            else:
                # st.warning(f"No se pudo calcular MVP: {e_mvp}") # REMOVED
                pass # handle message in app.py
            results['mvp_weights'], results['mvp_performance'] = None, None
        except Exception as e_mvp_other:
            # st.warning(f"Error inesperado en MVP: {e_mvp_other}") # REMOVED
            pass # handle message in app.py
            results['mvp_weights'], results['mvp_performance'] = None, None

        # Max Sharpe
        try:
            if not (mu > risk_free_rate).any(): # Si ningún activo supera Rf, Max Sharpe puede ser problemático
                # st.warning("Ningún activo supera la tasa libre de riesgo; Max Sharpe puede ser igual a MVP o fallar.") # REMOVED
                pass # handle message in app.py

            ef_max_sharpe = copy.deepcopy(ef)
            ef_max_sharpe.add_objective(objective_functions.L2_reg, gamma=0.1) # Regularización para estabilidad
            msr_weights_raw = ef_max_sharpe.max_sharpe(risk_free_rate=risk_free_rate)
            msr_weights = {k: v for k, v in msr_weights_raw.items() if abs(v) > 1e-5}
            msr_performance = ef_max_sharpe.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            results['msr_weights'] = msr_weights
            results['msr_performance'] = {'expected_return': msr_performance[0], 'annual_volatility': msr_performance[1], 'sharpe_ratio': msr_performance[2]}
        except (ValueError, cp.error.SolverError) as e_msr:
            if "infeasible" in str(e_msr).lower() or "optimal_inaccurate" in str(e_msr).lower():
                # st.warning(f"Optimización Max Sharpe infactible o imprecisa (posiblemente por restricciones): {e_msr}") # REMOVED
                pass # handle message in app.py
            elif "exceeding the risk-free rate" not in str(e_msr).lower(): # No mostrar este error si es el de Rf
                # st.warning(f"No se pudo calcular Max Sharpe: {e_msr}") # REMOVED
                pass # handle message in app.py
            results['msr_weights'], results['msr_performance'] = None, None
        except Exception as e_msr_other:
            # st.warning(f"Error inesperado en Max Sharpe: {e_msr_other}") # REMOVED
            pass # handle message in app.py
            results['msr_weights'], results['msr_performance'] = None, None

        # Frontera Eficiente
        try:
            ef_frontier_calc = copy.deepcopy(ef)
            n_samples = 100

            min_ret_for_frontier = mu.min()
            if results.get('mvp_performance') and results['mvp_performance']['expected_return'] > mu.min():
                min_ret_for_frontier = results['mvp_performance']['expected_return']

            max_ret_for_frontier = mu.max()

            if min_ret_for_frontier >= max_ret_for_frontier - 1e-4:
                # st.warning("Rango de retornos para la frontera es inválido o demasiado pequeño. No se puede trazar.") # REMOVED
                raise ValueError("Rango inválido para frontera")

            target_returns = np.linspace(min_ret_for_frontier, max_ret_for_frontier, n_samples)
            frontier_volatility, frontier_sharpe = [], []

            for target_ret in target_returns:
                try:
                    ef_point = copy.deepcopy(ef_frontier_calc) # Usar una copia fresca para cada punto
                    ef_point.efficient_return(target_return=target_ret)
                    perf = ef_point.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
                    frontier_volatility.append(perf[1])
                    frontier_sharpe.append(perf[2])
                except (ValueError, cp.error.SolverError): # Si un punto es infactible
                    frontier_volatility.append(np.nan)
                    frontier_sharpe.append(np.nan)

            frontier_df = pd.DataFrame({'Return': target_returns, 'Volatility': frontier_volatility, 'Sharpe': frontier_sharpe}).dropna().sort_values(by='Volatility')
            results['frontier_df'] = frontier_df
        except Exception as e_frontier:
            # st.warning(f"No se pudieron generar puntos de la frontera: {e_frontier}") # REMOVED
            pass # handle message in app.py
            results['frontier_df'] = None

        return (results.get('mvp_weights'), results.get('msr_weights'),
                results.get('mvp_performance'), results.get('msr_performance'),
                results.get('frontier_df'))
    except Exception as e:
        # st.error(f"Error general en optimización: {e}") # REMOVED
        return None, None, None, None, None