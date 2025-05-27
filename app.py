# app.py

import streamlit as st
import pandas as pd
import numpy as np
import locale

# Importar funciones de los módulos separados
from src.constants import RISK_FREE_RATE_DEFAULT, CONFIDENCE_LEVEL_VAR_DEFAULT, HISTORICAL_SCENARIOS
from src.utils import load_data, calculate_returns, calculate_sortino_ratio
from src.simulation import run_backtest
from src.analysis import (
    calculate_metrics, calculate_individual_metrics, calculate_correlation_matrix,
    calculate_rolling_correlation, calculate_risk_contribution, calculate_rolling_metrics,
    calculate_benchmark_metrics, calculate_rolling_beta, calculate_diversification_ratio
)
from src.models import (
    calculate_var_historical, calculate_es_historical, calculate_var_parametric,
    calculate_es_parametric, apply_hypothetical_shock, analyze_historical_scenario,
    run_monte_carlo_simulation
)
from src.optimization import optimize_portfolio
from src.visualization import (
    plot_performance_normalized, plot_rolling_metrics, plot_correlation_heatmap,
    plot_risk_return_scatter, plot_risk_contribution_bar, plot_efficient_frontier,
    plot_monte_carlo_simulations, plot_historical_scenario_evolution
)

# --- Configuración de la Página y Estilo ---
st.set_page_config(
    page_title="Backtester Quant v5.0 (Extendido)",
    page_icon="🚀",
    layout="wide"
)

# --- Interfaz de Streamlit ---
st.title("🚀 Backtester Quant v5.0 (Funcionalidades Extendidas)")
st.markdown("""
Sube tu archivo CSV con los precios históricos de tus fondos **y tu benchmark** para analizar, optimizar tu cartera y realizar análisis de riesgo avanzados.
**Formato esperado:** 1ª col Fecha, siguientes ISINs/Tickers y Benchmark; sep CSV (',' o ';'); decimal (',' o '.').
""")

# --- Barra Lateral ---
st.sidebar.header("⚙️ Configuración General")
uploaded_file = st.sidebar.file_uploader("1. Carga tu archivo CSV (con Benchmark)", type=["csv"])

# --- Estado de Sesión ---
default_session_state = {
    'data': None, 'weights': {}, 'run_results': None, 'last_uploaded_id': None,
    'benchmark_col': None, 'rolling_window': 60, 'risk_free_rate': RISK_FREE_RATE_DEFAULT,
    'fixed_income_selection': [], 'money_market_selection': [],
    'transaction_cost_bps': 0.0,
    'var_confidence_level': CONFIDENCE_LEVEL_VAR_DEFAULT,
    'var_dist_type': 'histórica',
    'hypothetical_shock_pct': -10.0,
    'selected_historical_scenario': "Ninguno",
    'mc_num_simulations': 1000,
    'mc_projection_days': 252,
}
for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Carga y Selección de Benchmark y Categorías de Activos ---
benchmark_col = None
asset_columns_for_selection = []

if uploaded_file is not None:
    uploaded_file_id = uploaded_file.name + str(uploaded_file.size)
    if st.session_state.last_uploaded_id != uploaded_file_id:
        st.session_state['data'] = load_data(uploaded_file)
        st.session_state.last_uploaded_id = uploaded_file_id
        st.session_state['run_results'] = None
        st.session_state['benchmark_col'] = None
        st.session_state['fixed_income_selection'] = []
        st.session_state['money_market_selection'] = []
        if st.session_state['data'] is not None:
            temp_cols = st.session_state['data'].columns.tolist()
            common_benchmarks = ['^GSPC', 'SPY', 'IBEX', '^IBEX', 'benchmark', 'indice', 'Benchmark', 'Index', 'MSCI World']
            guessed_bm = next((col for col in temp_cols if col in common_benchmarks or col.lower() in [bm.lower() for bm in common_benchmarks]), None)
            st.session_state['weights'] = {fund: 0.0 for fund in temp_cols if fund != guessed_bm}
        else: st.session_state['weights'] = {}
data = st.session_state['data']

if data is not None:
    st.sidebar.success(f"Archivo '{uploaded_file.name}' cargado.")
    st.sidebar.markdown("---")
    available_columns = data.columns.tolist()

    common_benchmarks_ui = ['^GSPC', 'SPY', 'IBEX', '^IBEX', 'benchmark', 'indice', 'Benchmark', 'Index', 'MSCI World']
    detected_benchmark_index = 0
    saved_benchmark = st.session_state.get('benchmark_col')
    if saved_benchmark and saved_benchmark in available_columns:
        detected_benchmark_index = available_columns.index(saved_benchmark)
    else:
        for i, col in enumerate(available_columns):
            if col in common_benchmarks_ui or col.lower() in [bm.lower() for bm in common_benchmarks_ui]:
                detected_benchmark_index = i; break
    st.session_state['benchmark_col'] = st.sidebar.selectbox("2. Selecciona la Columna Benchmark", options=available_columns, index=detected_benchmark_index, key='benchmark_selector')
    benchmark_col = st.session_state['benchmark_col']

    asset_columns_for_selection = [col for col in available_columns if col != benchmark_col]

    st.sidebar.markdown("---")
    st.sidebar.subheader("2a. Activos para Restricciones de Optimización")
    st.session_state['fixed_income_selection'] = st.sidebar.multiselect(
        "Activos de Renta Fija (Máx. 9% total en optimización)",
        options=asset_columns_for_selection, default=st.session_state.get('fixed_income_selection', []),
        help="Selecciona los ISINs/activos que son de Renta Fija."
    )
    st.session_state['money_market_selection'] = st.sidebar.multiselect(
        "Activos Monetarios (Máx. 1% total en optimización)",
        options=asset_columns_for_selection, default=st.session_state.get('money_market_selection', []),
        help="Selecciona los ISINs/activos que son Monetarios."
    )
    st.sidebar.markdown("---")

    st.sidebar.subheader("Parámetros del Backtest")
    min_date, max_date = data.index.min().date(), data.index.max().date()
    start_date = st.sidebar.date_input("3. Fecha de Inicio", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("4. Fecha de Fin", max_date, min_value=start_date, max_value=max_date)
    initial_investment = st.sidebar.number_input("5. Inversión Inicial (€)", min_value=1, value=10000, step=100)
    rebalance_freq = st.sidebar.selectbox("6. Frecuencia de Rebalanceo", ['No Rebalancear', 'Mensual', 'Trimestral', 'Anual'])
    st.session_state['transaction_cost_bps'] = st.sidebar.number_input("Coste Transacción (pb por operación)", min_value=0.0, value=st.session_state.get('transaction_cost_bps', 0.0), step=0.5, format="%.2f")
    st.sidebar.markdown("---")

    st.sidebar.subheader("Parámetros de Análisis")
    try:
        filtered_len = len(data.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)])
        max_window_val = max(20, filtered_len - 5) if filtered_len > 25 else 20
    except Exception: max_window_val = 252*3
    default_window_val = min(st.session_state.get('rolling_window', 60), max_window_val) if max_window_val >= 20 else 20
    st.session_state['rolling_window'] = st.sidebar.number_input("7. Ventana Análisis Rodante (días)", min_value=20, max_value=max_window_val, value=default_window_val, step=10)
    st.session_state['risk_free_rate'] = st.sidebar.number_input("8. Tasa Libre de Riesgo Anual (%)", min_value=-10.0, max_value=25.0, value=st.session_state.get('risk_free_rate', RISK_FREE_RATE_DEFAULT) * 100 , step=0.1, format="%.2f") / 100.0
    st.sidebar.markdown("---")

    st.sidebar.subheader("Análisis de Riesgo Avanzado y Escenarios")
    st.session_state['var_confidence_level'] = st.sidebar.slider("Nivel Confianza VaR/ES (%)", min_value=90.0, max_value=99.9, value=st.session_state.get('var_confidence_level', CONFIDENCE_LEVEL_VAR_DEFAULT)*100, step=0.1, format="%.1f") / 100.0
    st.session_state['var_dist_type'] = st.sidebar.selectbox("Distribución VaR/ES Paramétrico", ['normal', 't', 'histórica'], index=['histórica', 'normal', 't'].index(st.session_state.get('var_dist_type', 'histórica')))
    st.session_state['hypothetical_shock_pct'] = st.sidebar.number_input("Shock Hipotético de Mercado (%)", min_value=-99.0, max_value=50.0, value=st.session_state.get('hypothetical_shock_pct', -10.0), step=1.0, format="%.1f")
    st.session_state['selected_historical_scenario'] = st.sidebar.selectbox("Escenario Histórico de Crisis", options=list(HISTORICAL_SCENARIOS.keys()), index=list(HISTORICAL_SCENARIOS.keys()).index(st.session_state.get('selected_historical_scenario', "Ninguno")))
    st.sidebar.markdown("---")

    st.sidebar.subheader("Simulación Montecarlo (Proyección Futura)")
    st.session_state['mc_num_simulations'] = st.sidebar.number_input("Nº Simulaciones Montecarlo", min_value=100, max_value=10000, value=st.session_state.get('mc_num_simulations', 1000), step=100)
    st.session_state['mc_projection_days'] = st.sidebar.number_input("Días de Proyección Montecarlo", min_value=20, max_value=252*5, value=st.session_state.get('mc_projection_days', 252), step=10)
    st.sidebar.markdown("---")

    st.sidebar.subheader("9. Asignación de Pesos (%) [Excluye Benchmark]")
    with st.sidebar.form(key='weights_form'):
        weights_input = {}
        if not asset_columns_for_selection:
            st.sidebar.warning("No hay activos disponibles para asignar pesos (después de excluir benchmark).")
        for fund in asset_columns_for_selection:
            default_weight = st.session_state['weights'].get(fund, 0.0) * 100 if fund in st.session_state['weights'] else 0.0
            weights_input[fund] = st.number_input(f"{fund}", min_value=0.0, max_value=100.0, value=default_weight, step=1.0, format="%.2f") / 100.0

        submitted = st.form_submit_button("🚀 Ejecutar Análisis Completo")

        if submitted:
            st.session_state['weights'] = weights_input
            total_input_weight_sum = sum(weights_input.values())
            weights_norm = {}
            if not np.isclose(total_input_weight_sum, 1.0, atol=0.01) and total_input_weight_sum > 1e-6 :
                st.sidebar.warning(f"Pesos suman {total_input_weight_sum*100:.2f}%. Se normalizarán.")
                weights_norm = {k: v / total_input_weight_sum for k, v in weights_input.items()}
            elif np.isclose(total_input_weight_sum, 0.0, atol=1e-6):
                st.sidebar.error("Suma de pesos es 0%. Asigna pesos."); weights_norm = weights_input
            else:
                weights_norm = weights_input

            active_weights = {k: v for k, v in weights_norm.items() if v > 1e-6}
            if not active_weights:
                st.error("No hay fondos con peso > 0 asignado.")
                st.session_state['run_results'] = None
            else:
                funds_in_portfolio = list(active_weights.keys())
                cols_needed_for_backtest = funds_in_portfolio

                all_cols_in_range = data.columns.tolist()
                data_full_range_unfiltered = data.loc[pd.to_datetime(start_date):pd.to_datetime(end_date), all_cols_in_range].copy()
                data_full_range_unfiltered.ffill(inplace=True); data_full_range_unfiltered.bfill(inplace=True)

                asset_data_for_backtest = data.loc[pd.to_datetime(start_date):pd.to_datetime(end_date), cols_needed_for_backtest].copy()
                asset_data_for_backtest.ffill(inplace=True); asset_data_for_backtest.bfill(inplace=True)

                if asset_data_for_backtest.empty or asset_data_for_backtest.shape[0] < 2:
                    st.error("No hay suficientes datos comunes para los activos de la cartera en el rango seleccionado tras limpiar NaNs.")
                    st.session_state['run_results'] = None
                elif benchmark_col and benchmark_col not in data_full_range_unfiltered.columns:
                    st.error(f"Benchmark '{benchmark_col}' no encontrado en los datos del rango seleccionado.")
                    st.session_state['run_results'] = None
                else:
                    with st.spinner("Realizando cálculos avanzados y optimización..."):

                        portfolio_value, portfolio_returns, total_tx_costs = run_backtest(
                            asset_data_for_backtest, active_weights, initial_investment,
                            asset_data_for_backtest.index.min(), asset_data_for_backtest.index.max(),
                            rebalance_freq, st.session_state['transaction_cost_bps']
                        )
                        results_dict = {'run_successful': False}

                        if portfolio_value is not None and not portfolio_value.empty:
                            annual_rf_for_metrics = st.session_state['risk_free_rate'] # Tasa anual
                            daily_rf_for_sortino_calc = (1 + annual_rf_for_metrics)**(1/252) - 1 if annual_rf_for_metrics != 0 else 0.0

                            portfolio_metrics = calculate_metrics(portfolio_value, portfolio_returns, annual_rf_for_metrics)
                            portfolio_metrics['Sortino Ratio'] = calculate_sortino_ratio(portfolio_returns, daily_required_return_for_empyrical=daily_rf_for_sortino_calc)
                            portfolio_metrics['Costes Totales Transacción (€)'] = total_tx_costs
                            portfolio_metrics['Costes Totales Transacción (%)'] = (total_tx_costs / initial_investment) if initial_investment > 0 else 0.0

                            asset_data_analysis = data_full_range_unfiltered[funds_in_portfolio].loc[portfolio_value.index]
                            individual_metrics = calculate_individual_metrics(asset_data_analysis, annual_rf_for_metrics)
                            asset_returns = calculate_returns(asset_data_analysis)

                            benchmark_data_analysis = None
                            benchmark_returns = None
                            individual_asset_benchmark_metrics = pd.DataFrame(index=funds_in_portfolio, columns=['Beta', 'Alpha (anual)'])
                            rolling_beta_portfolio = None
                            if benchmark_col and benchmark_col in data_full_range_unfiltered.columns:
                                benchmark_data_analysis = data_full_range_unfiltered[benchmark_col].loc[portfolio_value.index]
                                benchmark_returns = calculate_returns(benchmark_data_analysis)
                                if benchmark_returns is not None and not benchmark_returns.empty:
                                    common_idx_bm = portfolio_returns.index.intersection(benchmark_returns.index)
                                    if len(common_idx_bm) >= max(2, st.session_state['rolling_window']):
                                        portfolio_returns_aligned_bm = portfolio_returns.loc[common_idx_bm]
                                        benchmark_returns_aligned_bm = benchmark_returns.loc[common_idx_bm]

                                        benchmark_metrics_calc = calculate_benchmark_metrics(portfolio_returns_aligned_bm, benchmark_returns_aligned_bm, annual_rf_for_metrics)
                                        portfolio_metrics.update(benchmark_metrics_calc)

                                        temp_asset_bm_metrics = {}
                                        if asset_returns is not None:
                                            for fund in funds_in_portfolio:
                                                if fund in asset_returns.columns:
                                                    fund_ret_aligned_bm = asset_returns[fund].loc[common_idx_bm]
                                                    if len(fund_ret_aligned_bm.dropna()) >= 2:
                                                        metrics_bm_ind = calculate_benchmark_metrics(fund_ret_aligned_bm, benchmark_returns_aligned_bm, annual_rf_for_metrics)
                                                        temp_asset_bm_metrics[fund] = {'Beta': metrics_bm_ind.get('Beta'), 'Alpha (anual)': metrics_bm_ind.get('Alpha (anual)')}
                                                    else: temp_asset_bm_metrics[fund] = {'Beta': np.nan, 'Alpha (anual)': np.nan}
                                                else: temp_asset_bm_metrics[fund] = {'Beta': np.nan, 'Alpha (anual)': np.nan}
                                            individual_asset_benchmark_metrics.update(pd.DataFrame(temp_asset_bm_metrics).T)

                                        rolling_beta_portfolio = calculate_rolling_beta(portfolio_returns_aligned_bm, benchmark_returns_aligned_bm, st.session_state['rolling_window'])
                                    else: st.warning("Insuficientes datos comunes con benchmark para algunas métricas relativas.")
                                else: st.warning("No se pudieron calcular retornos del benchmark.")

                            normalized_individual_prices = pd.DataFrame()
                            if not asset_data_analysis.empty and (asset_data_analysis.iloc[0].abs() > 1e-9).all():
                                normalized_individual_prices = asset_data_analysis / asset_data_analysis.iloc[0] * 100

                            normalized_portfolio_value = portfolio_value / portfolio_value.iloc[0] * 100
                            normalized_portfolio_value.name = "Cartera Total"

                            normalized_benchmark = None
                            if benchmark_data_analysis is not None and not benchmark_data_analysis.empty:
                                first_bm_val = benchmark_data_analysis.iloc[0]
                                if pd.notna(first_bm_val) and abs(first_bm_val) > 1e-9:
                                    normalized_benchmark = benchmark_data_analysis / first_bm_val * 100
                                    normalized_benchmark.name = benchmark_col

                            plot_data_normalized = pd.concat([normalized_portfolio_value, normalized_individual_prices], axis=1)
                            if normalized_benchmark is not None:
                                plot_data_normalized = pd.concat([plot_data_normalized, normalized_benchmark], axis=1)

                            corr_matrix = calculate_correlation_matrix(asset_returns)
                            avg_rolling_corr, pair_rolling_corr = calculate_rolling_correlation(asset_returns, window=st.session_state['rolling_window'])
                            risk_contribution = calculate_risk_contribution(asset_returns, active_weights)
                            diversification_ratio = calculate_diversification_ratio(asset_returns, active_weights)
                            portfolio_metrics['Diversification Ratio'] = diversification_ratio
                            rolling_vol, rolling_sharpe, rolling_sortino = calculate_rolling_metrics(portfolio_returns, st.session_state['rolling_window'], annual_rf_for_metrics)

                            var_hist = calculate_var_historical(portfolio_returns, st.session_state['var_confidence_level'])
                            es_hist = calculate_es_historical(portfolio_returns, st.session_state['var_confidence_level'])
                            var_param_dist_type = st.session_state['var_dist_type'] if st.session_state['var_dist_type'] != 'histórica' else 'normal'
                            var_param = calculate_var_parametric(portfolio_returns, st.session_state['var_confidence_level'], dist_type=var_param_dist_type)
                            es_param = calculate_es_parametric(portfolio_returns, st.session_state['var_confidence_level'], dist_type=var_param_dist_type)

                            advanced_risk_metrics = {
                                f"VaR Histórico ({st.session_state['var_confidence_level']:.1%}) Diario": var_hist,
                                f"ES Histórico ({st.session_state['var_confidence_level']:.1%}) Diario": es_hist,
                                f"VaR Paramétrico ({var_param_dist_type}, {st.session_state['var_confidence_level']:.1%}) Diario": var_param,
                                f"ES Paramétrico ({var_param_dist_type}, {st.session_state['var_confidence_level']:.1%}) Diario": es_param,
                            }

                            stress_test_results = {}
                            shocked_val, shock_loss = apply_hypothetical_shock(portfolio_value, st.session_state['hypothetical_shock_pct'])
                            stress_test_results['hypothetical_shock'] = {
                                'shock_pct': st.session_state['hypothetical_shock_pct'],
                                'valor_cartera_post_shock': shocked_val,
                                'perdida_absoluta_shock': shock_loss,
                                'ultimo_valor_cartera': portfolio_value.iloc[-1] if not portfolio_value.empty else np.nan
                            }
                            scenario_dates_selected = HISTORICAL_SCENARIOS.get(st.session_state['selected_historical_scenario'])
                            scenario_portfolio_evolution = None
                            scenario_metrics = {}
                            if scenario_dates_selected:
                                scenario_portfolio_evolution, scenario_metrics = analyze_historical_scenario(
                                    portfolio_returns,
                                    data,
                                    active_weights,
                                    scenario_dates_selected,
                                    initial_investment_for_scenario=initial_investment
                                )
                            stress_test_results['historical_scenario'] = {
                                'scenario_name': st.session_state['selected_historical_scenario'],
                                'evolution': scenario_portfolio_evolution,
                                'metrics': scenario_metrics
                            }

                            mc_simulations_df, mc_summary = run_monte_carlo_simulation(
                                portfolio_returns,
                                portfolio_value.iloc[-1] if not portfolio_value.empty else initial_investment,
                                st.session_state['mc_num_simulations'],
                                st.session_state['mc_projection_days']
                            )

                            mvp_weights, msr_weights, mvp_performance, msr_performance, frontier_df = optimize_portfolio(
                                asset_data_analysis,
                                risk_free_rate=annual_rf_for_metrics,
                                fixed_income_assets=st.session_state['fixed_income_selection'],
                                money_market_assets=st.session_state['money_market_selection']
                            )

                            current_portfolio_performance_opt = None
                            if active_weights and not asset_data_analysis.empty:
                                try:
                                    prices_cleaned_opt = asset_data_analysis.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='any')
                                    if not prices_cleaned_opt.empty and prices_cleaned_opt.shape[1] >=1 and prices_cleaned_opt.shape[0] >= 20:
                                        mu_current = expected_returns.mean_historical_return(prices_cleaned_opt, frequency=252)
                                        S_current = risk_models.CovarianceShrinkage(prices_cleaned_opt, frequency=252).ledoit_wolf()

                                        weights_array = np.array([active_weights.get(asset, 0) for asset in prices_cleaned_opt.columns])
                                        weights_sum = weights_array.sum()
                                        weights_array_norm = weights_array / weights_sum if weights_sum > 1e-9 else weights_array

                                        current_ret = pypfopt.objective_functions.portfolio_return(weights_array_norm, mu_current, negative=False)
                                        current_variance = pypfopt.objective_functions.portfolio_variance(weights_array_norm, S_current)
                                        current_vol = np.sqrt(current_variance) if current_variance >= 0 else np.nan
                                        current_sharpe = (current_ret - annual_rf_for_metrics) / current_vol if pd.notna(current_vol) and current_vol > 1e-9 else np.nan
                                        current_portfolio_performance_opt = {'expected_return': current_ret, 'annual_volatility': current_vol, 'sharpe_ratio': current_sharpe}
                                except Exception: pass


                            results_dict = {
                                'portfolio_value': portfolio_value, 'portfolio_metrics': portfolio_metrics, 'plot_data_normalized': plot_data_normalized,
                                'individual_metrics': individual_metrics, 'individual_asset_benchmark_metrics': individual_asset_benchmark_metrics,
                                'asset_returns': asset_returns, 'benchmark_returns': benchmark_returns, 'corr_matrix': corr_matrix,
                                'avg_rolling_corr': avg_rolling_corr, 'pair_rolling_corr': pair_rolling_corr, 'risk_contribution': risk_contribution,
                                'rolling_vol': rolling_vol, 'rolling_sharpe': rolling_sharpe, 'rolling_sortino': rolling_sortino,
                                'rolling_beta_portfolio': rolling_beta_portfolio, 'weights': active_weights, 'rolling_window_used': st.session_state['rolling_window'],
                                'benchmark_col_used': benchmark_col if benchmark_data_analysis is not None and benchmark_returns is not None else None,
                                'advanced_risk_metrics': advanced_risk_metrics,
                                'stress_test_results': stress_test_results,
                                'mc_simulations_df': mc_simulations_df,
                                'mc_summary': mc_summary,
                                'mvp_weights': mvp_weights, 'msr_weights': msr_weights, 'mvp_performance': mvp_performance,
                                'msr_performance': msr_performance, 'frontier_df': frontier_df, 'current_portfolio_performance_opt': current_portfolio_performance_opt,
                                'run_successful': True
                            }
                            st.session_state['run_results'] = results_dict
                            st.sidebar.success("Análisis y optimización completados.")
                        else:
                            st.error("La simulación del backtest no produjo resultados válidos.")
                            st.session_state['run_results'] = {'run_successful': False}

# --- Mostrar Resultados (Pestañas) ---
if 'run_results' in st.session_state and st.session_state['run_results'] is not None and st.session_state['run_results'].get('run_successful', False):
    results = st.session_state['run_results']
    benchmark_col_used = results.get('benchmark_col_used')
    portfolio_value = results.get('portfolio_value')
    portfolio_metrics = results.get('portfolio_metrics', {})
    plot_data_normalized = results.get('plot_data_normalized')
    individual_metrics = results.get('individual_metrics', pd.DataFrame())
    individual_asset_benchmark_metrics = results.get('individual_asset_benchmark_metrics', pd.DataFrame())
    benchmark_returns = results.get('benchmark_returns')
    corr_matrix = results.get('corr_matrix')
    avg_rolling_corr = results.get('avg_rolling_corr')
    risk_contribution = results.get('risk_contribution', pd.Series(dtype=float))
    rolling_vol, rolling_sharpe, rolling_sortino = results.get('rolling_vol'), results.get('rolling_sharpe'), results.get('rolling_sortino')
    rolling_beta_portfolio = results.get('rolling_beta_portfolio')
    rolling_window_used = results.get('rolling_window_used', 0)

    advanced_risk_metrics = results.get('advanced_risk_metrics', {})
    stress_test_results = results.get('stress_test_results', {})
    mc_simulations_df = results.get('mc_simulations_df')
    mc_summary = results.get('mc_summary')

    mvp_weights, msr_weights = results.get('mvp_weights'), results.get('msr_weights')
    mvp_performance, msr_performance = results.get('mvp_performance'), results.get('msr_performance')
    frontier_df = results.get('frontier_df')
    current_portfolio_performance_opt = results.get('current_portfolio_performance_opt')

    tabs_titles = ["📊 Visión General", "🔗 Correlación", "🧩 Activos y Riesgo", "🔬 Optimización", "⚠️ Riesgo Avanzado", "📉 Estrés y Escenarios", "🎰 Montecarlo"]
    if benchmark_col_used and benchmark_returns is not None:
        tabs_titles.insert(1, f"🆚 vs {benchmark_col_used}")
        tabs = st.tabs(tabs_titles)
        tab_vg, tab_bm, tab_corr, tab_ar, tab_opt, tab_ra, tab_stress, tab_mc = tabs[0], tabs[1], tabs[2], tabs[3], tabs[4], tabs[5], tabs[6], tabs[7]
    else:
        tabs = st.tabs(tabs_titles)
        tab_vg, tab_corr, tab_ar, tab_opt, tab_ra, tab_stress, tab_mc = tabs[0], tabs[1], tabs[2], tabs[3], tabs[4], tabs[5], tabs[6]
        tab_bm = None

    with tab_vg:
        st.header("Visión General de la Cartera")
        st.subheader("Métricas Principales (Cartera Total)")
        try: locale.setlocale(locale.LC_ALL, 'es_ES.UTF-8')
        except locale.Error: locale.setlocale(locale.LC_ALL, '')

        col1, col2, col3 = st.columns(3)
        col1.metric("Rentabilidad Total", f"{portfolio_metrics.get('Rentabilidad Total', np.nan):.2%}")
        col2.metric("Rentab. Anual (CAGR)", f"{portfolio_metrics.get('Rentabilidad Anualizada (CAGR)', np.nan):.2%}")
        col3.metric("Volatilidad Anualizada", f"{portfolio_metrics.get('Volatilidad Anualizada', np.nan):.2%}")

        col1b, col2b, col3b = st.columns(3)
        col1b.metric("Máximo Drawdown", f"{portfolio_metrics.get('Máximo Drawdown', np.nan):.2%}")
        col2b.metric("Ratio de Sharpe", f"{portfolio_metrics.get('Ratio de Sharpe', np.nan):.2f}")
        col3b.metric("Ratio Sortino", f"{portfolio_metrics.get('Sortino Ratio', np.nan):.2f}")

        col1c, col2c, col3c = st.columns(3)
        col1c.metric("Ratio Diversificación", f"{portfolio_metrics.get('Diversification Ratio', np.nan):.2f}", help="Vol. media pond. activos / Vol. Cartera")
        col2c.metric("Costes Transacción (€)", f"{portfolio_metrics.get('Costes Totales Transacción (€)', 0.0):,.2f}")
        col3c.metric("Costes Transacción (%)", f"{portfolio_metrics.get('Costes Totales Transacción (%)', 0.0):.2%}", help="% sobre inversión inicial")

        st.markdown("---"); st.subheader("Evolución Normalizada (Base 100)")
        fig_perf = plot_performance_normalized(plot_data_normalized, benchmark_col_used)
        if fig_perf:
            st.plotly_chart(fig_perf, use_container_width=True)
        else: st.warning("No hay datos para gráfico de evolución.")

        st.markdown("---"); st.subheader(f"Análisis Rodante de la Cartera (Ventana: {rolling_window_used} días)")
        col_roll1, col_roll2, col_roll3 = st.columns(3)
        with col_roll1:
            fig_vol = plot_rolling_metrics(rolling_vol, "Volatilidad Anualizada Rodante", "Vol. Anual.")
            if fig_vol: st.plotly_chart(fig_vol, use_container_width=True)
        with col_roll2:
            fig_sharpe = plot_rolling_metrics(rolling_sharpe, "Ratio de Sharpe Rodante", "Sharpe")
            if fig_sharpe: st.plotly_chart(fig_sharpe, use_container_width=True)
        with col_roll3:
            fig_sortino = plot_rolling_metrics(rolling_sortino, "Ratio de Sortino Rodante", "Sortino")
            if fig_sortino: st.plotly_chart(fig_sortino, use_container_width=True)

    if tab_bm:
        with tab_bm:
            st.header(f"Comparativa vs Benchmark ({benchmark_col_used})")
            st.subheader("Métricas Relativas al Benchmark")
            col_bm1, col_bm2, col_bm3, col_bm4 = st.columns(4)
            col_bm1.metric("Beta (β)", f"{portfolio_metrics.get('Beta', np.nan):.2f}")
            col_bm2.metric("Alpha (α) Anual", f"{portfolio_metrics.get('Alpha (anual)', np.nan):.2%}")
            col_bm3.metric("Tracking Error Anual", f"{portfolio_metrics.get('Tracking Error (anual)', np.nan):.2%}")
            col_bm4.metric("Information Ratio", f"{portfolio_metrics.get('Information Ratio', np.nan):.2f}")
            st.markdown("---")
            st.subheader(f"Beta Rodante vs {benchmark_col_used} (Ventana: {rolling_window_used} días)")
            fig_beta = plot_rolling_metrics(rolling_beta_portfolio, "Beta Rodante de la Cartera", "Beta")
            if fig_beta:
                st.plotly_chart(fig_beta, use_container_width=True)
            else: st.warning("Insuficientes datos comunes con benchmark para Beta rodante.")

    with tab_corr:
        st.header("Análisis de Correlación entre Activos")
        st.subheader("Matriz de Correlación (Período Completo)")
        fig_heatmap = plot_correlation_heatmap(corr_matrix)
        if fig_heatmap:
            st.pyplot(fig_heatmap)
        else: st.warning("No se pudo calcular matriz de correlación.")

        st.markdown("---"); st.subheader(f"Correlación Rodante (Ventana: {rolling_window_used} días)")
        fig_avg_corr = plot_rolling_metrics(avg_rolling_corr, "Correlación Promedio Rodante entre Activos", "Corr. Promedio")
        if fig_avg_corr:
            st.plotly_chart(fig_avg_corr, use_container_width=True)

            asset_list_corr = results.get('asset_returns').columns.tolist() if results.get('asset_returns') is not None else []
            if len(asset_list_corr) >= 2:
                pair_options = [(asset_list_corr[i], asset_list_corr[j]) for i in range(len(asset_list_corr)) for j in range(i + 1, len(asset_list_corr))]
                if pair_options:
                    selected_pairs = st.multiselect("Pares para correlación rodante específica:", options=pair_options, format_func=lambda p: f"{p[0]} vs {p[1]}")
                    if selected_pairs:
                        _, specific_pair_corr = calculate_rolling_correlation(results.get('asset_returns'), window=rolling_window_used, pair_list=selected_pairs)
                        if specific_pair_corr is not None and not specific_pair_corr.empty:
                            fig_pair = plot_rolling_metrics(specific_pair_corr, "Correlación Rodante (Pares Seleccionados)", "Correlación")
                            if fig_pair:
                                fig_pair.update_layout(legend_title_text='Pares') # Add legend title for pairs
                                st.plotly_chart(fig_pair, use_container_width=True)
        else: st.warning("Datos insuficientes para correlación rodante.")

    with tab_ar:
        st.header("Análisis de Activos Individuales y Riesgo de Cartera")
        st.subheader("Posicionamiento Riesgo/Retorno (Activos vs Cartera)")
        fig_scatter = plot_risk_return_scatter(individual_metrics, portfolio_metrics)
        if fig_scatter:
            st.plotly_chart(fig_scatter, use_container_width=True)
        else: st.warning("Faltan métricas para gráfico Riesgo/Retorno.")

        st.markdown("---"); st.subheader("Contribución de Activos a la Volatilidad de Cartera")
        fig_risk_contrib = plot_risk_contribution_bar(risk_contribution)
        if fig_risk_contrib:
            st.plotly_chart(fig_risk_contrib, use_container_width=True)
        else: st.warning("No se pudo calcular contribución al riesgo.")

        st.markdown("---"); st.subheader("Ranking Avanzado de Activos")
        if individual_metrics is not None and not individual_metrics.empty:
            ranking_df = individual_metrics.T.copy()
            if risk_contribution is not None and not risk_contribution.empty:
                ranking_df = ranking_df.join((risk_contribution.rename('Contribución Riesgo (%)') * 100), how='left')
            if benchmark_col_used and individual_asset_benchmark_metrics is not None and not individual_asset_benchmark_metrics.empty:
                ranking_df = ranking_df.join(individual_asset_benchmark_metrics[['Beta', 'Alpha (anual)']], how='left')

            if 'Ratio de Sharpe' in ranking_df.columns:
                ranking_df['Ratio de Sharpe'] = pd.to_numeric(ranking_df['Ratio de Sharpe'], errors='coerce')
                ranking_df.sort_values(by='Ratio de Sharpe', ascending=False, inplace=True, na_position='last')

            ranking_display = ranking_df.copy()
            fmt_pct = lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"; fmt_flt = lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"; fmt_pct_risk = lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
            cols_pct = ['Rentabilidad Total', 'Rentabilidad Anualizada (CAGR)', 'Volatilidad Anualizada', 'Máximo Drawdown', 'Alpha (anual)']
            cols_flt = ['Ratio de Sharpe', 'Sortino Ratio', 'Beta']; cols_pct_risk = ['Contribución Riesgo (%)']
            for col in cols_pct:
                if col in ranking_display.columns: ranking_display[col] = ranking_display[col].map(fmt_pct)
            for col in cols_flt:
                if col in ranking_display.columns: ranking_display[col] = ranking_display[col].map(fmt_flt)
            for col in cols_pct_risk:
                if col in ranking_display.columns: ranking_display[col] = ranking_display[col].map(fmt_pct_risk)
            st.dataframe(ranking_display, use_container_width=True)
        else: st.warning("Faltan datos para ranking avanzado.")

    with tab_opt:
        st.header("🔬 Optimización de Cartera (Frontera Eficiente)")
        st.markdown("Basado en datos históricos del periodo seleccionado. Solo posiciones largas. Las restricciones de Renta Fija/Monetario se aplican aquí.")
        fig_frontier = plot_efficient_frontier(frontier_df, mvp_performance, msr_performance, current_portfolio_performance_opt)
        if fig_frontier:
            st.plotly_chart(fig_frontier, use_container_width=True)

            st.markdown("---"); st.subheader("Pesos Sugeridos por Optimización")
            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                st.markdown("**Cartera Mínima Varianza (MVP)**")
                if mvp_weights:
                    mvp_df = pd.DataFrame.from_dict(mvp_weights, orient='index', columns=['Peso Sugerido']).rename_axis('Activo')
                    mvp_df['Peso'] = mvp_df['Peso Sugerido'].map('{:.2%}'.format)
                    st.dataframe(mvp_df.sort_values(by='Peso Sugerido', ascending=False)[['Peso']])
            with col_opt2:
                st.markdown("**Cartera Máximo Sharpe (MSR)**")
                if msr_weights:
                    msr_df = pd.DataFrame.from_dict(msr_weights, orient='index', columns=['Peso Sugerido']).rename_axis('Activo')
                    msr_df['Peso'] = msr_df['Peso Sugerido'].map('{:.2%}'.format)
                    st.dataframe(msr_df.sort_values(by='Peso Sugerido', ascending=False)[['Peso']])
        else: st.warning("No se pudo calcular frontera eficiente. Revisa datos, rango y restricciones.")

    with tab_ra:
        st.header("⚠️ Análisis de Riesgo Avanzado")
        st.subheader("Valor en Riesgo (VaR) y Expected Shortfall (ES) - Diarios")
        if advanced_risk_metrics:
            df_adv_risk = pd.DataFrame.from_dict(advanced_risk_metrics, orient='index', columns=['Valor'])
            df_adv_risk['Valor (%)'] = df_adv_risk['Valor'].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
            st.table(df_adv_risk[['Valor (%)']])
            st.caption(f"Calculado sobre los retornos diarios de la cartera durante el periodo del backtest. Nivel de confianza: {st.session_state['var_confidence_level']:.1%}.")
            if st.session_state['var_dist_type'] != 'histórica':
                st.caption(f"Paramétrico usa distribución {st.session_state['var_dist_type']}.")
        else:
            st.warning("No se pudieron calcular métricas de riesgo avanzado.")

    with tab_stress:
        st.header("📉 Pruebas de Estrés y Análisis de Escenarios")

        st.subheader(f"Shock Hipotético de Mercado ({st.session_state['hypothetical_shock_pct']:.1f}%)")
        hypo_shock_res = stress_test_results.get('hypothetical_shock', {})
        if hypo_shock_res.get('ultimo_valor_cartera') is not None and not pd.isna(hypo_shock_res['ultimo_valor_cartera']):
            st.metric(label="Último Valor de Cartera (Pre-Shock)", value=f"{hypo_shock_res['ultimo_valor_cartera']:,.2f} €")
            st.metric(label="Valor de Cartera Estimado (Post-Shock)", value=f"{hypo_shock_res['valor_cartera_post_shock']:,.2f} €",
                      delta=f"{hypo_shock_res['perdida_absoluta_shock']:,.2f} € ({st.session_state['hypothetical_shock_pct']:.1f}%)",
                      delta_color="inverse")
        else:
            st.warning("No se pudo calcular el shock hipotético (datos de cartera no disponibles).")
        st.markdown("---")

        st.subheader(f"Análisis de Escenario Histórico: {st.session_state['selected_historical_scenario']}")
        hist_scenario_res = stress_test_results.get('historical_scenario', {})
        if st.session_state['selected_historical_scenario'] == "Ninguno":
            st.info("Selecciona un escenario histórico en la barra lateral para verlo aquí.")
        elif hist_scenario_res.get('evolution') is not None and not hist_scenario_res['evolution'].empty:
            evo_df = hist_scenario_res['evolution']
            metrics_scen = hist_scenario_res.get('metrics', {})

            st.markdown(f"**Rendimiento Simulado de la Cartera Actual Durante el Escenario ({evo_df.index.min().date()} a {evo_df.index.max().date()})**")

            col_scen1, col_scen2, col_scen3 = st.columns(3)
            col_scen1.metric("Rentabilidad Total en Escenario", f"{metrics_scen.get('Rentabilidad Total', np.nan):.2%}")
            col_scen2.metric("Máximo Drawdown en Escenario", f"{metrics_scen.get('Máximo Drawdown', np.nan):.2%}")
            col_scen3.metric("Volatilidad Anual. en Escenario", f"{metrics_scen.get('Volatilidad Anualizada', np.nan):.2%}")

            fig_scenario = plot_historical_scenario_evolution(evo_df, st.session_state['selected_historical_scenario'])
            if fig_scenario:
                st.plotly_chart(fig_scenario, use_container_width=True)
        elif st.session_state['selected_historical_scenario'] != "Ninguno":
            st.warning(f"No se pudieron obtener datos para el escenario '{st.session_state['selected_historical_scenario']}'.")

    with tab_mc:
        st.header("🎰 Simulación de Montecarlo (Proyección Futura)")
        if mc_simulations_df is not None and not mc_simulations_df.empty and mc_summary:
            st.markdown(f"Proyección a **{st.session_state['mc_projection_days']} días hábiles** ({st.session_state['mc_num_simulations']} simulaciones) basada en retornos históricos de la cartera.")

            fig_mc = plot_monte_carlo_simulations(mc_simulations_df, mc_summary, st.session_state['mc_projection_days'], st.session_state['mc_num_simulations'])
            if fig_mc:
                st.plotly_chart(fig_mc, use_container_width=True)

            st.subheader("Resumen Estadístico al Final de la Proyección:")
            col_mc1, col_mc2, col_mc3 = st.columns(3)
            col_mc1.metric("Valor Medio Esperado", f"{mc_summary['mean']:,.2f} €")
            col_mc2.metric("Mediana", f"{mc_summary['median']:,.2f} €")
            col_mc3.metric("Peor 5% (Percentil 5)", f"{mc_summary['q5']:,.2f} €")
            col_mc1b, col_mc2b, col_mc3b = st.columns(3)
            col_mc1b.metric("Mejor 5% (Percentil 95)", f"{mc_summary['q95']:,.2f} €")
            col_mc2b.metric("Mínimo Simulado", f"{mc_summary['min']:,.2f} €")
            col_mc3b.metric("Máximo Simulado", f"{mc_summary['max']:,.2f} €")
        else:
            st.warning("No se pudo ejecutar la simulación de Montecarlo. Revisa los datos de retornos de la cartera.")


elif 'run_results' not in st.session_state or st.session_state['run_results'] is None:
    if data is not None: st.info("Configura parámetros y haz clic en '🚀 Ejecutar Análisis Completo'.")
    elif uploaded_file is not None: pass
    else: st.info("Por favor, carga un archivo CSV para comenzar.")
elif not st.session_state['run_results'].get('run_successful', False):
    st.error("La ejecución del análisis falló. Revisa mensajes de error o los parámetros de entrada.")

st.sidebar.markdown("---"); st.sidebar.markdown("Backtester Quant v5.0"); st.sidebar.markdown("Dios Familia y Cojones")