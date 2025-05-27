# src/visualization.py

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_performance_normalized(plot_data_normalized, benchmark_col_used):
    """Genera el gráfico de evolución normalizada de la cartera y activos."""
    if plot_data_normalized is not None and not plot_data_normalized.empty:
        plot_data_to_show = plot_data_normalized.dropna(axis=1, how='all')
        if not plot_data_to_show.empty:
            fig_perf = px.line(plot_data_to_show, title="Cartera vs. Activos" + (f" vs {benchmark_col_used}" if benchmark_col_used else ""), labels={'value': 'Valor (Base 100)', 'variable': 'Activo'})
            fig_perf.update_layout(xaxis_title="Fecha", yaxis_title="Valor Normalizado", legend_title_text='Activos')
            return fig_perf
    return None

def plot_rolling_metrics(rolling_data, title, yaxis_title):
    """Genera un gráfico de línea para métricas rodantes."""
    if rolling_data is not None and not rolling_data.empty:
        fig = px.line(rolling_data, title=title, labels={'value': yaxis_title})
        fig.update_layout(showlegend=False, yaxis_title=yaxis_title)
        return fig
    return None

def plot_correlation_heatmap(corr_matrix):
    """Genera un mapa de calor de la matriz de correlación."""
    if corr_matrix is not None and not corr_matrix.empty:
        if corr_matrix.shape[0] > 1 and corr_matrix.shape[1] > 1:
            fig_heatmap, ax = plt.subplots(figsize=(max(6, corr_matrix.shape[1]*0.8), max(5, corr_matrix.shape[0]*0.7)))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax, vmin=-1, vmax=1, annot_kws={"size": 8})
            ax.set_title('Correlación entre Activos'); plt.xticks(rotation=45, ha='right', fontsize=8); plt.yticks(rotation=0, fontsize=8)
            plt.tight_layout()
            return fig_heatmap
    return None

def plot_risk_return_scatter(individual_metrics, portfolio_metrics):
    """Genera un gráfico de dispersión de riesgo vs retorno."""
    if individual_metrics is not None and not individual_metrics.empty and portfolio_metrics:
        scatter_data = individual_metrics.T.copy()
        scatter_data['Rentab. Anual (CAGR)'] = pd.to_numeric(scatter_data.get('Rentabilidad Anualizada (CAGR)'), errors='coerce')
        scatter_data['Volatilidad Anualizada'] = pd.to_numeric(scatter_data.get('Volatilidad Anualizada'), errors='coerce')
        valid_cols_scatter = ['Rentab. Anual (CAGR)', 'Volatilidad Anualizada']
        if all(col in scatter_data.columns for col in valid_cols_scatter):
            scatter_data.dropna(subset=valid_cols_scatter, inplace=True)

        portfolio_scatter = pd.DataFrame({
            'Rentab. Anual (CAGR)': [portfolio_metrics.get('Rentabilidad Anualizada (CAGR)')],
            'Volatilidad Anualizada': [portfolio_metrics.get('Volatilidad Anualizada')]},
            index=['Cartera Total'])
        portfolio_scatter['Rentab. Anual (CAGR)'] = pd.to_numeric(portfolio_scatter['Rentab. Anual (CAGR)'], errors='coerce')
        portfolio_scatter['Volatilidad Anualizada'] = pd.to_numeric(portfolio_scatter['Volatilidad Anualizada'], errors='coerce')
        portfolio_scatter.dropna(inplace=True)

        if not scatter_data.empty or not portfolio_scatter.empty:
            scatter_data_final = pd.concat([scatter_data.assign(Tipo='Activo Individual'), portfolio_scatter.assign(Tipo='Cartera Total')])
            if not scatter_data_final.empty and all(col in scatter_data_final.columns for col in valid_cols_scatter):
                fig = px.scatter(scatter_data_final, x='Volatilidad Anualizada', y='Rentab. Anual (CAGR)', text=scatter_data_final.index, color='Tipo', hover_name=scatter_data_final.index, title="Rentabilidad vs. Volatilidad")
                fig.update_traces(textposition='top center'); fig.update_layout(xaxis_title="Volatilidad Anualizada", yaxis_title="Rentab. Anual (CAGR)", xaxis_tickformat=".1%", yaxis_tickformat=".1%")
                return fig
    return None

def plot_risk_contribution_bar(risk_contribution):
    """Genera un gráfico de barras de la contribución al riesgo."""
    if risk_contribution is not None and not risk_contribution.empty:
        risk_contribution_pct = (risk_contribution * 100).reset_index(); risk_contribution_pct.columns = ['Activo', 'Contribución al Riesgo (%)']
        fig = px.bar(risk_contribution_pct.sort_values(by='Contribución al Riesgo (%)', ascending=False), x='Activo', y='Contribución al Riesgo (%)', title='Contribución Porcentual al Riesgo Total', text='Contribución al Riesgo (%)')
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside'); fig.update_layout(yaxis_title="Contribución (%)", xaxis_title="Activo")
        return fig
    return None

def plot_efficient_frontier(frontier_df, mvp_performance, msr_performance, current_portfolio_performance_opt):
    """Genera el gráfico de la frontera eficiente."""
    if frontier_df is not None and not frontier_df.empty:
        fig_frontier = go.Figure()
        fig_frontier.add_trace(go.Scatter(x=frontier_df['Volatility'], y=frontier_df['Return'], mode='lines', name='Frontera Eficiente', line=dict(color='blue', width=2)))
        if mvp_performance:
            fig_frontier.add_trace(go.Scatter(x=[mvp_performance['annual_volatility']], y=[mvp_performance['expected_return']], mode='markers+text', marker=dict(color='green', size=12, symbol='star'), text="MVP", textposition="bottom center", name=f"Min Varianza (Sharpe: {mvp_performance['sharpe_ratio']:.2f})"))
        if msr_performance:
            fig_frontier.add_trace(go.Scatter(x=[msr_performance['annual_volatility']], y=[msr_performance['expected_return']], mode='markers+text', marker=dict(color='red', size=12, symbol='star'), text="MSR", textposition="bottom center", name=f"Max Sharpe (Sharpe: {msr_performance['sharpe_ratio']:.2f})"))
        if current_portfolio_performance_opt:
            fig_frontier.add_trace(go.Scatter(x=[current_portfolio_performance_opt['annual_volatility']], y=[current_portfolio_performance_opt['expected_return']], mode='markers+text', marker=dict(color='orange', size=12, symbol='circle'), text="Actual", textposition="bottom center", name=f"Tu Cartera (Sharpe: {current_portfolio_performance_opt['sharpe_ratio']:.2f})"))
        fig_frontier.update_layout(title='Frontera Eficiente y Carteras Óptimas', xaxis_title='Volatilidad Anualizada (Riesgo)', yaxis_title='Rentabilidad Esperada Anualizada', xaxis_tickformat=".1%", yaxis_tickformat=".1%", legend_title_text='Carteras', height=500)
        return fig_frontier
    return None

def plot_monte_carlo_simulations(mc_simulations_df, mc_summary, mc_projection_days, mc_num_simulations):
    """Genera el gráfico de la simulación de Montecarlo."""
    if mc_simulations_df is not None and not mc_simulations_df.empty and mc_summary:
        fig_mc = go.Figure()
        num_lines_to_plot = min(100, mc_num_simulations)
        for col in mc_simulations_df.columns[:num_lines_to_plot]:
            fig_mc.add_trace(go.Scatter(x=mc_simulations_df.index, y=mc_simulations_df[col], mode='lines', line=dict(width=0.5, color='lightblue'), showlegend=False))

        fig_mc.add_trace(go.Scatter(x=mc_simulations_df.index, y=mc_simulations_df.mean(axis=1), mode='lines', line=dict(width=2, color='blue'), name='Media'))
        fig_mc.add_trace(go.Scatter(x=mc_simulations_df.index, y=mc_simulations_df.quantile(0.05, axis=1), mode='lines', line=dict(width=1, color='red', dash='dash'), name='Percentil 5%'))
        fig_mc.add_trace(go.Scatter(x=mc_simulations_df.index, y=mc_simulations_df.quantile(0.95, axis=1), mode='lines', line=dict(width=1, color='green', dash='dash'), name='Percentil 95%'))

        fig_mc.update_layout(title="Distribución de Valores Futuros de Cartera (Montecarlo)", xaxis_title="Fecha Proyectada", yaxis_title="Valor de Cartera (€)", height=500)
        return fig_mc
    return None

def plot_historical_scenario_evolution(scenario_portfolio_evolution, scenario_name):
    """Genera el gráfico de evolución de la cartera en un escenario histórico."""
    if scenario_portfolio_evolution is not None and not scenario_portfolio_evolution.empty:
        evo_df = scenario_portfolio_evolution
        fig_scenario = px.line(evo_df / evo_df.iloc[0] * 100, title=f"Evolución Normalizada en Escenario: {scenario_name}")
        fig_scenario.update_layout(xaxis_title="Fecha del Escenario", yaxis_title="Valor Normalizado (Base 100)", showlegend=False)
        return fig_scenario
    return None