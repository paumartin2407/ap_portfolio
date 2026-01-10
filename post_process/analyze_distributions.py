import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURACIÓN
# ==========================================
LOGS_FILE = '../results_analysis/logs_metrics.csv'
FEATURES_FILE = '../all_features.csv'

# Compararemos SOLVER 5 (FF) vs SOLVER 8 (CG)
TARGET_SOLVERS = [5, 8]
SOLVER_NAMES = {5: "Solver 5 (FF)", 8: "Solver 8 (CG)"}

# Las features que el modelo anterior dijo que eran importantes
KEY_FEATURES = [
    'fd_sas_op_prevail_mean',      # Complejidad de precondiciones
    'sas_cg_edge_density',         # Estructura del grafo
    'lm_lm_num_reasonable_orders', # Hitos / Landmarks
    'fd_translate_seconds',        # Tamaño/Dificultad de traducción
    'heur_relaxed_plan_length'     # Profundidad estimada
]

def analyze_distributions():
    print("1. Cargando datos...")
    try:
        df_logs = pd.read_csv(os.path.join(os.path.dirname(__file__), LOGS_FILE))
        df_features = pd.read_csv(os.path.join(os.path.dirname(__file__), FEATURES_FILE))
    except FileNotFoundError:
        print("Error: Archivos no encontrados.")
        return

    # -------------------------------------------------------
    # 2. DETERMINAR EL GANADOR REAL (Calidad/Tiempo 50/50)
    # -------------------------------------------------------
    df_logs = df_logs[df_logs['solver'].isin(TARGET_SOLVERS)]
    df_solved = df_logs[df_logs['solved'] == 1].copy()
    df_solved['total_time'] = df_solved['total_time'].replace(0, 0.001)

    # Métricas del mejor caso posible (Virtual Best Solver)
    best_metrics = df_solved.groupby('problem').agg({
        'plan_cost': 'min',
        'total_time': 'min'
    }).rename(columns={'plan_cost': 'min_cost', 'total_time': 'min_time'})

    df_scored = df_solved.merge(best_metrics, on='problem')
    
    # Score 0.5 Tiempo / 0.5 Calidad
    df_scored['score'] = (0.5 * (df_scored['min_cost'] / df_scored['plan_cost'])) + \
                         (0.5 * (df_scored['min_time'] / df_scored['total_time']))

    # Elegir ganador por problema
    df_scored.sort_values(by=['problem', 'score'], ascending=[True, False], inplace=True)
    winners = df_scored.drop_duplicates(subset=['problem'], keep='first')[['problem', 'solver']]
    winners.rename(columns={'solver': 'winner'}, inplace=True)

    # Unir con features
    dataset = pd.merge(winners, df_features, left_on='problem', right_on='name')

    # -------------------------------------------------------
    # 3. ANÁLISIS ESTADÍSTICO
    # -------------------------------------------------------
    print("\n" + "="*50)
    print("ANÁLISIS COMPARATIVO: FF (5) vs CG (8)")
    print("="*50)
    
    # Agrupar por ganador y sacar la media de las features clave
    comparison = dataset.groupby('winner')[KEY_FEATURES].mean().transpose()
    comparison.columns = [SOLVER_NAMES.get(c, c) for c in comparison.columns]
    
    print("Promedio de valores según quién gana el problema:")
    print(comparison)
    
    # Diferencia porcentual
    if len(comparison.columns) == 2:
        col1 = comparison.columns[0]
        col2 = comparison.columns[1]
        comparison['Diff (%)'] = ((comparison[col2] - comparison[col1]) / comparison[col1]) * 100
        print("\n¿Cuánto mayor son los valores para Solver 8 respecto al 5?")
        print(comparison['Diff (%)'])

    # -------------------------------------------------------
    # 4. VISUALIZACIÓN
    # -------------------------------------------------------
    print("\nGenerando gráficas...")
    
    # Configurar estilo
    sns.set(style="whitegrid")
    
    # Gráfico 1: Scatter Plot de las 2 variables más importantes
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=dataset, 
        x='fd_sas_op_prevail_mean', 
        y='sas_cg_edge_density', 
        hue='winner', 
        palette={5: 'blue', 8: 'red'},
        style='winner',
        s=100,
        alpha=0.7
    )
    plt.title('Mapa de Decisión: Complejidad vs Densidad')
    plt.xlabel('Media de Precondiciones (Prevail Conditions)')
    plt.ylabel('Densidad del Grafo Causal (Edge Density)')
    plt.legend(title='Ganador', labels=['Solver 5 (FF)', 'Solver 8 (CG)'])
    
    # Guardar o mostrar
    plt.savefig('analisis_ff_vs_cg.png')
    print("Gráfica guardada como 'analisis_ff_vs_cg.png'. Ábrela para ver las conclusiones.")
    plt.show()

    # Gráfico 2: Boxplots para ver rangos
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.boxplot(ax=axes[0], x='winner', y='fd_sas_op_prevail_mean', data=dataset, palette="Set2")
    axes[0].set_title('Complejidad de Acciones')
    
    sns.boxplot(ax=axes[1], x='winner', y='sas_cg_edge_density', data=dataset, palette="Set2")
    axes[1].set_title('Densidad del Grafo')
    
    sns.boxplot(ax=axes[2], x='winner', y='lm_lm_num_reasonable_orders', data=dataset, palette="Set2")
    axes[2].set_title('Número de Landmarks (Hitos)')
    
    plt.savefig('boxplots_metricas.png')
    print("Gráfica guardada como 'boxplots_metricas.png'.")
    plt.show()

if __name__ == "__main__":
    analyze_distributions()