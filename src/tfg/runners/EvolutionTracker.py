import json
import random
import os
from time import time
import inspyred
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from src.tfg.paths import DATA_DIR, TEST_EXECUTIONS_DIR, ANALYSIS_OUTPUT_DIR, EXECUTIONS31_DIR

from src.tfg.loaders.LoadStreamingPlans import load_streaming_plan_json
from src.tfg.loaders.LoadUsers import load_users_from_json
from src.tfg.loaders.LoadPlatforms import load_platforms_json
from src.tfg.algorithms.SPEA2 import SPEA2
from src.tfg.algorithms.PACO import PACOStreaming, fitness_paco

from src.tfg.utils.evaluation import evaluator, calcular_minutos_ponderados
from src.tfg.generators.Individual_generator import generar_individuo

# Configurar matplotlib para mejor visualización
rcParams['font.size'] = 10
rcParams['axes.titlesize'] = 12
rcParams['axes.labelsize'] = 11
rcParams['legend.fontsize'] = 9


class EvolutionTracker:
    """Clase para trackear la evolución del fitness en cada generación"""

    def __init__(self):
        self.cost_history = []
        self.minutes_history = []
        self.generation = 0

    def reset(self):
        self.cost_history = []
        self.minutes_history = []
        self.generation = 0

    def record(self, population):
        """Registra el mejor fitness de la generación actual"""
        if not population:
            return

        # Obtener todos los fitness de la población
        costs = []
        minutes = []

        for ind in population:
            # Priorizar objective_values que contiene los valores reales de los objetivos
            objectives = None

            if hasattr(ind, 'objective_values') and ind.objective_values is not None:
                objectives = ind.objective_values
            elif hasattr(ind, 'fitness') and ind.fitness is not None:
                # Para NSGA-II, fitness contiene directamente los objetivos
                objectives = ind.fitness

            if objectives is not None:
                # Convertir a lista si es necesario
                if isinstance(objectives, (int, float, np.number)):
                    # Valor escalar (fitness SPEA2), skip
                    continue

                try:
                    obj_list = list(objectives)
                    if len(obj_list) >= 2:
                        # obj[0] = minutos ponderados (minimizar)
                        # obj[1] = costo total (minimizar)
                        minutes.append(float(obj_list[0]))
                        costs.append(float(obj_list[1]))
                except (TypeError, ValueError):
                    continue

        if costs and minutes:
            # Guardar el mejor (menor) costo y mejores (menores) minutos ponderados
            self.cost_history.append(min(costs))
            self.minutes_history.append(min(minutes))
            self.generation += 1
        else:
            # Si no hay datos válidos, mantener el último valor o usar 0
            if self.cost_history:
                self.cost_history.append(self.cost_history[-1])
                self.minutes_history.append(self.minutes_history[-1])
            else:
                self.cost_history.append(0)
                self.minutes_history.append(0)
            self.generation += 1


def observer_with_tracker(population, num_generations, num_evaluations, args):
    """Observer personalizado que trackea la evolución"""
    if 'tracker' in args:
        args['tracker'].record(population)


def run_nsga2(users, streamingPlans, platforms_indexed, tracker):
    """Ejecuta NSGA-II y retorna el historial de evolución"""
    print("  🔵 Ejecutando NSGA-II...")
    tracker.reset()

    seed = time()
    prng = random.Random(seed)

    algorithm = inspyred.ec.emo.NSGA2(prng)
    bounder = inspyred.ec.Bounder(1, len(platforms_indexed))

    algorithm.selector = inspyred.ec.selectors.tournament_selection
    algorithm.replacer = inspyred.ec.replacers.nsga_replacement
    algorithm.variator = [inspyred.ec.variators.uniform_crossover,
                          inspyred.ec.variators.inversion_mutation]
    algorithm.terminator = inspyred.ec.terminators.generation_termination
    algorithm.observer = observer_with_tracker

    args = {
        'users': users,
        'streamingPlans': streamingPlans,
        'platforms_indexed': platforms_indexed,
        'tracker': tracker
    }

    start_time = time()
    final_pop = algorithm.evolve(
        generator=generar_individuo,
        evaluator=evaluator,
        bounder=bounder,
        pop_size=50,
        maximize=False,
        max_generations=100,
        num_selected=50,
        tournament_size=3,
        num_elites=2,
        mutation_rate=0.1,
        crossover_rate=0.6,
        **args
    )
    end_time = time()

    # Invertir los minutos (están en negativo)
    cost_history = tracker.cost_history.copy()
    minutes_history = [-m for m in tracker.minutes_history]

    print(f"    ✅ Completado en {end_time - start_time:.2f}s")
    return cost_history, minutes_history


def run_spea2(users, streamingPlans, platforms_indexed, tracker):
    """Ejecuta SPEA2 y retorna el historial de evolución"""
    print("  🟢 Ejecutando SPEA2...")
    tracker.reset()

    seed = time()
    prng = random.Random(seed)

    algorithm = SPEA2(prng)
    bounder = inspyred.ec.Bounder(1, len(platforms_indexed))

    algorithm.selector = inspyred.ec.selectors.tournament_selection
    algorithm.variator = [inspyred.ec.variators.uniform_crossover,
                          inspyred.ec.variators.inversion_mutation]
    algorithm.terminator = inspyred.ec.terminators.generation_termination
    algorithm.observer = observer_with_tracker
    algorithm.evaluator = evaluator
    algorithm.generator = generar_individuo

    args = {
        'users': users,
        'streamingPlans': streamingPlans,
        'platforms_indexed': platforms_indexed,
        'tracker': tracker,
        'disable_early_termination': True  # Deshabilitar terminación temprana para comparación justa
    }

    start_time = time()
    final_pop = algorithm.evolve(
        evaluator=evaluator,
        bounder=bounder,
        max_generations=100,
        pop_size=50,
        maximize=False,
        num_selected=50,
        tournament_size=3,
        num_elites=2,
        mutation_rate=0.1,
        crossover_rate=0.6,
        **args
    )
    end_time = time()

    # Invertir los minutos (están en negativo)
    cost_history = tracker.cost_history.copy()
    minutes_history = [-m for m in tracker.minutes_history]

    generations_completed = len(cost_history)
    print(f"    ✅ Completado en {end_time - start_time:.2f}s ({generations_completed} generaciones)")

    return cost_history, minutes_history


def run_paco(users, streamingPlans, platforms_indexed):
    """Ejecuta PACO y retorna el historial de evolución"""
    print("  🟡 Ejecutando PACO...")

    plataformas_disponibles = [int(p) for p in platforms_indexed.keys()]

    args = {
        'users': users,
        'streamingPlans': streamingPlans,
        'platforms_indexed': platforms_indexed
    }

    paco = PACOStreaming(
        n_ants=25,
        n_iterations=100,  # Ajustado a 100 para comparar con las generaciones
        n_months=12,
        users=users,
        n_users=len(users),
        platform_options=plataformas_disponibles,
        rho=0.3,
        alpha=3.0,
        beta=2.0,
        archive_size=100,
        no_improvement_generations=100  # Sin terminación temprana para comparación justa
    )

    cost_history = []
    minutes_history = []

    start_time = time()

    pareto_archive = paco.optimize(lambda sol: fitness_paco(sol, args))

    if hasattr(paco, 'archive_history') and paco.archive_history:
        for iteration_archive in paco.archive_history:
            if iteration_archive:
                costs = [obj[1] for _, obj in iteration_archive]
                minutes = [obj[0] for _, obj in iteration_archive]

                if costs and minutes:
                    cost_history.append(min(costs))
                    minutes_history.append(min(minutes))
            else:
                if cost_history:
                    cost_history.append(cost_history[-1])
                    minutes_history.append(minutes_history[-1])

    end_time = time()

    iterations_completed = len(cost_history)
    print(f"    ✅ Completado en {end_time - start_time:.2f}s ({iterations_completed} iteraciones)")

    return cost_history, minutes_history


def plot_comparison(dataset_name, nsga2_data, spea2_data, paco_data, output_dir):
    """Genera las gráficas de comparación para un dataset"""

    nsga2_cost, nsga2_minutes = nsga2_data
    spea2_cost, spea2_minutes = spea2_data
    paco_cost, paco_minutes = paco_data

    # PACO ya viene en positivo, NSGA-II y SPEA2 ahora también están invertidos
    # Pero por si acaso, aseguramos valores positivos
    nsga2_minutes_pos = [abs(m) for m in nsga2_minutes] if nsga2_minutes else []
    spea2_minutes_pos = [abs(m) for m in spea2_minutes] if spea2_minutes else []
    paco_minutes_pos = [abs(m) for m in paco_minutes] if paco_minutes else []

    # Crear figura con 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Evolución del Fitness - {dataset_name}', fontsize=16, fontweight='bold')

    # Gráfica 1: Evolución del Costo (debe disminuir)
    if nsga2_cost:
        ax1.plot(range(len(nsga2_cost)), nsga2_cost, label='NSGA-II',
                 linewidth=2.5, color='#2E86AB', alpha=0.9)
    if spea2_cost:
        ax1.plot(range(len(spea2_cost)), spea2_cost, label='SPEA2',
                 linewidth=2.5, color='#A23B72', alpha=0.9)
    if paco_cost:
        ax1.plot(range(len(paco_cost)), paco_cost, label='PACO',
                 linewidth=2.5, color='#F18F01', alpha=0.9)

    ax1.set_xlabel('Generaciones/Iteraciones', fontsize=12)
    ax1.set_ylabel('Costo Total (€)', fontsize=12)
    ax1.set_title('Minimización del Costo', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(left=0)

    # Añadir información de valores finales
    final_costs = []
    if nsga2_cost:
        final_costs.append(f"NSGA-II: {nsga2_cost[-1]:.2f}€")
    if spea2_cost:
        final_costs.append(f"SPEA2: {spea2_cost[-1]:.2f}€")
    if paco_cost:
        final_costs.append(f"PACO: {paco_cost[-1]:.2f}€")

    if final_costs:
        ax1.text(0.02, 0.98, '\n'.join(final_costs), transform=ax1.transAxes,
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Gráfica 2: Evolución de Penalización (debe disminuir)
    if nsga2_minutes_pos:
        ax2.plot(range(len(nsga2_minutes_pos)), nsga2_minutes_pos, label='NSGA-II',
                 linewidth=2.5, color='#2E86AB', alpha=0.9)
    if spea2_minutes_pos:
        ax2.plot(range(len(spea2_minutes_pos)), spea2_minutes_pos, label='SPEA2',
                 linewidth=2.5, color='#A23B72', alpha=0.9)
    if paco_minutes_pos:
        ax2.plot(range(len(paco_minutes_pos)), paco_minutes_pos, label='PACO',
                 linewidth=2.5, color='#F18F01', alpha=0.9)

    ax2.set_xlabel('Generaciones/Iteraciones', fontsize=12)
    ax2.set_ylabel('Penalización por Déficit', fontsize=12)
    ax2.set_title('Minimización de Déficit de Visualización', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(left=0)

    # Añadir información de valores finales
    final_minutes = []
    if nsga2_minutes_pos:
        final_minutes.append(f"NSGA-II: {nsga2_minutes_pos[-1]:.0f}")
    if spea2_minutes_pos:
        final_minutes.append(f"SPEA2: {spea2_minutes_pos[-1]:.0f}")
    if paco_minutes_pos:
        final_minutes.append(f"PACO: {paco_minutes_pos[-1]:.0f}")

    if final_minutes:
        ax2.text(0.02, 0.98, '\n'.join(final_minutes), transform=ax2.transAxes,
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()

    # Guardar la figura
    output_path = os.path.join(output_dir, f'evolution_{dataset_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    📊 Gráfica guardada: {output_path}")


def main():
    print("🚀 Iniciando comparación de algoritmos con gráficas de evolución\n")

    # Cargar datos comunes
    streamingPlans = load_streaming_plan_json(DATA_DIR / "streamingPlans.json")
    platforms_indexed = load_platforms_json(DATA_DIR / "indice_plataformas.json")

    user_datasets = [
        "users1.json",
        "users2.json",
        "users3.json",
        "users4.json",
        "users5.json"
    ]

    # Directorio para guardar las gráficas
    output_dir = "C:\\Users\\hctr0\\PycharmProjects\\TFG_Hector\\AlgorithmComparison"
    os.makedirs(output_dir, exist_ok=True)

    # Tracker para NSGA-II y SPEA2
    tracker = EvolutionTracker()

    # Resultados para resumen
    results_summary = []

    # Ejecutar para cada dataset
    for dataset_name in user_datasets:
        dataset_path = DATA_DIR / dataset_name
        print(f"\n{'=' * 60}")
        print(f"📂 Procesando dataset: {dataset_name}")
        print(f"{'=' * 60}")

        users = load_users_from_json(dataset_path)

        # Ejecutar cada algoritmo
        nsga2_cost, nsga2_minutes = run_nsga2(users, streamingPlans, platforms_indexed, tracker)
        spea2_cost, spea2_minutes = run_spea2(users, streamingPlans, platforms_indexed, tracker)
        paco_cost, paco_minutes = run_paco(users, streamingPlans, platforms_indexed)

        # Debug: mostrar longitudes de historias
        print(f"  📏 Longitud de historias:")
        print(f"     NSGA-II: {len(nsga2_cost)} generaciones")
        print(f"     SPEA2:   {len(spea2_cost)} generaciones")
        print(f"     PACO:    {len(paco_cost)} iteraciones")

        # Generar gráfica inmediatamente para este dataset
        dataset_key = dataset_name.replace('.json', '')
        print(f"  📈 Generando gráfica para {dataset_key}...")
        plot_comparison(
            dataset_key,
            (nsga2_cost, nsga2_minutes),
            (spea2_cost, spea2_minutes),
            (paco_cost, paco_minutes),
            output_dir
        )

        # Guardar resumen de resultados
        results_summary.append({
            'dataset': dataset_name,
            'nsga2': {
                'final_cost': nsga2_cost[-1] if nsga2_cost else None,
                'final_minutes': nsga2_minutes[-1] if nsga2_minutes else None,
                'generations': len(nsga2_cost)
            },
            'spea2': {
                'final_cost': spea2_cost[-1] if spea2_cost else None,
                'final_minutes': spea2_minutes[-1] if spea2_minutes else None,
                'generations': len(spea2_cost)
            },
            'paco': {
                'final_cost': paco_cost[-1] if paco_cost else None,
                'final_minutes': paco_minutes[-1] if paco_minutes else None,
                'iterations': len(paco_cost)
            }
        })

    # Guardar resumen en JSON
    summary_path = os.path.join(output_dir, "comparison_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print("✅ Comparación completada!")
    print(f"📁 Gráficas guardadas en: {output_dir}")
    print(f"📄 Resumen guardado en: {summary_path}")
    print(f"{'=' * 60}\n")

    # Mostrar resumen final
    print("📊 RESUMEN DE RESULTADOS FINALES:")
    print(f"{'=' * 60}")
    for result in results_summary:
        print(f"\n📂 Dataset: {result['dataset']}")
        print(
            f"  NSGA-II  - Costo final: {result['nsga2']['final_cost']:.2f}€ | Minutos: {result['nsga2']['final_minutes']:.2f}")
        print(
            f"  SPEA2    - Costo final: {result['spea2']['final_cost']:.2f}€ | Minutos: {result['spea2']['final_minutes']:.2f}")
        print(
            f"  PACO     - Costo final: {result['paco']['final_cost']:.2f}€ | Minutos: {result['paco']['final_minutes']:.2f}")


if __name__ == "__main__":
    main()