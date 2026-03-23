from src.tfg.algorithms.PACO import PACOStreaming, fitness_paco
from src.tfg.loaders.LoadStreamingPlans import load_streaming_plan_json
from src.tfg.loaders.LoadUsers import load_users_from_json
import json
import os
from time import time
from src.tfg.paths import DATA_DIR, TEST_EXECUTIONS_DIR, ANALYSIS_OUTPUT_DIR, EXECUTIONS31_DIR


def create_directory_structure(base_path, dataset_name, config_name, run_number):
    solutions_dir = os.path.join(base_path, "solutions", dataset_name.replace('.json', ''), config_name,
                                 f"run{run_number}")

    summaries_dir = os.path.join(base_path, "summaries", dataset_name.replace('.json', ''))

    os.makedirs(solutions_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)

    return solutions_dir, summaries_dir


def main():
    configurations = [
        {
            "name": "balanced_original",
            "n_ants": 20,
            "n_iterations": 200,
            "rho": 0.4,
            "alpha": 1.0,
            "beta": 3.0,
            "archive_size": 100,
            "no_improvement_generations": 12
        },
        {
            "name": "high_pheromone_influence",
            "n_ants": 20,
            "n_iterations": 200,
            "rho": 0.3,
            "alpha": 3.0,
            "beta": 2.0,
            "archive_size": 100,
            "no_improvement_generations": 12
        },
        {
            "name": "high_heuristic_influence",
            "n_ants": 20,
            "n_iterations": 200,
            "rho": 0.4,
            "alpha": 0.5,
            "beta": 5.0,
            "archive_size": 100,
            "no_improvement_generations": 12
        },
        {
            "name": "slow_evaporation_exploitative",
            "n_ants": 20,
            "n_iterations": 200,
            "rho": 0.2,
            "alpha": 2.0,
            "beta": 3.0,
            "archive_size": 100,
            "no_improvement_generations": 12
        },
        {
            "name": "fast_evaporation_explorative",
            "n_ants": 20,
            "n_iterations": 200,
            "rho": 0.7,
            "alpha": 1.0,
            "beta": 2.5,
            "archive_size": 100,
            "no_improvement_generations": 12
        },
        {
            "name": "minimal_guidance",
            "n_ants": 20,
            "n_iterations": 200,
            "rho": 0.5,
            "alpha": 0.2,
            "beta": 1.0,
            "archive_size": 100,
            "no_improvement_generations": 12
        }
    ]

    streamingPlans = load_streaming_plan_json(DATA_DIR / "streamingPlans.json")

    with open(DATA_DIR / "indice_plataformas.json", "r", encoding="utf-8") as f:
        platforms_indexed = json.load(f)

    plataformas_disponibles = [int(p) for p in platforms_indexed.keys()]

    user_datasets = [
        "users1.json",
        "users2.json",
        "users3.json",
        "users4.json",
        "users5.json"
    ]

    base_results_path = TEST_EXECUTIONS_DIR / "PACO"
    all_results = []

    os.makedirs(base_results_path, exist_ok=True)

    for dataset_name in user_datasets:
        dataset_path = DATA_DIR / dataset_name
        print(f"\n📂 Ejecutando para dataset: {dataset_name}")
        users = load_users_from_json(dataset_path)

        dataset_results = []

        for config in configurations:
            print(f"\n🔧 Configuración: {config['name']}")
            print(f"  Hormigas: {config['n_ants']}, Iteraciones: {config['n_iterations']}")
            print(f"  Parámetros: rho={config['rho']}, alpha={config['alpha']}, beta={config['beta']}")
            print(f"  Archivo: {config['archive_size']}, Terminación: {config['no_improvement_generations']}")

            config_results = []

            for run in range(5):
                print(f"🔁 Run {run + 1}/5")

                solutions_dir, summaries_dir = create_directory_structure(
                    base_results_path, dataset_name, config['name'], run + 1
                )

                args = {
                    'users': users,
                    'streamingPlans': streamingPlans,
                    'platforms_indexed': platforms_indexed
                }

                paco = PACOStreaming(
                    n_ants=config['n_ants'],
                    n_iterations=config['n_iterations'],
                    n_months=12,
                    users=users,
                    n_users=len(users),
                    platform_options=plataformas_disponibles,
                    rho=config['rho'],
                    alpha=config['alpha'],
                    beta=config['beta'],
                    archive_size=config['archive_size'],
                    no_improvement_generations=config['no_improvement_generations']
                )

                # Optimizar
                print(f"  Iniciando optimización...")
                start_time = time()
                pareto_archive = paco.optimize(lambda sol: fitness_paco(sol, args))
                end_time = time()
                execution_time = end_time - start_time

                # Recopilar métricas del run
                iterations_completed = len(paco.archive_history)
                pareto_size = len(pareto_archive)
                pareto_points = [list(objectives) for _, objectives in pareto_archive]
                solutions_evaluated = iterations_completed * paco.n_ants

                run_result = {
                    'dataset': dataset_name,
                    'config': config['name'],
                    'run': run + 1,
                    'execution_time': execution_time,
                    'iterations_completed': iterations_completed,
                    'max_iterations': config['n_iterations'],
                    'terminated_early': paco.terminated_early,
                    'generations_without_improvement': paco.generations_without_improvement,
                    'pareto_size': pareto_size,
                    'pareto_points': pareto_points,
                    'solutions_evaluated': solutions_evaluated,
                    'config_params': {
                        'n_ants': config['n_ants'],
                        'rho': config['rho'],
                        'alpha': config['alpha'],
                        'beta': config['beta'],
                        'archive_size': config['archive_size'],
                        'no_improvement_generations': config['no_improvement_generations']
                    }
                }

                config_results.append(run_result)

                if paco.terminated_early:
                    print(f"    ✓ Terminó tempranamente: {iterations_completed}/{config['n_iterations']} iteraciones")
                else:
                    print(f"    ✓ Completó todas las iteraciones: {iterations_completed}")

                print(f"    📊 Frente de Pareto: {pareto_size} soluciones")
                print(f"    ⏱️  Tiempo de ejecución: {execution_time:.2f}s")

                for idx, (solution, objectives) in enumerate(pareto_archive):
                    export = {
                        "solution": solution.tolist() if hasattr(solution, 'tolist') else solution,
                        "objectives": list(objectives),
                        "fitness": {
                            "minutos_ponderados": objectives[0],
                            "costo_total": objectives[1]
                        }
                    }

                    solution_filename = f"sol{idx}.json"
                    solution_path = os.path.join(solutions_dir, solution_filename)

                    with open(solution_path, 'w', encoding='utf-8') as f:
                        json.dump(export, f, ensure_ascii=False, indent=2)

                if pareto_archive:
                    objectives = [obj for _, obj in pareto_archive]
                    min_cost = min(obj[1] for obj in objectives)
                    max_cost = max(obj[1] for obj in objectives)
                    min_minutes = min(obj[0] for obj in objectives)
                    max_minutes = max(obj[0] for obj in objectives)

                    objective_stats = {
                        'cost_range': {'min': min_cost, 'max': max_cost},
                        'minutes_range': {'min': min_minutes, 'max': max_minutes}
                    }
                else:
                    objective_stats = None

                run_summary = {
                    'run_info': run_result,
                    'solutions_count': len(pareto_archive),
                    'solutions_directory': solutions_dir,
                    'objective_statistics': objective_stats,
                    'algorithm_info': {
                        'archive_history_length': len(paco.archive_history),
                        'max_archive_size_reached': max(
                            len(arch) for arch in paco.archive_history) if paco.archive_history else 0
                    }
                }

                run_summary_path = os.path.join(solutions_dir, "run_summary.json")
                with open(run_summary_path, 'w') as f:
                    json.dump(run_summary, f, indent=2)

            config_summary_path = os.path.join(summaries_dir, f"summary_{config['name']}.json")
            with open(config_summary_path, 'w') as f:
                json.dump(config_results, f, indent=2)

            avg_time = sum(r['execution_time'] for r in config_results) / len(config_results)
            avg_pareto_size = sum(r['pareto_size'] for r in config_results) / len(config_results)
            early_terminations = sum(1 for r in config_results if r['terminated_early'])

            print(f"  📈 Resumen configuración {config['name']}:")
            print(f"    Tiempo promedio: {avg_time:.2f}s")
            print(f"    Tamaño Pareto promedio: {avg_pareto_size:.1f}")
            print(f"    Terminaciones tempranas: {early_terminations}/5")

            dataset_results.extend(config_results)

        dataset_summary_path = os.path.join(summaries_dir, f"summary_complete_{dataset_name.replace('.json', '')}.json")
        with open(dataset_summary_path, 'w') as f:
            json.dump(dataset_results, f, indent=2)

        all_results.extend(dataset_results)

    general_summary_path = os.path.join(base_results_path, "summary_all_experiments.json")
    with open(general_summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Experimentos PACO completados!")
    print(f"📁 Estructura de archivos creada en: {base_results_path}")
    print(f"📊 Summary general guardado en: {general_summary_path}")

    total_runs = len(all_results)
    total_time = sum(r['execution_time'] for r in all_results)
    avg_pareto_size = sum(r['pareto_size'] for r in all_results) / total_runs
    total_early_terminations = sum(1 for r in all_results if r['terminated_early'])

    print(f"\n📊 Estadísticas generales:")
    print(f"  Total de runs: {total_runs}")
    print(f"  Tiempo total de ejecución: {total_time:.2f}s ({total_time / 60:.1f} min)")
    print(f"  Tamaño promedio del frente de Pareto: {avg_pareto_size:.1f}")
    print(
        f"  Terminaciones tempranas: {total_early_terminations}/{total_runs} ({100 * total_early_terminations / total_runs:.1f}%)")


if __name__ == "__main__":
    main()