from algorithms.PACO import PACOStreaming, fitness_paco
from Loaders.LoadStreamingPlans import load_streaming_plan_json
from Loaders.LoadPlatforms import load_platforms_json
from Loaders.LoadUsers import load_users_from_json
from utils.logging_custom import plot_pareto_front_paco
from utils.logging_custom import plot_ant_paths_lines
from utils.save import save_pareto_archive_paco
from utils.logging_custom import plot_user_platforms_over_time
from utils.logging_custom import observer_paco
import json
import os
from time import time


def create_directory_structure(base_path, dataset_name, config_name, run_number):
    solutions_dir = os.path.join(base_path, "solutions", dataset_name.replace('.json', ''), config_name,
                                 f"run{run_number}")

    summaries_dir = os.path.join(base_path, "summaries", dataset_name.replace('.json', ''))

    os.makedirs(solutions_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)

    return solutions_dir, summaries_dir


def get_resume_point():
    """Determina desde dónde continuar la ejecución"""
    # Punto de reanudación: users4 desde run 22, luego users5 completo
    resume_config = {
        "users4.json": 22,  # Empezar desde run 22 (run 22-31)
        "users5.json": 1  # Empezar desde run 1 (run 1-31)
    }
    return resume_config


def main():
    # Solo la configuración que mejor hipervolumen ha dado
    config = {
        "name": "high_heuristic_influence",
        "n_ants": 20,
        "n_iterations": 200,
        "rho": 0.4,
        "alpha": 0.5,
        "beta": 5.0,
        "archive_size": 100,
        "no_improvement_generations": 12
    }

    print(f"🔧 Reanudando ejecución desde users4 run 22...")
    print(f"📋 Configuración: {config['name']}")
    print(f"  Hormigas: {config['n_ants']}, Iteraciones: {config['n_iterations']}")
    print(f"  Parámetros: rho={config['rho']}, alpha={config['alpha']}, beta={config['beta']}")
    print(f"  Archivo: {config['archive_size']}, Terminación: {config['no_improvement_generations']}")
    print(f"  Datasets restantes: users4 (runs 22-31), users5 (runs 1-31)\n")

    streamingPlans = load_streaming_plan_json("../Data/streamingPlans.json")

    with open("../Data/indice_plataformas.json", "r", encoding="utf-8") as f:
        platforms_indexed = json.load(f)

    plataformas_disponibles = [int(p) for p in platforms_indexed.keys()]

    # Solo procesar los datasets restantes
    user_datasets_to_resume = [
        "users4.json",
        "users5.json"
    ]

    base_results_path = "C:\\Users\\hctr0\\PycharmProjects\\TFG_Hector\\31Executions\\PACO"
    resume_points = get_resume_point()

    # Cargar resultados existentes si existen
    general_summary_path = os.path.join(base_results_path, "summary_all_experiments.json")
    if os.path.exists(general_summary_path):
        with open(general_summary_path, 'r') as f:
            all_results = json.load(f)
        print(f"📂 Cargados {len(all_results)} resultados previos")
    else:
        all_results = []
        print("⚠️  No se encontraron resultados previos, empezando desde cero")

    os.makedirs(base_results_path, exist_ok=True)

    # Calcular runs totales restantes
    total_remaining_runs = (31 - 22 + 1) + 31  # users4: runs 22-31 (10 runs) + users5: runs 1-31 (31 runs)
    current_run = 0

    for dataset_name in user_datasets_to_resume:
        dataset_path = f"../Data/{dataset_name}"
        print(f"\n📂 Procesando dataset: {dataset_name}")
        users = load_users_from_json(dataset_path)

        start_run = resume_points[dataset_name]
        end_run = 31

        # Cargar resultados existentes del dataset si existen
        summaries_dir = os.path.join(base_results_path, "summaries", dataset_name.replace('.json', ''))
        config_summary_path = os.path.join(summaries_dir, f"summary_{config['name']}.json")

        if os.path.exists(config_summary_path):
            with open(config_summary_path, 'r') as f:
                dataset_results = json.load(f)
            print(f"📋 Cargados {len(dataset_results)} resultados previos para {dataset_name}")
        else:
            dataset_results = []
            print(f"🆕 No hay resultados previos para {dataset_name}")

        print(f"🔄 Ejecutando runs {start_run}-{end_run} para {dataset_name}")

        for run in range(start_run, end_run + 1):
            current_run += 1
            print(f"🏃 Run {run}/31 - Progreso restante: {current_run}/{total_remaining_runs}")

            # Verificar si este run ya existe
            existing_run = next((r for r in dataset_results if r['run'] == run), None)
            if existing_run:
                print(f"    ✅ Run {run} ya existe, saltando...")
                continue

            # Crear estructura de directorios para este run específico
            solutions_dir, summaries_dir = create_directory_structure(
                base_results_path, dataset_name, config['name'], run
            )

            # Verificar si ya existen archivos de soluciones para este run
            if os.path.exists(solutions_dir) and len(os.listdir(solutions_dir)) > 0:
                print(f"    ✅ Soluciones del run {run} ya existen, saltando...")
                continue

            args = {
                'users': users,
                'streamingPlans': streamingPlans,
                'platforms_indexed': platforms_indexed
            }

            # Instanciar PACO con la configuración específica
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
                'run': run,
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

            dataset_results.append(run_result)

            # Información sobre la terminación
            if paco.terminated_early:
                print(f"    ✓ Terminó tempranamente: {iterations_completed}/{config['n_iterations']} iteraciones")
            else:
                print(f"    ✓ Completó todas las iteraciones: {iterations_completed}")

            print(f"    📊 Frente de Pareto: {pareto_size} soluciones")
            print(f"    ⏱️  Tiempo: {execution_time:.2f}s")

            # Guardar soluciones individuales en la carpeta específica del run
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

            # Guardar summary individual del run
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

            # Guardar progreso parcial después de cada run
            config_summary_path = os.path.join(summaries_dir, f"summary_{config['name']}.json")
            with open(config_summary_path, 'w') as f:
                json.dump(dataset_results, f, indent=2)

        # Actualizar all_results con los nuevos resultados del dataset
        # Remover resultados anteriores de este dataset si existen
        all_results = [r for r in all_results if r['dataset'] != dataset_name]
        all_results.extend(dataset_results)

        # Guardar summary general actualizado después de cada dataset
        with open(general_summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        # Estadísticas del dataset actual
        current_dataset_runs = [r for r in dataset_results if r['run'] >= start_run]
        if current_dataset_runs:
            avg_time = sum(r['execution_time'] for r in current_dataset_runs) / len(current_dataset_runs)
            avg_pareto_size = sum(r['pareto_size'] for r in current_dataset_runs) / len(current_dataset_runs)
            early_terminations = sum(1 for r in current_dataset_runs if r['terminated_early'])
            min_pareto_size = min(r['pareto_size'] for r in current_dataset_runs)
            max_pareto_size = max(r['pareto_size'] for r in current_dataset_runs)

            print(f"\n  📈 Resumen runs completados para {dataset_name}:")
            print(f"    Runs ejecutados: {len(current_dataset_runs)}")
            print(f"    Tiempo promedio: {avg_time:.2f}s")
            print(f"    Tamaño Pareto promedio: {avg_pareto_size:.1f} (min: {min_pareto_size}, max: {max_pareto_size})")
            print(
                f"    Terminaciones tempranas: {early_terminations}/{len(current_dataset_runs)} ({100 * early_terminations / len(current_dataset_runs):.1f}%)")

    print(f"\n✅ Reanudación de experimentos PACO completada!")
    print(f"📁 Estructura de archivos en: {base_results_path}")
    print(f"📊 Summary general actualizado en: {general_summary_path}")

    # Estadísticas finales
    total_runs_executed = len(all_results)
    if total_runs_executed > 0:
        total_time = sum(r['execution_time'] for r in all_results)
        avg_pareto_size = sum(r['pareto_size'] for r in all_results) / total_runs_executed
        total_early_terminations = sum(1 for r in all_results if r['terminated_early'])
        min_pareto_global = min(r['pareto_size'] for r in all_results)
        max_pareto_global = max(r['pareto_size'] for r in all_results)

        print(f"\n📊 Estadísticas generales finales:")
        print(f"  Configuración: {config['name']}")
        print(f"  Total de runs ejecutados: {total_runs_executed}")
        print(f"  Tiempo total de ejecución: {total_time:.2f}s ({total_time / 60:.1f} min)")
        print(f"  Tiempo promedio por run: {total_time / total_runs_executed:.2f}s")
        print(f"  Tamaño promedio del frente de Pareto: {avg_pareto_size:.1f}")
        print(f"  Rango tamaño frente de Pareto: {min_pareto_global} - {max_pareto_global}")
        print(
            f"  Terminaciones tempranas: {total_early_terminations}/{total_runs_executed} ({100 * total_early_terminations / total_runs_executed:.1f}%)")


if __name__ == "__main__":
    main()