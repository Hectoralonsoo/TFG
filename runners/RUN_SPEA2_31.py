import json
import random
import os
from time import time
import inspyred
import numpy as np

from Loaders.LoadStreamingPlans import load_streaming_plan_json
from Loaders.LoadUsers import load_users_from_json
from Loaders.LoadPlatforms import load_platforms_json

from utils.evaluation import evaluator, calcular_minutos_ponderados
from utils.logging_custom import observer
from generators.Individual_generator import generar_individuo
from algorithms.SPEA2 import SPEA2


def create_directory_structure(base_path, dataset_name, config_name, run_number):
    solutions_dir = os.path.join(base_path, "solutions", dataset_name.replace('.json', ''), config_name,
                                 f"run{run_number}")

    summaries_dir = os.path.join(base_path, "summaries", dataset_name.replace('.json', ''))

    os.makedirs(solutions_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)

    return solutions_dir, summaries_dir


def safe_fitness_to_list(fitness):
    """
    Safely convert fitness to list, handling both single values and iterables
    """
    if isinstance(fitness, (int, float, np.number)):
        return [float(fitness)]
    elif hasattr(fitness, '__iter__'):
        return [float(f) for f in fitness]
    else:
        try:
            return list(fitness)
        except TypeError:
            return [float(fitness)]


def main():
    # Configuración única a ejecutar
    config = {
        "name": "uniform_inversion_high_mutation",
        "pop_size": 50,
        "mutation_rate": 0.1,
        "crossover_rate": 0.6,
        "variator": [inspyred.ec.variators.uniform_crossover, inspyred.ec.variators.inversion_mutation]
    }

    streamingPlans = load_streaming_plan_json("../Data/streamingPlans.json")
    platforms_indexed = load_platforms_json("../Data/indice_plataformas.json")
    user_datasets = [
        "users1.json",
        "users2.json",
        "users3.json",
        "users4.json",
        "users5.json"
    ]

    base_results_path = "C:\\Users\\gonza\\TFG\\31Executions\\SPEA2"
    all_results = []

    os.makedirs(base_results_path, exist_ok=True)

    for dataset_name in user_datasets:
        dataset_path = f"../Data/{dataset_name}"
        print(f"\n📂 Ejecutando para dataset: {dataset_name}")
        users = load_users_from_json(dataset_path)

        dataset_results = []

        print(f"\n🔧 Configuración: {config['name']}")

        config_results = []

        # Ejecutar 31 veces
        for run in range(31):
            print(f"🔁 Run {run + 1}/31")

            solutions_dir, summaries_dir = create_directory_structure(
                base_results_path, dataset_name, config['name'], run + 1
            )

            seed = time()
            prng = random.Random(seed)

            # Usar la clase SPEA2 personalizada
            algorithm = SPEA2(prng)
            bounder = inspyred.ec.Bounder(1, len(platforms_indexed))

            # Configurar el algoritmo SPEA2
            algorithm.selector = inspyred.ec.selectors.tournament_selection
            algorithm.variator = config["variator"]
            algorithm.terminator = inspyred.ec.terminators.no_improvement_termination
            algorithm.observer = observer
            algorithm.evaluator = evaluator
            algorithm.generator = generar_individuo

            args = {
                'users': users,
                'streamingPlans': streamingPlans,
                'platforms_indexed': platforms_indexed,
                'max_generations_without_improvement': 10,
                'improvement_tolerance': 1e-6,
                'k': 3,  # Para el cálculo de densidad en SPEA2
                'archive_size': 300,  # Tamaño del archivo
            }

            start_time = time()

            try:
                final_pop = algorithm.evolve(
                    pop_size=config["pop_size"],
                    maximize=False,
                    max_generations=300,
                    num_selected=config["pop_size"],
                    tournament_size=3,
                    mutation_rate=config["mutation_rate"],
                    crossover_rate=config["crossover_rate"],
                    **args
                )

                end_time = time()
                execution_time = end_time - start_time
                generations = algorithm.num_generations

                # Verificar que el archivo existe y no está vacío
                if hasattr(algorithm, 'archive') and algorithm.archive:
                    pareto_size = len(algorithm.archive)
                    # Usar objective_values en lugar de fitness para los puntos del frente de Pareto
                    pareto_points = []
                    for ind in algorithm.archive:
                        if hasattr(ind, 'objective_values') and ind.objective_values:
                            pareto_points.append(safe_fitness_to_list(ind.objective_values))
                        elif hasattr(ind, 'fitness'):
                            pareto_points.append(safe_fitness_to_list(ind.fitness))
                        else:
                            pareto_points.append([0.0])  # Valor por defecto

                    print(f"📊 Archivo SPEA2 tiene {pareto_size} soluciones")

                    # Guardar soluciones del archivo SPEA2
                    solutions_saved = 0
                    for idx, ind in enumerate(algorithm.archive):
                        try:
                            # Asegurar que el individuo tenga los datos necesarios
                            if hasattr(ind, 'candidate'):
                                calcular_minutos_ponderados(ind.candidate, args)
                                ind.monthly_data = args.get('monthly_data_by_user', {})

                                export = {
                                    "candidate": ind.candidate,
                                    "fitness": safe_fitness_to_list(ind.fitness) if hasattr(ind,
                                                                                            'fitness') else safe_fitness_to_list(
                                        ind.objective_values),
                                    "objective_values": safe_fitness_to_list(ind.objective_values) if hasattr(ind,
                                                                                                              'objective_values') else [],
                                    "monthly_data": ind.monthly_data
                                }
                            else:
                                # Si no tiene candidate, usar el individuo completo
                                calcular_minutos_ponderados(ind, args)
                                export = {
                                    "candidate": ind,
                                    "fitness": "N/A",
                                    "objective_values": [],
                                    "monthly_data": args.get('monthly_data_by_user', {})
                                }

                            solution_filename = f"sol{idx}.json"
                            solution_path = os.path.join(solutions_dir, solution_filename)

                            # Verificar que el directorio existe antes de escribir
                            if not os.path.exists(solutions_dir):
                                os.makedirs(solutions_dir, exist_ok=True)
                                print(f"⚠️ Recreando directorio: {solutions_dir}")

                            with open(solution_path, 'w', encoding='utf-8') as f:
                                json.dump(export, f, ensure_ascii=False, indent=2)

                            solutions_saved += 1

                        except Exception as e:
                            print(f"❌ Error guardando solución {idx}: {e}")
                            continue

                    print(f"💾 Guardadas {solutions_saved} soluciones en {solutions_dir}")

                else:
                    print("⚠️ El archivo SPEA2 está vacío o no existe")
                    pareto_size = 0
                    pareto_points = []

                run_result = {
                    'dataset': dataset_name,
                    'config': config['name'],
                    'run': run + 1,
                    'execution_time': execution_time,
                    'generations': generations,
                    'pareto_size': pareto_size,
                    'pareto_points': pareto_points
                }

                config_results.append(run_result)

                # Guardar resumen del run
                run_summary = {
                    'run_info': run_result,
                    'solutions_count': pareto_size,
                    'solutions_directory': solutions_dir
                }

                run_summary_path = os.path.join(solutions_dir, "run_summary.json")

                try:
                    with open(run_summary_path, 'w') as f:
                        json.dump(run_summary, f, indent=2)
                    print(f"📝 Resumen del run guardado en: {run_summary_path}")
                except Exception as e:
                    print(f"❌ Error guardando resumen del run: {e}")

            except Exception as e:
                print(f"❌ Error durante la evolución en run {run + 1}: {e}")
                continue

        # Guardar resumen de la configuración por dataset
        if config_results:
            config_summary_path = os.path.join(summaries_dir, f"summary_{config['name']}.json")
            try:
                with open(config_summary_path, 'w') as f:
                    json.dump(config_results, f, indent=2)
                print(f"📄 Resumen de configuración guardado en: {config_summary_path}")
            except Exception as e:
                print(f"❌ Error guardando resumen de configuración: {e}")

        dataset_results.extend(config_results)

        # Guardar resumen completo del dataset
        if dataset_results:
            dataset_summary_path = os.path.join(summaries_dir,
                                                f"summary_complete_{dataset_name.replace('.json', '')}.json")
            try:
                with open(dataset_summary_path, 'w') as f:
                    json.dump(dataset_results, f, indent=2)
                print(f"📋 Resumen completo del dataset guardado en: {dataset_summary_path}")
            except Exception as e:
                print(f"❌ Error guardando resumen del dataset: {e}")

        all_results.extend(dataset_results)

    # Guardar resumen general de todos los experimentos
    if all_results:
        general_summary_path = os.path.join(base_results_path, "summary_all_experiments.json")
        try:
            with open(general_summary_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"📊 Summary general guardado en: {general_summary_path}")
        except Exception as e:
            print(f"❌ Error guardando resumen general: {e}")

    print(f"\n✅ Experimentos completados!")
    print(f"📁 Estructura de archivos creada en: {base_results_path}")
    print(f"🎯 Configuración ejecutada: {config['name']}")
    print(f"🔢 Total de runs por dataset: 31")
    print(f"🗂️ Total de datasets: {len(user_datasets)}")
    print(f"📈 Total de experimentos: {31 * len(user_datasets)}")


if __name__ == "__main__":
    main()