import json
import random
import os
from time import time
import inspyred

from Loaders.LoadStreamingPlans import load_streaming_plan_json
from Loaders.LoadUsers import load_users_from_json
from Loaders.LoadPlatforms import load_platforms_json

from utils.evaluation import evaluator, calcular_minutos_ponderados
from utils.logging_custom import observer, plot_evolution, plot_generation_improve, plot_pareto_front
from generators.Individual_generator import generar_individuo


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
            "name": "uniform_reset_low_mutation",
            "pop_size": 75,
            "mutation_rate": 0.01,
            "crossover_rate": 0.6,
            "variator": [inspyred.ec.variators.uniform_crossover, inspyred.ec.variators.random_reset_mutation]
        },
        {
            "name": "uniform_reset_high_crossover",
            "pop_size": 75,
            "mutation_rate": 0.025,
            "crossover_rate": 0.8,
            "variator": [inspyred.ec.variators.uniform_crossover, inspyred.ec.variators.random_reset_mutation]
        },
        {
            "name": "uniform_inversion_high_mutation",
            "pop_size": 75,
            "mutation_rate": 0.1,
            "crossover_rate": 0.6,
            "variator": [inspyred.ec.variators.uniform_crossover, inspyred.ec.variators.inversion_mutation]
        },
        {
            "name": "npoint_reset_low_crossover",
            "pop_size": 75,
            "mutation_rate": 0.025,
            "crossover_rate": 0.4,
            "variator": [inspyred.ec.variators.n_point_crossover, inspyred.ec.variators.random_reset_mutation]
        },
        {
            "name": "npoint_inversion_low_mutation",
            "pop_size": 75,
            "mutation_rate": 0.01,
            "crossover_rate": 0.6,
            "variator": [inspyred.ec.variators.n_point_crossover, inspyred.ec.variators.inversion_mutation]
        },
        {
            "name": "npoint_inversion_high_all",
            "pop_size": 75,
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
            "variator": [inspyred.ec.variators.n_point_crossover, inspyred.ec.variators.inversion_mutation]
        }
    ]

    streamingPlans = load_streaming_plan_json("../Data/streamingPlans.json")
    platforms_indexed = load_platforms_json("../Data/indice_plataformas.json")
    user_datasets = [
        "users1.json",
        "users2.json",
        "users3.json",
        "users4.json",
        "users5.json"
    ]

    base_results_path = "../results/NSGA2"
    all_results = []

    # Crear directorio base si no existe
    os.makedirs(base_results_path, exist_ok=True)

    for dataset_name in user_datasets:
        dataset_path = f"../Data/{dataset_name}"
        print(f"\n📂 Ejecutando para dataset: {dataset_name}")
        users = load_users_from_json(dataset_path)

        dataset_results = []  # Resultados específicos del dataset

        for config in configurations:
            print(f"\n🔧 Configuración: {config['name']}")

            config_results = []

            for run in range(5):
                print(f"🔁 Run {run + 1}/5")

                # Crear estructura de directorios para este run específico
                solutions_dir, summaries_dir = create_directory_structure(
                    base_results_path, dataset_name, config['name'], run + 1
                )

                seed = time()
                prng = random.Random(seed)

                algorithm = inspyred.ec.emo.NSGA2(prng)
                bounder = inspyred.ec.Bounder(1, len(platforms_indexed))

                algorithm.selector = inspyred.ec.selectors.tournament_selection
                algorithm.replacer = inspyred.ec.replacers.nsga_replacement
                algorithm.variator = config["variator"]
                algorithm.terminator = inspyred.ec.terminators.no_improvement_termination
                algorithm.observer = observer

                args = {
                    'users': users,
                    'streamingPlans': streamingPlans,
                    'platforms_indexed': platforms_indexed,
                    'max_generations': 10,
                }

                start_time = time()
                final_pop = algorithm.evolve(
                    generator=generar_individuo,
                    evaluator=evaluator,
                    bounder=bounder,
                    pop_size=config["pop_size"],
                    maximize=False,
                    num_selected=config["pop_size"],
                    tournament_size=3,
                    num_elites=2,
                    mutation_rate=config["mutation_rate"],
                    crossover_rate=config["crossover_rate"],
                    **args
                )
                end_time = time()
                execution_time = end_time - start_time
                generations = algorithm.num_generations
                pareto_size = len(algorithm.archive)
                pareto_points = [list(ind.fitness) for ind in algorithm.archive]

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

                # Guardar soluciones individuales en la carpeta específica del run
                for idx, ind in enumerate(algorithm.archive):
                    calcular_minutos_ponderados(ind.candidate, args)
                    ind.monthly_data = args['monthly_data_by_user']
                    export = {
                        "candidate": ind.candidate,
                        "fitness": list(ind.fitness),
                        "monthly_data": ind.monthly_data
                    }

                    solution_filename = f"sol{idx}.json"
                    solution_path = os.path.join(solutions_dir, solution_filename)
                    plot_evolution()
                    plot_generation_improve()
                    plot_pareto_front(algorithm)

                    with open(solution_path, 'w', encoding='utf-8') as f:
                        json.dump(export, f, ensure_ascii=False, indent=2)

                # Guardar summary individual del run
                run_summary = {
                    'run_info': run_result,
                    'solutions_count': len(algorithm.archive),
                    'solutions_directory': solutions_dir
                }

                run_summary_path = os.path.join(solutions_dir, "run_summary.json")
                with open(run_summary_path, 'w') as f:
                    json.dump(run_summary, f, indent=2)

            # Guardar summary de la configuración específica
            config_summary_path = os.path.join(summaries_dir, f"summary_{config['name']}.json")
            with open(config_summary_path, 'w') as f:
                json.dump(config_results, f, indent=2)

            dataset_results.extend(config_results)

        # Guardar summary completo del dataset
        dataset_summary_path = os.path.join(summaries_dir, f"summary_complete_{dataset_name.replace('.json', '')}.json")
        with open(dataset_summary_path, 'w') as f:
            json.dump(dataset_results, f, indent=2)

        all_results.extend(dataset_results)

    # Guardar summary general de todos los experimentos
    general_summary_path = os.path.join(base_results_path, "summary_all_experiments.json")
    with open(general_summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Experimentos completados!")
    print(f"📁 Estructura de archivos creada en: {base_results_path}")
    print(f"📊 Summary general guardado en: {general_summary_path}")


if __name__ == "__main__":
    main()