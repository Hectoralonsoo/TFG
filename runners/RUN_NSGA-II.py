import json
import random
from time import time
import inspyred


from Loaders.LoadStreamingPlans import load_streaming_plan_json
from Loaders.LoadUsers import load_users_from_json
from Loaders.LoadPlatforms import load_platforms_json

from utils.evaluation import evaluator, calcular_minutos_ponderados
from utils.logging_custom import observer, plot_evolution, plot_generation_improve, plot_pareto_front
from generators.Individual_generator import generar_individuo



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

    all_results = []

    for dataset_name in user_datasets:
        dataset_path = f"../Data/{dataset_name}"
        print(f"\n📂 Ejecutando para dataset: {dataset_name}")
        users = load_users_from_json(dataset_path)

        for config in configurations:
            print(f"\n🔧 Configuración: {config['name']}")
            config_results = []

            for run in range(5):
                print(f"🔁 Run {run + 1}/5")

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

                config_results.append({
                    'dataset': dataset_name,
                    'config': config['name'],
                    'run': run + 1,
                    'execution_time': execution_time,
                    'generations': generations,
                    'pareto_size': pareto_size,
                    'pareto_points': pareto_points
                })

                for idx, ind in enumerate(algorithm.archive):
                    calcular_minutos_ponderados(ind.candidate, args)
                    ind.monthly_data = args['monthly_data_by_user']
                    export = {
                        "candidate": ind.candidate,
                        "fitness": list(ind.fitness),
                        "monthly_data": ind.monthly_data
                    }

                    output_path = f"../results/NSGA2/{dataset_name.replace('.json', '')}_{config['name']}_run{run + 1}_sol{idx}.json"
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(export, f, ensure_ascii=False, indent=2)

            summary_path = f"../results/NSGA2/summary_{dataset_name.replace('.json', '')}_{config['name']}.json"
            with open(summary_path, 'w') as f:
                json.dump(config_results, f, indent=2)

            all_results.extend(config_results)

    with open("../results/NSGA2/summary_all_runs.json", 'w') as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()