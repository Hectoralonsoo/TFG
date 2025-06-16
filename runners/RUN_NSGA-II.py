import json
import random
from time import time
import inspyred


from Loaders.LoadStreamingPlans import load_streaming_plan_json
from Loaders.LoadUsers import load_users_from_json
from Loaders.LoadPlatforms import load_platforms_json

from utils.evaluation import evaluator, calcular_minutos_ponderados
from utils.logging import observer, plot_evolution, plot_generation_improve, plot_pareto_front
from generators.Individual_generator import generar_individuo



#from utils.save import save_pareto_archive_ea


def main():

    streamingPlans = load_streaming_plan_json("../Data/streamingPlans.json")
    users = load_users_from_json("../Data/users.json")
    platforms_indexed = load_platforms_json("../Data/indice_plataformas.json")

    num_runs = 5
    results = []
    for run in range(num_runs):
        print(f"\nEjecutando run {run + 1} / {num_runs}")

        seed = time()
        print(f"\n🎲 Random seed: {seed}")
        prng = random.Random(seed)

        # Configurar NSGA-II
        algorithm = inspyred.ec.emo.NSGA2(prng)
        bounder = inspyred.ec.Bounder(1, len(platforms_indexed))

        algorithm.selector = inspyred.ec.selectors.tournament_selection
        algorithm.replacer = inspyred.ec.replacers.nsga_replacement
        algorithm.variator = [
            inspyred.ec.variators.uniform_crossover,
            inspyred.ec.variators.random_reset_mutation
        ]

        algorithm.terminator = inspyred.ec.terminators.no_improvement_termination
        algorithm.observer = observer
       # algorithm.observer = inspyred.ec.observers.stats_observer

        # Parámetros del algoritmo
        max_gen = 200
        pop_size = 75

        args = {
            'users': users,
            'streamingPlans': streamingPlans,
            'platforms_indexed': platforms_indexed,
            'max_generations': max_gen,
            'max_generations_no_improve': 10
        }

        start_time = time()

        print(args['users'])

        # Ejecutar algoritmo
        print("\n🚀 Iniciando NSGA-II...")
        final_pop = algorithm.evolve(
            generator=generar_individuo,
            evaluator=evaluator,
            bounder=bounder,
            pop_size=pop_size,
            maximize=False,
            num_selected=pop_size,
            tournament_size=3,
            num_elites=2,
            mutation_rate=0.02,
            crossover_rate=0.5,
            **args
        )
        end_time = time()

        execution_time = end_time - start_time
        generations = algorithm.num_generations
        pareto_size = len(algorithm.archive)

        pareto_points = [list(ind.fitness) for ind in algorithm.archive]

        print(f"⏱ Tiempo ejecución: {execution_time:.2f} segundos")
        print(f"📊 Generaciones ejecutadas: {generations}")
        print(f"🎯 Tamaño frente Pareto: {pareto_size}")

        results.append({
            'run': run + 1,
            'execution_time': execution_time,
            'generations': generations,
            'pareto_size': pareto_size,
            'pareto_points': pareto_points
        })

        archive = algorithm.archive

        for idx, ind in enumerate(archive):
            output_path = f"../results/NSGA2/pareto_solution_{idx}NSGA-II-3Prueba.json"

            calcular_minutos_ponderados(ind.candidate, args)
            ind.monthly_data = args['monthly_data_by_user']

            export = {
                "candidate": ind.candidate,
                "fitness": list(ind.fitness),
                "monthly_data": ind.monthly_data
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export, f, ensure_ascii=False, indent=2)

        print("\n✅ Evolución completada.")

        import pprint
        pprint.pprint(vars(algorithm))

        # Gráficas
       # plot_evolution()
       # plot_generation_improve()
      #  plot_pareto_front(algorithm)

    with open("../results/NSGA2/summary_runs.json", 'w') as f:
        json.dump(results, f, indent=2)






if __name__ == "__main__":
    main()