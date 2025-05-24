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

    algorithm.terminator = [inspyred.ec.terminators.generation_termination]
    algorithm.observer = observer

    # Parámetros del algoritmo
    max_gen = 30
    pop_size = 25

    args = {
        'users': users,
        'streamingPlans': streamingPlans,
        'platforms_indexed': platforms_indexed,
        'max_generations': max_gen
    }


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
        mutation_rate=0.5,
        crossover_rate=0.5,
        gaussian_stdev=0.8,
        **args
    )

    print("\n✅ Evolución completada.")

    import pprint
    pprint.pprint(vars(algorithm))

    # Gráficas
    plot_evolution()
    plot_generation_improve()
    plot_pareto_front(algorithm)






if __name__ == "__main__":
    main()