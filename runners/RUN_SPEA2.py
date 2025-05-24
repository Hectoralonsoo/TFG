import random
from time import time

import inspyred

from Loaders.LoadStreamingPlans import load_streaming_plan_json
from Loaders.LoadUsers import load_users_from_json
from Loaders.LoadPlatforms import load_platforms_json

from generators.Individual_generator import generar_individuo
from utils.evaluation import calcular_costo_total, calcular_minutos_ponderados
from utils.logging import observador_spea2, plot_pareto_front_spea2
from algorithms.SPEA2 import SPEA2
from utils.evaluation import evaluatorSPEA2
#from utils.save import save_pareto_archive_ea



def main():
    # Cargar datos
    streamingPlans = load_streaming_plan_json("../Data/streamingPlans.json")
    users = load_users_from_json("../Data/users.json")
    platforms_indexed = load_platforms_json("../Data/indice_plataformas.json")

    seed = int(time())
    print(f"🎲 Seed: {seed}")
    prng = random.Random(seed)

    # Configurar algorithmritmo SPEA2
    algorithm = SPEA2(prng)
    algorithm.archive_size = 30
    algorithm.selector = inspyred.ec.selectors.tournament_selection
    algorithm.generator = generar_individuo
    algorithm.evaluator = evaluatorSPEA2
    algorithm.variator = algorithm.default_variator
    algorithm.terminator = algorithm.default_terminator
    algorithm.observer = observador_spea2

    bounder = inspyred.ec.Bounder(1, len(platforms_indexed))
    max_gen = 30
    pop_size = 20

    args = {
        'users': users,
        'streamingPlans': streamingPlans,
        'platforms_indexed': platforms_indexed,
        'max_generations': max_gen,
    }

    # Ejecutar evolución
    final_pop = algorithm.evolve(
        generator=generar_individuo,
        pop_size=pop_size,
        maximize=False,
        bounder=bounder,
        num_selected=pop_size,
        tournament_size=3,
        num_elites=2,
        mutation_rate=0.5,
        crossover_rate=0.5,
        gaussian_stdev=0.7,
        **args
    )

    print("✅ Evolución SPEA2 completada.")
    plot_pareto_front_spea2(algorithm)

    # save_pareto_archive_ea(algorithm.archive, n_users=len(users), path="soluciones_SPEA2.json")


if __name__ == "__main__":
    main()
