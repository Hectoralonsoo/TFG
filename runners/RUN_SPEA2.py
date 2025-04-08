import random
from time import time

from Loaders.LoadStreamingPlans import load_streaming_plan_json
from Loaders.LoadUsers import load_users_from_json
from Loaders.LoadPlatforms import load_platforms_json

from generators.Individual_generator import generar_individuo
from utils.evaluation import calcular_costo_total, calcular_minutos_ponderados
from utils.logging import observador_spea2, plot_pareto_front_spea2
from algorithms.SPEA2 import SPEA2
from utils.evaluation import evaluatorSPEA2


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
    algorithm.selector = algorithm.default_selector  # puedes cambiarlo si tienes uno propio
    algorithm.generator = generar_individuo
    algorithm.evaluator = evaluatorSPEA2
    algorithm.variator = algorithm.default_variator
    algorithm.terminator = algorithm.default_terminator
    algorithm.observer = observador_spea2

    max_generations = 50
    pop_size = 30

    args = {
        'users': users,
        'streamingPlans': streamingPlans,
        'platforms_indexed': platforms_indexed,
    }

    # Ejecutar evolución
    final_pop = algorithm.evolve(
        pop_size=pop_size,
        maximize=False,
        max_generations=max_generations,
        num_selected=pop_size,
        tournament_size=3,
        num_elites=2,
        mutation_rate=0.5,
        crossover_rate=0.5,
        gaussian_stdev=0.8,
        **args
    )

    print("✅ Evolución SPEA2 completada.")
    plot_pareto_front_spea2(algorithm)


if __name__ == "__main__":
    main()
