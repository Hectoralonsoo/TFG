from algorithms.PACO import PACOStreaming, fitness_paco
from Loaders.LoadStreamingPlans import load_streaming_plan_json
from Loaders.LoadPlatforms import load_platforms_json
from Loaders.LoadUsers import load_users_from_json
from utils.logging import plot_pareto_front_paco
from utils.logging import plot_ant_paths_lines
from utils.save import save_pareto_archive_paco
from utils.logging import plot_user_platforms_over_time
from utils.logging import observer_paco
import json

def main():
    # Cargar datos
    streamingPlans = load_streaming_plan_json("../Data/streamingPlans.json")
    users = load_users_from_json("../Data/users1.json")

    with open("../Data/indice_plataformas.json", "r", encoding="utf-8") as f:
        platforms_indexed = json.load(f)

    plataformas_disponibles = [int(p) for p in platforms_indexed.keys()]  # Asegurar que son ints

    args = {
        'users': users,
        'streamingPlans': streamingPlans,
        'platforms_indexed': platforms_indexed
    }


    # Instanciar PACO
    paco = PACOStreaming(
        n_ants=4,
        n_iterations=15,
        n_months=12,
        n_users=len(users),
        platform_options=plataformas_disponibles,
        rho=0.4,
        alpha=1,
        beta=3,
        archive_size=100
    )

    # Optimizar
    pareto_archive = paco.optimize(lambda sol: fitness_paco(sol, args))

    save_pareto_archive_paco(pareto_archive,paco.n_users, args)

    # Mostrar resultados finales
    print("\nPareto Front encontrado:")
    for i, (solution, objectives) in enumerate(pareto_archive):
        print(f"Solución {i+1}: Minutos ponderados: {objectives[0]:.2f}, Costo total: {objectives[1]:.2f}€")

    # Usar utilidades de logging
    plot_pareto_front_paco(pareto_archive, title="Pareto Front - PACO", save_path="../pareto_paco.png")

    plot_ant_paths_lines(paco.all_solutions, n_months=12, n_users=len(users), platform_options=plataformas_disponibles)




if __name__ == "__main__":
    main()