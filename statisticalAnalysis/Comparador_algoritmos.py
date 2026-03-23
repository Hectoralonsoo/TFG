import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from statds.no_parametrics import friedman, shaffer
import os

# Configuración Global
ALPHA = 0.05


def run_analysis(csv_path, metric_name, minimize, output_img):
    print(f"\n{'=' * 40}")


    print(f"Analizando: {metric_name}")
    print(f"Archivo: {csv_path} | Minimizar: {minimize}")
    print(f"{'=' * 40}")

    if not os.path.exists(csv_path):
        print(f"ERROR: No se encuentra el archivo {csv_path}. Saltando...")
        return

    # 1. Cargar datos
    data = pd.read_csv(csv_path)
    names = list(data.columns)

    # 2. Test de Friedman
    rankings, stat, p_value, crit_val, decision = friedman(data, ALPHA, minimize=minimize)

    print(f"Friedman Stat: {stat:.4f}")
    print(f"P-value: {p_value}")
    print(f"Decisión: {decision}")
    print("Rankings medios:")
    for n, r in rankings.items():
        print(f"  {n:8s}: {r:.3f}")


    # Determinar si se rechaza H0
    def reject_friedman(alpha, p_val, st, c_val, dec):
        if p_val is not None:
            return float(p_val) < alpha
        if st is not None and c_val is not None:
            return float(st) > float(c_val)
        s = str(dec).lower()
        return ("reject" in s) and ("fail" not in s)


    reject = reject_friedman(ALPHA, p_value, stat, crit_val, decision)

    # 3. Preparar grafo
    present = [n for n in names if n in rankings]
    G = nx.Graph()

    # Añadir nodos con sus rankings
    G.add_nodes_from([(n, {"rank": float(rankings[n])}) for n in present])

    # 4. Test de Shaffer (si procede)
    if reject:
        results_shaffer, _ = shaffer(rankings, data.shape[0], ALPHA, type_rank="Friedman")
        print("\nResultados Shaffer (Diferencias significativas):")
        # print(results_shaffer)

        pcols = ["p-value (Shaffer)", "Adjusted p-value", "p_adj", "p-value"]
        pcol = next((c for c in pcols if c in results_shaffer.columns), None)
        if pcol:
            if "Comparison" in results_shaffer.columns:
                for _, row in results_shaffer.iterrows():
                    parts = str(row["Comparison"]).split(" vs ")
                    if len(parts) == 2:
                        left, right = parts[0].strip(), parts[1].strip()
                        p_adj = float(row[pcol])
                        if left in present and right in present and p_adj >= ALPHA:
                            G.add_edge(left, right)
            else:
                print("Formato de columnas de Shaffer no reconocido para el grafo.")
    else:
        print("\nNo hay diferencias significativas (Friedman no rechaza H0).")
        G = nx.complete_graph(present)

    # 5. Dibujar y guardar grafo
    pos = nx.kamada_kawai_layout(G, scale=0.5)
    node_ranks = [rankings[n] for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(7, 5))  # Un poco más ancho para evitar cortes

    # AUMENTO DE TAMAÑO: Base 2500 para que entre el texto, + 1500 por ranking
    sizes = [2500 + (3.0 - r) * 1500 for r in node_ranks]

    cmap_name = "plasma_r"

    nodes_artist = nx.draw_networkx_nodes(
        G,
        pos=pos,
        node_size=sizes,
        node_color=node_ranks,
        cmap=cmap_name,
        ax=ax
    )

    # Aristas
    nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color="#555", width=1.5)

    # --- LÓGICA DE CONTRASTE DEL TEXTO ---
    norm = mcolors.Normalize(vmin=min(node_ranks), vmax=max(node_ranks))
    cmap = plt.get_cmap(cmap_name)

    for node, (x, y) in pos.items():
        rank = rankings[node]
        rgba = cmap(norm(rank))
        # Luminancia para decidir color de fuente
        luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        text_color = "white" if luminance < 0.5 else "black"
        plt.text(
            x, y,
            node,
            fontsize=10,
            fontweight='bold',
            color=text_color,
            ha='center',
            va='center'
        )
    # -------------------------------------

    # Barra de color
    cbar = fig.colorbar(nodes_artist, ax=ax, shrink=0.8)
    cbar.set_label("Ranking Medio (Menor es mejor)")

    ax.set_title(f"Friedman + Shaffer: {metric_name}")

    # --- FIX CORTE DE NODOS ---
    ax.axis('off')  # Quitamos ejes
    ax.margins(0.25)  # Añadimos 25% de margen extra alrededor de los nodos
    # --------------------------

    # Guardamos con bbox_inches='tight' para que incluya los márgenes que acabamos de dar
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfica guardada en: {output_img}")

mis_analisis = [
    {
        "file": "hv_matrix.csv",
        "name": "Hipervolumen",
        "min": False,
        "out": "friedman_hv.png"
    },
    {
        "file": "time_matrix.csv",
        "name": "Tiempo de Ejecución",
        "min": True,
        "out": "friedman_time.png"
    },
    {
        "file": "gens_matrix.csv",
        "name": "Generaciones",
        "min": True,
        "out": "friedman_gens.png"
    },
    {
        "file": "pareto_matrix.csv",
        "name": "Tamaño del Frente",
        "min": False,
        "out": "friedman_pareto.png"
    }
]

for analisis in mis_analisis:
    run_analysis(
        csv_path=analisis["file"],
        metric_name=analisis["name"],
        minimize=analisis["min"],
        output_img=analisis["out"]
    )