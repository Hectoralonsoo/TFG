import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from statds.no_parametrics import friedman, shaffer

ALPHA = 0.05
CSV_PATH = "hv_matrix.csv"
FIG_PATH = "friedman_shaffer.png"
MINIMIZE = False


hv = pd.read_csv(CSV_PATH)


names = list(hv.columns)
print(names)

rankings, stat, p_value, crit_val, decision = friedman(hv, ALPHA, minimize=MINIMIZE)
print(f"Friedman: stat={stat}, p={p_value}, decision={decision}")
for n, r in rankings.items():
    print(f"{n:8s} rank={r:.3f}")

def reject_friedman(alpha, p_value, stat, crit_val, decision):
    if p_value is not None:
        return float(p_value) < alpha
    if stat is not None and crit_val is not None:
        return float(stat) > float(crit_val)
    s = str(decision).lower()
    return ("reject" in s) and ("fail" not in s)

reject = reject_friedman(ALPHA, p_value, stat, crit_val, decision)

present = [n for n in names if n in rankings]
G = nx.Graph()

if reject:
    results_shaffer, _ = shaffer(rankings, hv.shape[0], ALPHA, type_rank="Friedman")
    print("\nShaffer:")
    print(results_shaffer)

    pcols = ["p-value (Shaffer)", "Adjusted p-value", "p_adj", "p-value"]
    pcol = next((c for c in pcols if c in results_shaffer.columns), None)
    if pcol is None:
        raise ValueError(f"No encuentro p-valor en columnas {list(results_shaffer.columns)}")

    if {"Left", "Right"}.issubset(results_shaffer.columns):
        df_pairs = results_shaffer[["Left", "Right", pcol]].copy()
    elif "Comparison" in results_shaffer.columns:
        lr = results_shaffer["Comparison"].astype(str).str.split(r"\s+vs\s+", n=1, expand=True)
        lr.columns = ["Left", "Right"]
        df_pairs = pd.concat([lr, results_shaffer[[pcol]]], axis=1)
    else:
        raise ValueError(f"Formato de Shaffer desconocido: {list(results_shaffer.columns)}")

    G.add_nodes_from([(n, {"rank": float(rankings[n])}) for n in present])

    for _, row in df_pairs.iterrows():
        left, right = str(row["Left"]).strip(), str(row["Right"]).strip()
        p_adj = float(row[pcol])
        if left in present and right in present and p_adj >= ALPHA:
            G.add_edge(left, right)
else:
    G = nx.complete_graph(present)
    nx.set_node_attributes(G, {n: {"rank": float(rankings[n])} for n in present})

pos = nx.kamada_kawai_layout(G, scale=0.5)
node_ranks = [d["rank"] for _, d in G.nodes(data=True)]

fig, ax = plt.subplots(figsize=(6, 4))

nodes_artist = nx.draw_networkx_nodes(
    G, pos=pos,
    node_size=[r * 500 for r in node_ranks],
    node_color=node_ranks,
    cmap="plasma",
    ax=ax
)
nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color="#444")
nx.draw_networkx_labels(G, pos=pos, ax=ax, font_color="w")

cbar = fig.colorbar(
    nodes_artist, ax=ax, shrink=0.8, label="Ranking medio (menor = mejor)"
)

ax.set_title("Friedman + Shaffer (HV)")
ax.set_axis_off()
plt.tight_layout()
plt.savefig(FIG_PATH, dpi=300)
plt.close()
print(f"Figura guardada en '{FIG_PATH}'")
