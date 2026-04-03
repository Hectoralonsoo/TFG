# Optimización de Suscripciones a Plataformas de Streaming mediante Algoritmos Evolutivos Multi-objetivo

**Trabajo de Fin de Grado — Héctor Alonso de las Heras**

Este proyecto implementa y compara tres algoritmos evolutivos multi-objetivo (NSGA-II, SPEA2 y PACO) para resolver el problema de selección óptima de plataformas de streaming. Dado un conjunto de usuarios con preferencias sobre películas y series, el objetivo es encontrar la combinación mensual de plataformas que maximice los minutos de contenido ponderado visualizado y minimice el coste total de las suscripciones a lo largo de 12 meses.

---

## Índice

1. [Descripción del problema](#descripción-del-problema)
2. [Estructura del proyecto](#estructura-del-proyecto)
3. [Instalación](#instalación)
4. [Datos disponibles](#datos-disponibles)
5. [Cómo ejecutar los algoritmos](#cómo-ejecutar-los-algoritmos)
6. [Cómo ejecutar el análisis estadístico](#cómo-ejecutar-el-análisis-estadístico)
7. [Configuraciones de los algoritmos](#configuraciones-de-los-algoritmos)
8. [Formato de los resultados](#formato-de-los-resultados)

---

## Descripción del problema

Dado un grupo de usuarios con preferencias de contenido (películas y series) y un presupuesto mensual de minutos de visualización, se busca decidir para cada usuario y cada mes del año qué plataforma de streaming contratar. La solución es un **frente de Pareto** de soluciones no dominadas que equilibra:

- **Objetivo 1 (maximizar):** Minutos ponderados visualizados — suma de duración × nivel de interés de todo el contenido accesible.
- **Objetivo 2 (minimizar):** Coste total anual — calculado buscando la combinación óptima de planes por plataforma mediante programación dinámica.

Una solución se representa como una matriz `[usuarios × 12 meses]` donde cada celda contiene el ID de la plataforma contratada ese mes.

---

## Estructura del proyecto

```
TFG_Hector/
├── models/          # Clases de datos: User, Movie, Serie, StreamingPlan, Content
├── loaders/         # Carga de JSON: usuarios, películas, series, planes, plataformas
├── algorithms/      # PACO.py (ACO multi-objetivo), SPEA2.py (custom sobre inspyred)
├── generators/      # Generador de individuos iniciales aleatorios
├── utils/
│   ├── evaluation.py      # Funciones de fitness: minutos ponderados, coste total, evaluadores
│   ├── heuristics.py      # Programación dinámica para combinación óptima de planes
│   └── logging_custom.py  # Observers, visualización de evolución y frentes de Pareto
├── runners/         # Scripts de ejecución de experimentos
│   ├── run_nsga2.py       # NSGA-II — ejecuciones de prueba (5 runs)
│   ├── run_nsga2_31.py    # NSGA-II — ejecuciones estadísticas (31 runs)
│   ├── run_paco.py        # PACO — ejecuciones de prueba
│   ├── run_paco_31.py     # PACO — ejecuciones estadísticas
│   ├── run_spea2.py       # SPEA2 — ejecuciones de prueba
│   ├── run_spea2_31.py    # SPEA2 — ejecuciones estadísticas
│   └── evolution_tracker.py  # Seguimiento de métricas por generación
├── analysis/        # Análisis estadístico post-ejecución
├── scripts/         # Generador de usuarios sintéticos, loaders de API TMDB
├── data/            # Datasets JSON: películas, series, usuarios, planes de streaming
├── results/         # Resultados de las 31 ejecuciones por algoritmo/dataset
├── results_test/    # Resultados de ejecuciones de prueba
├── outputs/         # Gráficas y CSVs del análisis estadístico
└── requirements.txt
```

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd TFG_Hector
```

### 2. Crear y activar un entorno virtual

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Datos disponibles

La carpeta `data/` contiene los datasets listos para usar:

| Archivo | Descripción |
|---|---|
| `movies.json` | Catálogo completo de películas con duración y plataformas |
| `series.json` | Catálogo de series con temporadas, episodios y plataformas |
| `users1.json` | Dataset pequeño (~4 MB) — pocos usuarios, pruebas rápidas |
| `users2.json` | Dataset pequeño-medio (~6 MB) |
| `users3.json` | Dataset medio (~10 MB) |
| `users4.json` | Dataset grande (~17 MB) |
| `users5.json` | Dataset muy grande (~29 MB) — mayor diversidad de preferencias |
| `streamingPlans.json` | Planes y precios por plataforma (Netflix, HBO, Prime, etc.) |
| `indice_plataformas.json` | Mapeo ID → nombre de plataforma |

Para pruebas rápidas se recomienda empezar con `users1.json`.

---

## Cómo ejecutar los algoritmos

Todos los runners deben ejecutarse **desde la raíz del proyecto**:

```bash
cd TFG_Hector
```

### NSGA-II

```bash
# Ejecución de prueba: 6 configuraciones × 5 runs × 5 datasets
python runners/run_nsga2.py

# Ejecución estadística: mejor configuración × 31 runs × 5 datasets
python runners/run_nsga2_31.py
```

### SPEA2

```bash
# Ejecución de prueba: 6 configuraciones × 5 runs × 5 datasets
python runners/run_spea2.py

# Ejecución estadística: mejor configuración × 31 runs × 5 datasets
python runners/run_spea2_31.py
```

### PACO (Ant Colony Optimization multi-objetivo)

```bash
# Ejecución de prueba: 6 configuraciones × 5 runs × 5 datasets
python runners/run_paco.py

# Ejecución estadística: mejor configuración × 31 runs × 5 datasets
python runners/run_paco_31.py
```

### Resultados de salida

Los resultados se guardan automáticamente en:

- **Ejecuciones de prueba:** `results_test/{ALGORITMO}/solutions/{dataset}/{config}/run{N}/`
- **Ejecuciones estadísticas:** `results/{ALGORITMO}/solutions/{dataset}/{config}/run{N}/`

Cada carpeta de run contiene:
- `sol0.json`, `sol1.json`, ... — soluciones del frente de Pareto final
- `run_summary.json` — métricas de la ejecución (tiempo, generaciones, tamaño del frente)

---

## Cómo ejecutar el análisis estadístico

Una vez completadas las 31 ejecuciones, se pueden calcular métricas comparativas desde la carpeta `analysis/`:

```bash
# Análisis de hipervolumen unificado (los 3 algoritmos)
python analysis/unified_hv_analysis.py

# Análisis por algoritmo individual
python analysis/hv_analysis_nsga.py
python analysis/hv_analysis_spea2.py
python analysis/hv_analysis_paco.py

# Comparación estadística entre algoritmos (test de Friedman + Shaffer)
python analysis/Comparador_algoritmos.py

# Análisis de robustez (consistencia entre las 31 ejecuciones)
python analysis/robustez_plots.py

# Gráficas de ejecuciones individuales
python analysis/individual_runs.py
```

Los resultados (gráficas PNG y tablas CSV) se guardan en `outputs/`.

---

## Configuraciones de los algoritmos

### NSGA-II y SPEA2 — Operadores genéticos probados

| Nombre | Cruce | Mutación | p_cruce | p_mutación |
|---|---|---|---|---|
| `uniform_reset_low_mutation` | Uniforme | Reset aleatorio | 0.6 | 0.01 |
| `uniform_reset_high_crossover` | Uniforme | Reset aleatorio | 0.8 | 0.025 |
| `uniform_inversion_high_mutation` | Uniforme | Inversión | 0.6 | 0.1 |
| `npoint_reset_low_crossover` | N-punto | Reset aleatorio | 0.4 | 0.025 |
| `npoint_inversion_low_mutation` | N-punto | Inversión | 0.6 | 0.01 |
| `npoint_inversion_high_all` | N-punto | Inversión | 0.8 | 0.1 |

- **Tamaño de población:** 75 (NSGA-II) / 50 (SPEA2)
- **Terminación:** sin mejora en el frente de Pareto durante N generaciones

### PACO — Parámetros de colonia de hormigas

| Nombre | ρ (evaporación) | α (feromona) | β (heurística) |
|---|---|---|---|
| `balanced_original` | 0.4 | 1.0 | 3.0 |
| `high_pheromone_influence` | 0.3 | 3.0 | 2.0 |
| `high_heuristic_influence` | 0.4 | 0.5 | 5.0 |
| `slow_evaporation_exploitative` | 0.2 | 2.0 | 3.0 |
| `fast_evaporation_explorative` | 0.7 | 1.0 | 2.5 |
| `minimal_guidance` | 0.5 | 0.2 | 1.0 |

- **Hormigas:** 20 | **Iteraciones máx.:** 200 | **Archivo:** 100 soluciones

---

## Formato de los resultados

### Solución individual (`solN.json`)

```json
{
  "candidate": [[1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3],
                [2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3]],
  "fitness": [-125430.5, 287.40],
  "monthly_data": {
    "user_0": [
      {
        "month": 0,
        "platform": 1,
        "watched_movies": [...],
        "watched_series": [...]
      }
    ]
  }
}
```

- `candidate`: lista de listas `[usuario][mes]` con el ID de plataforma asignado
- `fitness`: `[-minutos_ponderados, coste_total_anual_€]`
- `monthly_data`: contenido visualizado por usuario y mes

### Resumen de ejecución (`run_summary.json`)

```json
{
  "run_info": {
    "dataset": "users1.json",
    "config": "uniform_inversion_high_mutation",
    "run": 1,
    "execution_time": 142.3,
    "generations": 87,
    "pareto_size": 23
  },
  "solutions_count": 23,
  "solutions_directory": "results/NSGA2/solutions/users1/..."
}
```
