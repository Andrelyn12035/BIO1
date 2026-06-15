"""Punto de entrada para ejecutar los experimentos del Eurotrip HGA."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_loader import ensure_dataset, output_dir


class NumpyEncoder(json.JSONEncoder):
    """Serializa tipos NumPy a tipos nativos de Python para json.dump."""

    def default(self, obj: object) -> object:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
from src.hga import HybridGeneticAlgorithm
from src.visualizer import create_itinerary_map, plot_convergence, plot_timeline


def configure_logging() -> None:
	"""Configura el registro de eventos del programa."""
	logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


def route_names(route: list[int], cities: list[dict[str, Any]], start_city: int = 0) -> list[str]:
	"""Convierte una ruta de índices a nombres de ciudad."""
	full_route = [start_city, *route, start_city]
	return [cities[index]["name"] for index in full_route]


def itinerary_rows(evaluation: Any) -> list[dict[str, Any]]:
	"""Convierte el itinerario detallado en filas tabulares."""
	rows: list[dict[str, Any]] = []
	for step in evaluation.itinerary:
		rows.append(
			{
				"origin": step["origin"],
				"destination": step["destination"],
				"departure_abs": step["departure_abs"],
				"arrival_abs": step["arrival_abs"],
				"arrival_local": step["arrival_local"],
				"modal": step["modal"],
				"duration_h": step["duration_h"],
			}
		)
	return rows


def run_single_experiment(
	*,
	cities: list[dict[str, Any]],
	distances: dict[str, Any],
	train_schedules: dict[str, Any],
	experiment_name: str,
	allow_trains: bool,
	apply_windows: bool,
	n_runs: int = 5,
	n_generations: int = 100,
	pop_size: int = 20,
	pm: float = 0.1,
	m: int = 3,
	W1: float = 100.0,
	start_city: int = 0,
	start_hour: float = 9.0,
) -> dict[str, Any]:
	"""Ejecuta un experimento completo y guarda sus artefactos."""
	base_output = output_dir() / experiment_name
	base_output.mkdir(parents=True, exist_ok=True)

	run_results: list[dict[str, Any]] = []
	histories: list[list[float]] = []

	for run_index in range(n_runs):
		rng = np.random.default_rng(2026 + run_index)
		hga = HybridGeneticAlgorithm(
			cities=cities,
			distances=distances,
			train_schedules=train_schedules,
			pop_size=pop_size,
			n_generations=n_generations,
			pm=pm,
			m=m,
			W1=W1,
			start_city=start_city,
			start_hour=start_hour,
			allow_trains=allow_trains,
			apply_windows=apply_windows,
			rng=rng,
		)
		result = hga.run()
		histories.append(result.history)
		run_results.append(
			{
				"run": run_index + 1,
				"fitness": result.best_evaluation.fitness,
				"objective_hours": result.best_evaluation.objective_hours,
				"penalty": result.best_evaluation.penalty,
				"route": result.best_route,
				"route_names": route_names(result.best_route, cities, start_city=start_city),
				"evaluation": result.best_evaluation,
			}
		)

	summary = pd.DataFrame(
		{
			"fitness": [item["fitness"] for item in run_results],
			"objective_hours": [item["objective_hours"] for item in run_results],
			"penalty": [item["penalty"] for item in run_results],
		}
	)

	best_run = min(run_results, key=lambda item: item["fitness"])
	best_evaluation = best_run["evaluation"]

	summary_path = base_output / "summary.csv"
	summary.to_csv(summary_path, index=False)

	best_route_path = base_output / "best_route.json"
	with best_route_path.open("w", encoding="utf-8") as handle:
		json.dump(
			{
				"route": best_run["route"],
				"route_names": best_run["route_names"],
				"fitness": best_run["fitness"],
				"objective_hours": best_run["objective_hours"],
				"penalty": best_run["penalty"],
			},
			handle,
			ensure_ascii=False,
			indent=2,
			cls=NumpyEncoder,
		)

	itinerary_path = base_output / "itinerary.csv"
	pd.DataFrame(itinerary_rows(best_evaluation)).to_csv(itinerary_path, index=False)

	map_path = create_itinerary_map(
		cities,
		best_evaluation,
		base_output / "itinerary_map.html",
		title=experiment_name,
		distances=distances,
	)
	convergence_path = plot_convergence(histories, base_output / "convergence.png")
	timeline_path = plot_timeline(best_evaluation, cities, base_output / "timeline.png")

	result_payload = {
		"experiment": experiment_name,
		"allow_trains": allow_trains,
		"apply_windows": apply_windows,
		"summary": {
			"best": float(summary["fitness"].min()),
			"average": float(summary["fitness"].mean()),
			"worst": float(summary["fitness"].max()),
		},
		"best_solution": {
			"route": best_run["route"],
			"route_names": best_run["route_names"],
			"fitness": best_run["fitness"],
			"objective_hours": best_run["objective_hours"],
			"penalty": best_run["penalty"],
			"full_route": best_evaluation.full_route,
			"itinerary": best_evaluation.itinerary,
		},
		"artifacts": {
			"summary_csv": str(summary_path),
			"best_route_json": str(best_route_path),
			"itinerary_csv": str(itinerary_path),
			"map_html": str(map_path),
			"convergence_png": str(convergence_path),
			"timeline_png": str(timeline_path),
		},
	}

	with (base_output / "result.json").open("w", encoding="utf-8") as handle:
		json.dump(result_payload, handle, ensure_ascii=False, indent=2, cls=NumpyEncoder)

	print(f"\n=== {experiment_name} ===")
	print(summary.to_string(index=False))
	print("Mejor ruta:", best_run["route_names"])
	print(f"Tiempo total: {best_run['objective_hours']:.2f} h")
	print(f"Fitness: {best_run['fitness']:.2f}")
	print(f"Mapa: {map_path}")
	print(f"Convergencia: {convergence_path}")
	print(f"Timeline: {timeline_path}")

	return result_payload


def main() -> None:
	"""Ejecuta los dos experimentos solicitados y guarda todos los resultados."""
	configure_logging()
	dataset = ensure_dataset()
	cities = dataset["cities"]
	distances = dataset["distances"]
	train_schedules = dataset["train_schedules"]

	run_single_experiment(
		cities=cities,
		distances=distances,
		train_schedules=train_schedules,
		experiment_name="experiment_1_restricciones",
		allow_trains=True,
		apply_windows=True,
	)

	run_single_experiment(
		cities=cities,
		distances=distances,
		train_schedules=train_schedules,
		experiment_name="experiment_2_sin_restricciones",
		allow_trains=False,
		apply_windows=False,
	)


if __name__ == "__main__":
	main()
