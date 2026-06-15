"""Implementación del Algoritmo Genético Híbrido para Eurotrip."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .fitness import RouteEvaluation, evaluate_route
from .operators import abrupt_removal, cycle_crossover, family_selection


@dataclass
class HGAResult:
    """Empaqueta el mejor individuo encontrado y su historial."""

    best_route: list[int]
    best_evaluation: RouteEvaluation
    history: list[float]
    population_history: list[float]

    def to_dict(self) -> dict[str, Any]:
        """Convierte el resultado a un diccionario serializable."""
        return {
            "best_route": self.best_route,
            "best_evaluation": self.best_evaluation.to_dict(),
            "history": self.history,
            "population_history": self.population_history,
        }


class HybridGeneticAlgorithm:
    """Algoritmo genético híbrido con CX, mejora local y selección familiar."""

    def __init__(
        self,
        *,
        cities: list[dict[str, Any]],
        distances: dict[str, Any],
        train_schedules: dict[str, Any],
        pop_size: int = 20,
        n_generations: int = 100,
        pm: float = 0.1,
        m: int = 3,
        W1: float = 100.0,
        start_city: int = 0,
        start_hour: float = 9.0,
        allow_trains: bool = True,
        apply_windows: bool = True,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Inicializa el optimizador con los datos y parámetros del experimento."""
        self.cities = cities
        self.distances = distances
        self.train_schedules = train_schedules
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.pm = pm
        self.m = m
        self.W1 = W1
        self.start_city = start_city
        self.start_hour = start_hour
        self.allow_trains = allow_trains
        self.apply_windows = apply_windows
        self.rng = rng or np.random.default_rng()
        self.candidate_cities = [index for index in range(len(cities)) if index != self.start_city]

    def _random_route(self) -> list[int]:
        """Genera una permutación aleatoria excluyendo la ciudad de inicio."""
        return list(self.rng.permutation(self.candidate_cities))

    def _evaluate(self, route: list[int]) -> RouteEvaluation:
        """Evalúa una ruta candidata con la función objetivo configurada."""
        return evaluate_route(
            route,
            self.cities,
            self.distances,
            self.train_schedules,
            start_city=self.start_city,
            start_hour=self.start_hour,
            allow_trains=self.allow_trains,
            apply_windows=self.apply_windows,
            W1=self.W1,
        )

    def _improve(self, route: list[int]) -> list[int]:
        """Aplica remoción de abruptos como mejora local."""
        return abrupt_removal(
            route,
            self.cities,
            self.distances,
            m=self.m,
            start_city=self.start_city,
        )

    def _build_individual(self, route: list[int]) -> dict[str, Any]:
        """Construye la estructura estándar de individuo del algoritmo."""
        evaluation = self._evaluate(route)
        return {"route": route, "fitness": evaluation.fitness, "evaluation": evaluation}

    def run(self) -> HGAResult:
        """Ejecuta el HGA completo y devuelve el mejor recorrido hallado."""
        population = [self._build_individual(self._random_route()) for _ in range(self.pop_size)]
        for index, individual in enumerate(population):
            improved_route = self._improve(individual["route"])
            population[index] = self._build_individual(improved_route)

        population.sort(key=lambda individual: float(individual["fitness"]))
        history: list[float] = [float(population[0]["fitness"])]
        population_history: list[float] = [float(individual["fitness"]) for individual in population]

        for _generation in range(self.n_generations):
            next_population: list[dict[str, Any]] = []
            shuffled_indices = list(self.rng.permutation(len(population)))
            if len(shuffled_indices) % 2 == 1:
                shuffled_indices.append(shuffled_indices[-1])

            for index in range(0, len(shuffled_indices), 2):
                parent_a = population[shuffled_indices[index]]
                parent_b = population[shuffled_indices[index + 1]]

                child_a_route = cycle_crossover(parent_a["route"], parent_b["route"])
                child_b_route = cycle_crossover(parent_b["route"], parent_a["route"])

                child_a_route = self._improve(child_a_route)
                child_b_route = self._improve(child_b_route)

                child_a = self._build_individual(child_a_route)
                child_b = self._build_individual(child_b_route)
                survivors = family_selection(parent_a, parent_b, child_a, child_b)
                next_population.extend(survivors)

            if len(next_population) > self.pop_size:
                next_population.sort(key=lambda individual: float(individual["fitness"]))
                next_population = next_population[: self.pop_size]

            if self.rng.random() < self.pm:
                random_index = int(self.rng.integers(0, len(next_population)))
                injected_route = self._improve(self._random_route())
                next_population[random_index] = self._build_individual(injected_route)

            next_population.sort(key=lambda individual: float(individual["fitness"]))
            population = next_population
            population_history.extend(float(individual["fitness"]) for individual in population)
            history.append(float(population[0]["fitness"]))

        best_individual = population[0]
        return HGAResult(
            best_route=list(best_individual["route"]),
            best_evaluation=best_individual["evaluation"],
            history=history,
            population_history=population_history,
        )
