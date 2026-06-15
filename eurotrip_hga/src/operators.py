"""Operadores genéticos del HGA: CX, remoción de abruptos y selección familiar."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _pair_key(origin_name: str, destination_name: str) -> str:
    """Construye la clave estándar de una pareja de ciudades."""
    return f"{origin_name}-{destination_name}"


def route_distance(
    route: list[int],
    cities: list[dict[str, Any]],
    distances: dict[str, Any],
    *,
    start_city: int = 0,
) -> float:
    """Calcula la distancia total de una ruta cerrada usando kilómetros viales.

    Cuando un par no tiene ruta vial (km es null), se usa un valor centinela
    grande para que la heurística local evite ese arco.
    """
    full_route = [start_city, *route, start_city]
    total = 0.0
    for left, right in zip(full_route[:-1], full_route[1:]):
        key = _pair_key(cities[left]["name"], cities[right]["name"])
        km_val = distances["pairs"][key].get("km")
        total += float(km_val) if km_val is not None else 99_999.0
    return total


def cycle_crossover(parent_a: list[int], parent_b: list[int]) -> list[int]:
    """Aplica Cycle Crossover tomando el ciclo que arranca en la posición cero."""
    size = len(parent_a)
    if size != len(parent_b):
        raise ValueError("Los padres deben tener la misma longitud.")
    if size == 0:
        return []

    child = list(parent_b)
    index_from_value_b = {value: index for index, value in enumerate(parent_b)}
    cycle_indices: list[int] = []
    current_index = 0

    while current_index not in cycle_indices:
        cycle_indices.append(current_index)
        current_value = parent_a[current_index]
        current_index = index_from_value_b[current_value]

    for index in cycle_indices:
        child[index] = parent_a[index]
    return child


def _nearest_neighbor_positions(
    city: int,
    route: list[int],
    cities: list[dict[str, Any]],
    distances: dict[str, Any],
    m: int,
) -> list[int]:
    """Devuelve las posiciones candidatas alrededor de los m vecinos más cercanos."""
    full_candidates = []
    for candidate in route:
        if candidate == city:
            continue
        key = _pair_key(cities[city]["name"], cities[candidate]["name"])
        km_val = distances["pairs"][key].get("km")
        full_candidates.append((float(km_val) if km_val is not None else 99_999.0, candidate))

    nearest = [candidate for _, candidate in sorted(full_candidates, key=lambda item: item[0])[:m]]
    positions: set[int] = {0, len(route)}
    for neighbor in nearest:
        if neighbor in route:
            index = route.index(neighbor)
            positions.add(index)
            positions.add(index + 1)
    return sorted(pos for pos in positions if 0 <= pos <= len(route))


def abrupt_removal(
    route: list[int],
    cities: list[dict[str, Any]],
    distances: dict[str, Any],
    *,
    m: int = 3,
    start_city: int = 0,
) -> list[int]:
    """Mejora local por reubicación guiada por los vecinos más cercanos."""
    if len(route) <= 2:
        return list(route)

    improved = list(route)
    changed = True
    while changed:
        changed = False
        current_distance = route_distance(improved, cities, distances, start_city=start_city)
        for index, city in enumerate(list(improved)):
            reduced_route = improved[:index] + improved[index + 1 :]
            candidate_positions = _nearest_neighbor_positions(city, reduced_route, cities, distances, m)
            best_route = improved
            best_distance = current_distance
            for position in candidate_positions:
                candidate = reduced_route[:position] + [city] + reduced_route[position:]
                candidate_distance = route_distance(candidate, cities, distances, start_city=start_city)
                if candidate_distance + 1e-9 < best_distance:
                    best_distance = candidate_distance
                    best_route = candidate
            if best_route != improved:
                improved = best_route
                changed = True
                break
    return improved


def family_selection(
    parent_a: dict[str, Any],
    parent_b: dict[str, Any],
    child_a: dict[str, Any],
    child_b: dict[str, Any],
) -> list[dict[str, Any]]:
    """Selecciona los dos mejores individuos de la familia de cuatro."""
    family = [parent_a, parent_b, child_a, child_b]
    family.sort(key=lambda individual: float(individual["fitness"]))
    return family[:2]


def inject_random_individual(
    population: list[dict[str, Any]],
    candidate_cities: list[int],
    rng: np.random.Generator,
) -> None:
    """Sustituye un individuo aleatorio por una permutación nueva."""
    if not population:
        return
    index = int(rng.integers(0, len(population)))
    route = list(rng.permutation(candidate_cities))
    population[index] = {"route": route}
