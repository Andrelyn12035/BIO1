"""Intento de descarga de datos ferroviarios reales con respaldo sintético."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)

REAL_SOURCES: tuple[tuple[str, str], ...] = (
    ("Deutsche Bahn", "https://data.deutschebahn.com/dataset/data-strecke"),
    (
        "SNCF Open Data",
        "https://ressources.data.sncf.com/api/explore/v2.1/catalog/datasets/horaires-des-trains-voyageurs-tgv-inouigo/exports/json",
    ),
    (
        "Transitland",
        "https://transit.land/api/v2/rest/routes?operator_onestop_id=o-u0-sncf",
    ),
)

DIRECT_TRAIN_CORRIDORS: set[frozenset[str]] = {
    frozenset(("Madrid", "Barcelona")),
    frozenset(("Madrid", "París")),
    frozenset(("Madrid", "Lisboa")),
    frozenset(("Barcelona", "París")),
    frozenset(("París", "Bruselas")),
    frozenset(("París", "Amsterdam")),
    frozenset(("París", "Frankfurt")),
    frozenset(("París", "Zúrich")),
    frozenset(("Bruselas", "Amsterdam")),
    frozenset(("Bruselas", "Frankfurt")),
    frozenset(("Amsterdam", "Berlín")),
    frozenset(("Frankfurt", "Berlín")),
    frozenset(("Frankfurt", "Múnich")),
    frozenset(("Frankfurt", "Zúrich")),
    frozenset(("Múnich", "Zúrich")),
    frozenset(("Múnich", "Viena")),
    frozenset(("Múnich", "Praga")),
    frozenset(("Viena", "Praga")),
    frozenset(("Viena", "Budapest")),
    frozenset(("Praga", "Berlín")),
    frozenset(("Berlín", "Amsterdam")),
    frozenset(("Berlín", "Múnich")),
    frozenset(("Zúrich", "Milán")),
    frozenset(("Milán", "Roma")),
    frozenset(("Milán", "Zúrich")),
    frozenset(("Roma", "Viena")),
}


def _fetch_json(url: str) -> Any:
    """Descarga un JSON desde una URL con tiempo de espera acotado."""
    response = requests.get(
        url,
        timeout=10,
        headers={"User-Agent": "eurotrip-hga/1.0"},
    )
    response.raise_for_status()
    return response.json()


def _probe_real_sources() -> tuple[bool, list[str]]:
    """Comprueba si las fuentes públicas responden sin error."""
    available_sources: list[str] = []
    for source_name, url in REAL_SOURCES:
        try:
            payload = _fetch_json(url)
            if payload is not None:
                available_sources.append(source_name)
                LOGGER.info("Fuente real accesible: %s", source_name)
        except Exception as exc:  # pragma: no cover - depende de red externa
            LOGGER.warning("No fue posible consultar %s: %s", source_name, exc)
    return bool(available_sources), available_sources


def _train_departures(distance_km: float, corridor_strength: float) -> list[dict[str, Any]]:
    """Genera salidas diarias realistas para un corredor ferroviario."""
    base_duration = max(1.0, (distance_km / 180.0) * corridor_strength + 0.35)
    departures: list[dict[str, Any]] = []
    schedule_template = [6.0, 8.5, 12.0, 16.0, 19.5]
    for depart in schedule_template:
        arrive = depart + base_duration
        departures.append(
            {
                "depart": round(depart, 2),
                "arrive": round(arrive, 2),
                "duration_h": round(base_duration, 2),
                "operator": "DB/SNCF/Local",
                "days": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            }
        )
    return departures


def _corridor_strength(origin: str, destination: str) -> float:
    """Devuelve un multiplicador de velocidad para un corredor dado."""
    special_fast = {
        frozenset(("Madrid", "Barcelona")),
        frozenset(("París", "Bruselas")),
        frozenset(("París", "Amsterdam")),
        frozenset(("Frankfurt", "Múnich")),
        frozenset(("Múnich", "Viena")),
        frozenset(("Viena", "Budapest")),
        frozenset(("Milán", "Roma")),
    }
    if frozenset((origin, destination)) in special_fast:
        return 0.72
    return 0.82


def build_train_schedules(
    cities: list[dict[str, Any]],
    distances: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    """Construye horarios de tren y guarda un respaldo sintético reproducible."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    real_sources_ok, real_sources = _probe_real_sources()
    if real_sources_ok:
        LOGGER.info(
            "Se consultaron fuentes reales (%s), pero se usará una malla sintética controlada para garantizar ejecución reproducible.",
            ", ".join(real_sources),
        )
    else:
        LOGGER.warning(
            "No se logró reconstruir un GTFS completo desde las fuentes públicas; se usará un calendario sintético realista."
        )

    schedules: dict[str, Any] = {}
    city_lookup = {city["name"]: city for city in cities}

    for origin in cities:
        for destination in cities:
            if origin["name"] == destination["name"]:
                continue

            key = f"{origin['name']}-{destination['name']}"
            pair_info = distances["pairs"][key]
            car_hours = pair_info.get("car_hours")  # None si no hay ruta vial

            # Para estimar duración de tren usamos la distancia en línea recta
            # (los trenes de alta velocidad son más directos que las carreteras).
            km_for_train = pair_info.get("km_straight") or pair_info.get("km") or 0.0
            km_road = pair_info.get("km")

            corridor = frozenset((origin["name"], destination["name"]))
            has_long_enough_road = km_road is not None and km_road <= 900
            if corridor in DIRECT_TRAIN_CORRIDORS or has_long_enough_road:
                departures = _train_departures(
                    km_for_train,
                    _corridor_strength(origin["name"], destination["name"]),
                )
            else:
                departures = []

            schedules[key] = {
                "departures": departures,
                "car_hours": car_hours,
                "origin_timezone_offset": city_lookup[origin["name"]]["timezone_offset"],
                "destination_timezone_offset": city_lookup[destination["name"]]["timezone_offset"],
                "source_mode": "synthetic",
            }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(schedules, handle, ensure_ascii=False, indent=2)

    LOGGER.info("Horarios ferroviarios guardados en %s", output_path)
    return schedules
