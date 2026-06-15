"""Construcción de la matriz de distancias reales por carretera usando la API pública de OSRM."""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)

OSRM_BASE = "http://router.project-osrm.org/route/v1/driving"
REQUEST_DELAY_S = 0.5
REQUEST_TIMEOUT_S = 15


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula la distancia geodésica entre dos puntos como respaldo si OSRM falla."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    return 2.0 * R * math.asin(math.sqrt(a))


def osrm_road_route(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
) -> tuple[float | None, float | None, list]:
    """Consulta la API pública de OSRM para obtener distancia, duración y geometría vial.

    Retorna (km, horas, geometry) donde geometry es una lista de [lon, lat] que representa
    el trazado real por carretera.  Devuelve (None, None, []) si OSRM reporta código NoRoute
    (el par cruza un cuerpo de agua sin puente) o si hay error de red.
    """
    url = (
        f"{OSRM_BASE}/{lon1},{lat1};{lon2},{lat2}"
        "?overview=simplified&geometries=geojson"
    )
    try:
        resp = requests.get(
            url,
            timeout=REQUEST_TIMEOUT_S,
            headers={"User-Agent": "eurotrip-hga/1.0"},
        )
        data = resp.json()
        if data.get("code") == "Ok" and data.get("routes"):
            route = data["routes"][0]
            km = round(route["distance"] / 1000.0, 2)
            hours = round(route["duration"] / 3600.0, 3)
            geometry = route.get("geometry", {}).get("coordinates", [])
            return km, hours, geometry
        LOGGER.warning(
            "OSRM devolvió código '%s' para (%.4f,%.4f)→(%.4f,%.4f)",
            data.get("code", "?"),
            lon1, lat1, lon2, lat2,
        )
        return None, None, []
    except Exception as exc:
        LOGGER.error("Error de red al consultar OSRM: %s", exc)
        return None, None, []


def build_distance_matrix_from_cities(
    cities: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    """Construye y guarda la matriz de distancias reales por carretera consultando OSRM.

    Para cada par de ciudades almacena:
      - km: distancia vial real (OSRM).
      - km_straight: distancia en línea recta (Haversine), usada para estimar tiempos de tren.
      - car_hours: tiempo en auto por carretera real (OSRM).  null si no hay ruta vial.
      - route_geometry: polilínea simplificada [[lon, lat], …] del trayecto por carretera,
        apta para dibujar en el mapa sin cruzar cuerpos de agua.

    Reutiliza pares ya almacenados en caché (source: OSRM) para no repetir consultas.
    Si OSRM falla para un par, usa Haversine como respaldo y geometry vacía.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Intentar cargar caché OSRM previa
    existing_pairs: dict[str, Any] = {}
    if output_path.exists():
        try:
            with output_path.open("r", encoding="utf-8") as fh:
                cached = json.load(fh)
            if cached.get("source", "").startswith("OSRM"):
                existing_pairs = cached.get("pairs", {})
                LOGGER.info("Caché OSRM cargada: %d pares ya calculados.", len(existing_pairs))
        except Exception:
            pass

    matrix: dict[str, Any] = {
        "source": "OSRM (router.project-osrm.org)",
        "pairs": dict(existing_pairs),
    }
    computed = 0
    total_needed = sum(
        1
        for o in cities
        for d in cities
        if o["name"] != d["name"] and f"{o['name']}-{d['name']}" not in existing_pairs
    )
    LOGGER.info("%d pares nuevos por consultar a OSRM (~%.0f s).", total_needed, total_needed * REQUEST_DELAY_S)

    for origin in cities:
        for destination in cities:
            if origin["name"] == destination["name"]:
                continue
            key = f"{origin['name']}-{destination['name']}"
            if key in existing_pairs:
                continue

            km_straight = round(
                haversine_km(origin["lat"], origin["lon"], destination["lat"], destination["lon"]),
                2,
            )
            LOGGER.info(
                "OSRM [%d/%d]: %s → %s",
                computed + 1, total_needed, origin["name"], destination["name"],
            )
            km_road, car_hours, geometry = osrm_road_route(
                float(origin["lon"]), float(origin["lat"]),
                float(destination["lon"]), float(destination["lat"]),
            )

            if km_road is None:
                # Sin ruta vial: usar Haversine como respaldo; car_hours queda null
                # en el JSON para que fitness.py lo trate como imposible en auto.
                LOGGER.warning(
                    "Sin ruta OSRM para %s → %s; se almacena car_hours=null.",
                    origin["name"], destination["name"],
                )
                matrix["pairs"][key] = {
                    "from": origin["name"],
                    "to": destination["name"],
                    "km": None,
                    "km_straight": km_straight,
                    "car_hours": None,
                    "route_geometry": [],
                }
            else:
                matrix["pairs"][key] = {
                    "from": origin["name"],
                    "to": destination["name"],
                    "km": km_road,
                    "km_straight": km_straight,
                    "car_hours": car_hours,
                    "route_geometry": geometry,
                }

            computed += 1
            time.sleep(REQUEST_DELAY_S)

            # Guardar parcialmente cada 20 pares para no perder trabajo si se interrumpe
            if computed % 20 == 0:
                with output_path.open("w", encoding="utf-8") as fh:
                    json.dump(matrix, fh, ensure_ascii=False)

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(matrix, fh, ensure_ascii=False, indent=2)

    LOGGER.info("Matriz de distancias OSRM completada: %d pares nuevos.", computed)
    return matrix
