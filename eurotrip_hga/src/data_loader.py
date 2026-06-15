"""Carga, validación y generación perezosa de los datos del proyecto."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def project_root() -> Path:
    """Devuelve la raíz del proyecto."""
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    """Devuelve la carpeta de datos."""
    return project_root() / "data"


def output_dir() -> Path:
    """Devuelve la carpeta de salidas."""
    return project_root() / "outputs"


def cities_path() -> Path:
    """Devuelve la ruta del archivo GeoJSON de ciudades."""
    return data_dir() / "cities.geojson"


def distances_path() -> Path:
    """Devuelve la ruta del archivo de distancias."""
    return data_dir() / "distances.json"


def train_schedules_path() -> Path:
    """Devuelve la ruta del archivo de horarios de trenes."""
    return data_dir() / "train_schedules.json"


def load_cities() -> list[dict[str, Any]]:
    """Carga las ciudades desde el GeoJSON y conserva el orden del archivo."""
    with cities_path().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    cities: list[dict[str, Any]] = []
    for index, feature in enumerate(payload["features"]):
        properties = dict(feature["properties"])
        coordinates = feature["geometry"]["coordinates"]
        properties["lon"] = float(properties.get("lon", coordinates[0]))
        properties["lat"] = float(properties.get("lat", coordinates[1]))
        properties["index"] = index
        cities.append(properties)
    return cities


def load_distances() -> dict[str, Any]:
    """Carga la matriz de distancias OSRM, regenerándola si no existe o si es Haversine.

    La presencia del campo ``source`` con valor que empiece por 'OSRM' distingue la
    versión vial de la versión Haversine antigua.  Cualquier archivo sin ese campo se
    trata como obsoleto y se vuelve a generar con OSRM.
    """
    from .distance_matrix import build_distance_matrix_from_cities

    path = distances_path()
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if data.get("source", "").startswith("OSRM"):
                return data
        except Exception:
            pass

    return build_distance_matrix_from_cities(load_cities(), path)


def load_train_schedules() -> dict[str, Any]:
    """Carga los horarios de tren, generándolos si aún no existen."""
    path = train_schedules_path()
    if not path.exists():
        from .gtfs_fetcher import build_train_schedules

        build_train_schedules(load_cities(), load_distances(), path)

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_dataset() -> dict[str, Any]:
    """Asegura que todos los archivos de datos existan y los devuelve cargados."""
    data_dir().mkdir(parents=True, exist_ok=True)
    output_dir().mkdir(parents=True, exist_ok=True)

    cities = load_cities()
    distances = load_distances()
    schedules = load_train_schedules()
    return {
        "cities": cities,
        "distances": distances,
        "train_schedules": schedules,
    }
