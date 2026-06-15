"""Evaluación del itinerario, penalizaciones y reconstrucción del estado temporal."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Penalización por arco donde no existe carretera ni tren disponible
W2_IMPOSSIBLE_ARC = 10_000.0
# Tiempo ficticio (horas) asignado internamente cuando no hay carretera,
# para que cualquier tren disponible siempre sea preferido
_CAR_IMPOSSIBLE_HOURS = 9_999.0


@dataclass
class RouteEvaluation:
    """Resultado detallado de evaluar una ruta candidata."""

    fitness: float
    objective_hours: float
    penalty: float
    full_route: list[int]
    arrival_times: list[float]
    departure_times: list[float]
    modals: list[str]
    itinerary: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convierte el resultado a un diccionario serializable."""
        return {
            "fitness": self.fitness,
            "objective_hours": self.objective_hours,
            "penalty": self.penalty,
            "full_route": self.full_route,
            "arrival_times": self.arrival_times,
            "departure_times": self.departure_times,
            "modals": self.modals,
            "itinerary": self.itinerary,
        }


def _local_hour(absolute_hour: float, timezone_offset: float) -> float:
    """Convierte una hora absoluta a hora local decimal en el intervalo [0, 24)."""
    return (absolute_hour + timezone_offset) % 24.0


def _best_train_departure(
    ready_hour: float,
    origin_offset: float,
    departures: list[dict[str, Any]],
) -> tuple[float, dict[str, Any]] | None:
    """Encuentra la primera salida ferroviaria viable dado el horario de trenes diario.

    Recibe ready_hour como la hora absoluta en que el viajero puede salir de la ciudad
    (arribo + estancia mínima) y busca el primer tren local que salga con al menos 0.5 h
    de margen para llegar a la estación.  Si ningún tren sale hoy, usa el primero del día
    siguiente.
    """
    if not departures:
        return None

    ordered = sorted(departures, key=lambda d: float(d["depart"]))
    local_ready = _local_hour(ready_hour, origin_offset)
    earliest_local = local_ready + 0.5  # margen de 30 min para llegar a la estación

    for dep in ordered:
        depart_local = float(dep["depart"])
        if depart_local >= earliest_local:
            depart_abs = ready_hour + (depart_local - local_ready)
            return depart_abs, dep

    # No hay tren hoy: usar el primero del día siguiente
    first = ordered[0]
    depart_abs = ready_hour + (24.0 - local_ready) + float(first["depart"])
    return depart_abs, first


def _resolve_car_hours(
    schedule: dict[str, Any],
    distances: dict[str, Any],
    pair_key: str,
) -> float | None:
    """Obtiene el tiempo en auto para un par de ciudades.

    Devuelve None si no existe ruta vial (p. ej. el par requiere cruzar el mar).
    El orden de precedencia es: horario > distances.json.
    """
    # Primero: valor explícito en train_schedules
    sched_val = schedule.get("car_hours")
    if sched_val is not None:
        return float(sched_val)

    # Segundo: valor en distances.json
    pair_data = distances.get("pairs", {}).get(pair_key, {})
    dist_val = pair_data.get("car_hours")
    if dist_val is not None:
        return float(dist_val)

    return None  # Sin ruta vial conocida


def evaluate_route(
    route: list[int],
    cities: list[dict[str, Any]],
    distances: dict[str, Any],
    train_schedules: dict[str, Any],
    *,
    start_city: int = 0,
    start_hour: float = 9.0,
    allow_trains: bool = True,
    apply_windows: bool = True,
    W1: float = 100.0,
) -> RouteEvaluation:
    """Evalúa una permutación cerrada bajo tren o auto con penalizaciones por ventana.

    Cuando car_hours es None para un arco (sin ruta vial), el tren es obligatorio.
    Si tampoco hay tren disponible, se aplica una penalización W2 = 10 000 h.

    La variable ready_hour representa el momento en que el viajero puede salir de la
    ciudad actual, incorporando la estancia mínima (min_stay_hours) después de cada
    arribo.  Para el tramo de regreso al origen no se suma estancia.
    """
    full_route = [start_city, *route, start_city]
    n_legs = len(full_route) - 1

    ready_hour = start_hour
    arrival_times: list[float] = [start_hour]
    departure_times: list[float] = [start_hour]
    modals: list[str] = []
    itinerary: list[dict[str, Any]] = []
    penalty = 0.0

    for leg_index, destination_city in enumerate(full_route[1:], start=1):
        origin_city = full_route[leg_index - 1]
        origin = cities[origin_city]
        destination = cities[destination_city]
        pair_key = f"{origin['name']}-{destination['name']}"
        schedule = train_schedules.get(pair_key, {})

        car_hours = _resolve_car_hours(schedule, distances, pair_key)
        no_road = car_hours is None  # True cuando el par cruza mar u obstáculo infranqueable

        # --- Opción auto ---
        if no_road:
            # Sin carretera: forzar tren; si tampoco hay tren se penalizará abajo
            chosen_modal = "car_impossible"
            chosen_departure = ready_hour
            chosen_arrival = ready_hour + _CAR_IMPOSSIBLE_HOURS
        else:
            chosen_modal = "car"
            chosen_departure = ready_hour
            chosen_arrival = ready_hour + car_hours

        chosen_departure_payload: dict[str, Any] | None = None

        # --- Opción tren (si está habilitado y existe horario) ---
        if allow_trains:
            train_candidate = _best_train_departure(
                ready_hour,
                float(origin["timezone_offset"]),
                schedule.get("departures", []),
            )
            if train_candidate is not None:
                candidate_departure, payload = train_candidate
                candidate_arrival = candidate_departure + float(payload["duration_h"])
                if candidate_arrival <= chosen_arrival:
                    chosen_modal = "train"
                    chosen_departure = candidate_departure
                    chosen_arrival = candidate_arrival
                    chosen_departure_payload = payload

        # Penalizar arco imposible (sin carretera y sin tren disponible)
        if no_road and chosen_modal != "train":
            penalty += W2_IMPOSSIBLE_ARC

        # --- Aplicar restricciones de ventana horaria en el destino ---
        local_arrival = _local_hour(chosen_arrival, float(destination["timezone_offset"]))
        city_open = float(destination.get("open_hour", 0.0))
        city_close = float(destination.get("close_hour", 24.0))
        if apply_windows:
            if local_arrival < city_open:
                wait_hours = city_open - local_arrival
                chosen_arrival += wait_hours
                local_arrival = city_open
            if local_arrival > city_close:
                violation = local_arrival - city_close
                penalty += W1 * violation * violation

        arrival_times.append(chosen_arrival)

        # Calcular cuándo puede salir de la ciudad destino.
        # En el último tramo (regreso al origen) no se suma estancia.
        is_return_leg = leg_index == n_legs
        if is_return_leg:
            ready_hour = chosen_arrival
        else:
            min_stay = float(destination.get("min_stay_hours", 0.0))
            ready_hour = chosen_arrival + min_stay

        departure_times.append(ready_hour)

        display_modal = chosen_modal if chosen_modal != "car_impossible" else "car"
        modals.append(display_modal)

        itinerary.append(
            {
                "origin": origin["name"],
                "destination": destination["name"],
                "origin_index": origin_city,
                "destination_index": destination_city,
                "departure_abs": round(chosen_departure, 2),
                "arrival_abs": round(chosen_arrival, 2),
                "arrival_local": round(local_arrival, 2),
                "departure_next_abs": round(ready_hour, 2),
                "modal": display_modal,
                "no_road": no_road,
                "duration_h": round(chosen_arrival - chosen_departure, 2),
                "train_departure": chosen_departure_payload["depart"] if chosen_departure_payload else None,
                "train_arrival": chosen_departure_payload["arrive"] if chosen_departure_payload else None,
            }
        )

    objective_hours = arrival_times[-1] - start_hour
    fitness = objective_hours + penalty
    return RouteEvaluation(
        fitness=fitness,
        objective_hours=objective_hours,
        penalty=penalty,
        full_route=full_route,
        arrival_times=arrival_times,
        departure_times=departure_times,
        modals=modals,
        itinerary=itinerary,
    )
