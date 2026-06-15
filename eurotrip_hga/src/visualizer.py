"""Visualizaciones del itinerario, convergencia y cronograma del viaje."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from branca.element import Element


def _city_lookup(cities: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Construye un índice por nombre de ciudad."""
    return {city["name"]: city for city in cities}


def create_itinerary_map(
    cities: list[dict[str, Any]],
    evaluation: Any,
    output_path: Path,
    *,
    title: str,
    distances: dict[str, Any] | None = None,
) -> Path:
    """Genera un mapa HTML interactivo con la ruta y los modos de viaje.

    Cuando se proporciona ``distances`` (matriz OSRM), los tramos en auto se dibujan
    siguiendo la geometría real de la carretera almacenada en route_geometry, evitando
    así que las líneas crucen visualmente cuerpos de agua.  Los tramos en tren se
    representan con líneas rectas (los trenes circulan por vías dedicadas).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    city_map = _city_lookup(cities)
    latitudes = [city["lat"] for city in cities]
    longitudes = [city["lon"] for city in cities]
    center = [sum(latitudes) / len(latitudes), sum(longitudes) / len(longitudes)]

    fmap = folium.Map(location=center, zoom_start=4, tiles="cartodbpositron")

    for step in evaluation.itinerary:
        city = city_map[step["destination"]]
        day_num = int(step["arrival_abs"] // 24) + 1
        hour_local = step["arrival_local"]
        popup = folium.Popup(
            html=(
                f"<b>{step['destination']}</b><br>"
                f"Llegada: día {day_num} a las {hour_local:.1f} h (local)<br>"
                f"Hora abs.: {step['arrival_abs']:.2f} h<br>"
                f"Modal: {step['modal']}<br>"
                f"Estancia mínima: {city.get('min_stay_hours', '?')} h<br>"
                f"Listo para salir: {step['departure_next_abs']:.2f} h"
            ),
            max_width=340,
        )
        folium.Marker(
            location=[city["lat"], city["lon"]],
            popup=popup,
            tooltip=f"{step['destination']} ({step['modal']})",
        ).add_to(fmap)

    pairs_data = distances.get("pairs", {}) if distances else {}
    full_route = evaluation.full_route

    for step, left, right in zip(evaluation.itinerary, full_route[:-1], full_route[1:]):
        origin_city = city_map[cities[left]["name"]]
        dest_city = city_map[cities[right]["name"]]
        is_train = step["modal"] == "train"
        color = "#1a56db" if is_train else "#e02424"

        if not is_train and distances is not None:
            pair_key = f"{cities[left]['name']}-{cities[right]['name']}"
            geometry = pairs_data.get(pair_key, {}).get("route_geometry", [])
            if geometry:
                # OSRM devuelve [lon, lat]; folium requiere [lat, lon]
                locations = [[coord[1], coord[0]] for coord in geometry]
            else:
                locations = [
                    [origin_city["lat"], origin_city["lon"]],
                    [dest_city["lat"], dest_city["lon"]],
                ]
        else:
            locations = [
                [origin_city["lat"], origin_city["lon"]],
                [dest_city["lat"], dest_city["lon"]],
            ]

        folium.PolyLine(
            locations=locations,
            color=color,
            weight=4 if is_train else 3,
            opacity=0.85,
            dash_array=None if is_train else "8 4",
            tooltip=f"{cities[left]['name']} → {cities[right]['name']} ({step['modal']})",
        ).add_to(fmap)

    legend_html = f"""
    <div style="position: fixed; bottom: 32px; left: 32px; z-index: 9999; background: white;
                padding: 12px 16px; border: 2px solid #444; border-radius: 10px;
                font-size: 13px; box-shadow: 0 4px 20px rgba(0,0,0,0.15); min-width: 200px;">
      <div style="font-weight: 700; margin-bottom: 8px; font-size: 14px;">{title}</div>
      <div style="margin-bottom: 4px;">
        <span style="display:inline-block;width:24px;height:4px;background:#1a56db;margin-right:8px;vertical-align:middle;"></span>Tren
      </div>
      <div style="margin-bottom: 8px;">
        <span style="display:inline-block;width:24px;height:4px;background:#e02424;margin-right:8px;vertical-align:middle;border-top: 4px dashed #e02424;"></span>Auto (ruta real OSRM)
      </div>
      <div style="border-top: 1px solid #ddd; padding-top: 6px;">
        Tiempo total: <b>{evaluation.objective_hours:.1f} h</b><br>
        Penalización: <b>{evaluation.penalty:.1f}</b>
      </div>
    </div>
    """
    fmap.get_root().html.add_child(Element(legend_html))
    fmap.save(str(output_path))
    return output_path


def plot_convergence(histories: list[list[float]], output_path: Path) -> Path:
    """Grafica cinco ejecuciones y la media de la convergencia."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not histories:
        raise ValueError("Se requieren historiales para graficar la convergencia.")

    max_length = max(len(history) for history in histories)
    aligned = []
    for history in histories:
        if len(history) < max_length:
            history = history + [history[-1]] * (max_length - len(history))
        aligned.append(history)

    mean_history = [sum(values) / len(values) for values in zip(*aligned)]
    plt.figure(figsize=(10, 5))
    for history in aligned:
        plt.plot(history, alpha=0.35, linewidth=1.4)
    plt.plot(mean_history, color="black", linewidth=2.8, label="Promedio")
    plt.xlabel("Generaciones")
    plt.ylabel("Mejor fitness")
    plt.title("Convergencia del HGA")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def plot_timeline(evaluation: Any, cities: list[dict[str, Any]], output_path: Path) -> Path:
    """Dibuja un cronograma tipo Gantt del itinerario."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    city_map = _city_lookup(cities)
    y_positions = list(range(len(evaluation.itinerary)))
    plt.figure(figsize=(12, max(6, len(evaluation.itinerary) * 0.35)))

    for idx, step in enumerate(evaluation.itinerary):
        city = city_map[step["destination"]]
        stay = float(city["min_stay_hours"])
        color = "#2b6cb0" if step["modal"] == "train" else "#c53030"
        plt.barh(
            y_positions[idx],
            stay,
            left=step["arrival_abs"],
            color=color,
            alpha=0.85,
        )
        plt.text(
            step["arrival_abs"] + stay / 2.0,
            y_positions[idx],
            step["destination"],
            va="center",
            ha="center",
            color="white",
            fontsize=9,
        )

    plt.yticks(y_positions, [step["destination"] for step in evaluation.itinerary])
    plt.xlabel("Horas absolutas del viaje")
    plt.title("Timeline del itinerario")
    plt.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path
