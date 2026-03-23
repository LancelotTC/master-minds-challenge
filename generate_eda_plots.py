from __future__ import annotations

import argparse
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from utils import SqliteManager


TABLE_NAME = "mouvements_aero_insa"
NULL_LIKE = {"", "NULL", "NONE", "NAN"}
SELECTED_PROFILE_COLUMNS = [
    "LTScheduledDatetime",
    "Direction",
    "IdTraficType",
    "Terminal",
    "airlineOACICode",
    "IdAircraftType",
    "NbPaxTotal",
    "DelayMainReasonDuration",
    "TaxiDurationMinutes",
    "TurnsBlockTimeMinutes",
]


@dataclass
class PlotArtifact:
    name: str
    file_path: Path
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate exploratory plots for the mouvements_aero_insa table."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("database.db"),
        help="Path to the SQLite database.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eda_outputs"),
        help="Directory where plots will be written.",
    )
    parser.add_argument(
        "--table",
        default=TABLE_NAME,
        help="Source table name.",
    )
    return parser.parse_args()


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().upper() in NULL_LIKE
    return False


def clean_text(value: object) -> str | None:
    if is_missing(value):
        return None
    return str(value).strip()


def to_float(value: object) -> float | None:
    if is_missing(value):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
    else:
        text = str(value).strip().replace(",", ".")
        try:
            numeric = float(text)
        except ValueError:
            return None
    if math.isfinite(numeric):
        return numeric
    return None


def parse_datetime(value: object) -> datetime | None:
    if is_missing(value):
        return None
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def fetch_all(db_path: Path, query: str, params: Iterable[object] = ()) -> list[tuple]:
    with SqliteManager(db_path, commit_on_exit=False) as (cursor, _connection):
        cursor.execute(query, tuple(params))
        return cursor.fetchall()


def fetch_one(db_path: Path, query: str, params: Iterable[object] = ()) -> tuple:
    with SqliteManager(db_path, commit_on_exit=False) as (cursor, _connection):
        cursor.execute(query, tuple(params))
        row = cursor.fetchone()
    if row is None:
        raise ValueError(f"Query returned no rows: {query}")
    return row


def fetch_column_values(db_path: Path, table: str, column: str) -> list[object]:
    query = f"SELECT {quote_identifier(column)} FROM {quote_identifier(table)}"
    rows = fetch_all(db_path, query)
    return [row[0] for row in rows]


def ensure_metadata_tables(db_path: Path) -> None:
    with SqliteManager(db_path) as (cursor, _connection):
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS eda_run_log (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                executed_at TEXT NOT NULL,
                output_dir TEXT NOT NULL,
                source_table TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                plots_generated INTEGER NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS eda_plot_artifact (
                run_id INTEGER NOT NULL,
                plot_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                description TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES eda_run_log(run_id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS eda_column_profile (
                run_id INTEGER NOT NULL,
                column_name TEXT NOT NULL,
                non_missing_count INTEGER NOT NULL,
                missing_count INTEGER NOT NULL,
                distinct_count INTEGER NOT NULL,
                numeric_min REAL,
                numeric_max REAL,
                numeric_avg REAL,
                FOREIGN KEY(run_id) REFERENCES eda_run_log(run_id)
            )
            """
        )


def persist_metadata(
    db_path: Path,
    table: str,
    output_dir: Path,
    row_count: int,
    artifacts: list[PlotArtifact],
    column_profiles: list[dict[str, object]],
) -> int:
    executed_at = datetime.now().isoformat(timespec="seconds")
    with SqliteManager(db_path) as (cursor, _connection):
        cursor.execute(
            """
            INSERT INTO eda_run_log (
                executed_at,
                output_dir,
                source_table,
                row_count,
                plots_generated
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (executed_at, str(output_dir.resolve()), table, row_count, len(artifacts)),
        )
        run_id = int(cursor.lastrowid)
        cursor.executemany(
            """
            INSERT INTO eda_plot_artifact (
                run_id,
                plot_name,
                file_path,
                description
            ) VALUES (?, ?, ?, ?)
            """,
            [
                (run_id, artifact.name, str(artifact.file_path.resolve()), artifact.description)
                for artifact in artifacts
            ],
        )
        cursor.executemany(
            """
            INSERT INTO eda_column_profile (
                run_id,
                column_name,
                non_missing_count,
                missing_count,
                distinct_count,
                numeric_min,
                numeric_max,
                numeric_avg
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    profile["column_name"],
                    profile["non_missing_count"],
                    profile["missing_count"],
                    profile["distinct_count"],
                    profile["numeric_min"],
                    profile["numeric_max"],
                    profile["numeric_avg"],
                )
                for profile in column_profiles
            ],
        )
        return run_id


def build_column_profiles(db_path: Path, table: str, columns: list[str]) -> list[dict[str, object]]:
    profiles: list[dict[str, object]] = []
    for column in columns:
        values = fetch_column_values(db_path, table, column)
        non_missing_values = [value for value in values if not is_missing(value)]
        numeric_values = [value for value in (to_float(raw) for raw in non_missing_values) if value is not None]
        cleaned_text_values = [str(value).strip() for value in non_missing_values]
        profiles.append(
            {
                "column_name": column,
                "non_missing_count": len(non_missing_values),
                "missing_count": len(values) - len(non_missing_values),
                "distinct_count": len(set(cleaned_text_values)),
                "numeric_min": min(numeric_values) if numeric_values else None,
                "numeric_max": max(numeric_values) if numeric_values else None,
                "numeric_avg": float(np.mean(numeric_values)) if numeric_values else None,
            }
        )
    return profiles


def save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / filename
    fig.tight_layout()
    fig.savefig(file_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return file_path


def plot_monthly_volume(db_path: Path, table: str, output_dir: Path) -> PlotArtifact:
    query = f"""
        SELECT substr(LTScheduledDatetime, 1, 7) AS month, Direction, COUNT(*) AS movement_count
        FROM {quote_identifier(table)}
        WHERE LTScheduledDatetime IS NOT NULL
          AND TRIM(CAST(LTScheduledDatetime AS TEXT)) NOT IN ('', 'NULL')
        GROUP BY month, Direction
        HAVING month IS NOT NULL AND month <> 'NULL'
        ORDER BY month
    """
    rows = fetch_all(db_path, query)

    month_map: dict[str, dict[str, int]] = defaultdict(dict)
    directions: set[str] = set()
    for month, direction, count in rows:
        month_map[str(month)][clean_text(direction) or "Inconnu"] = int(count)
        directions.add(clean_text(direction) or "Inconnu")

    months = sorted(month_map)
    fig, ax = plt.subplots(figsize=(12, 5))
    for direction in sorted(directions):
        ax.plot(
            months,
            [month_map[month].get(direction, 0) for month in months],
            marker="o",
            linewidth=2,
            label=direction,
        )
    ax.set_title("Traffic volume by scheduled month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of movements")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.25)
    ax.legend()
    path = save_figure(fig, output_dir, "01_monthly_volume.png")
    return PlotArtifact(
        name="monthly_volume",
        file_path=path,
        description="Monthly movement counts split by arrival and departure.",
    )


def plot_hourly_heatmaps(db_path: Path, table: str, output_dir: Path) -> PlotArtifact:
    query = f"""
        SELECT Direction,
               CAST(strftime('%w', LTScheduledDatetime) AS INTEGER) AS weekday_index,
               CAST(strftime('%H', LTScheduledDatetime) AS INTEGER) AS hour_index,
               COUNT(*) AS movement_count
        FROM {quote_identifier(table)}
        WHERE LTScheduledDatetime IS NOT NULL
          AND TRIM(CAST(LTScheduledDatetime AS TEXT)) NOT IN ('', 'NULL')
        GROUP BY Direction, weekday_index, hour_index
        ORDER BY Direction, weekday_index, hour_index
    """
    rows = fetch_all(db_path, query)
    day_labels = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    directions = sorted({clean_text(row[0]) or "Inconnu" for row in rows})

    fig, axes = plt.subplots(
        nrows=max(1, len(directions)),
        ncols=1,
        figsize=(14, 4 * max(1, len(directions))),
        squeeze=False,
    )

    for axis, direction in zip(axes.flat, directions):
        matrix = np.zeros((7, 24), dtype=float)
        for raw_direction, weekday_index, hour_index, movement_count in rows:
            current_direction = clean_text(raw_direction) or "Inconnu"
            if current_direction != direction:
                continue
            if weekday_index is None or hour_index is None:
                continue
            matrix[int(weekday_index), int(hour_index)] = movement_count

        image = axis.imshow(matrix, cmap="YlOrRd", aspect="auto")
        axis.set_title(f"Scheduled activity heatmap: {direction}")
        axis.set_xlabel("Scheduled hour")
        axis.set_ylabel("Weekday")
        axis.set_xticks(range(24))
        axis.set_yticks(range(7))
        axis.set_yticklabels(day_labels)
        fig.colorbar(image, ax=axis, label="Movements")

    path = save_figure(fig, output_dir, "02_hourly_heatmaps.png")
    return PlotArtifact(
        name="hourly_heatmaps",
        file_path=path,
        description="Weekday-by-hour heatmaps for scheduled movements by direction.",
    )


def plot_top_airlines(db_path: Path, table: str, output_dir: Path, limit: int = 15) -> PlotArtifact:
    query = f"""
        SELECT airlineOACICode, COUNT(*) AS movement_count
        FROM {quote_identifier(table)}
        WHERE airlineOACICode IS NOT NULL
          AND TRIM(CAST(airlineOACICode AS TEXT)) NOT IN ('', 'NULL')
        GROUP BY airlineOACICode
        ORDER BY movement_count DESC
        LIMIT ?
    """
    rows = fetch_all(db_path, query, (limit,))
    labels = [clean_text(code) or "Inconnu" for code, _count in rows][::-1]
    counts = [int(count) for _code, count in rows][::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(labels, counts, color="#33658A")
    ax.set_title(f"Top {limit} airlines by movement count")
    ax.set_xlabel("Movements")
    ax.set_ylabel("Airline OACI code")
    ax.grid(axis="x", alpha=0.2)
    path = save_figure(fig, output_dir, "03_top_airlines.png")
    return PlotArtifact(
        name="top_airlines",
        file_path=path,
        description="Horizontal bar chart of the busiest airline OACI codes.",
    )


def plot_terminal_traffic_mix(db_path: Path, table: str, output_dir: Path) -> PlotArtifact:
    query = f"""
        SELECT Terminal, IdTraficType, COUNT(*) AS movement_count
        FROM {quote_identifier(table)}
        WHERE Terminal IS NOT NULL
          AND TRIM(CAST(Terminal AS TEXT)) NOT IN ('', 'NULL')
        GROUP BY Terminal, IdTraficType
        ORDER BY Terminal, IdTraficType
    """
    rows = fetch_all(db_path, query)
    terminal_totals: dict[str, int] = defaultdict(int)
    traffic_by_terminal: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    label_map = {"0": "Cancelled", "1": "Realized", "2": "Scheduled"}

    for terminal, traffic_type, movement_count in rows:
        terminal_name = clean_text(terminal) or "Inconnu"
        traffic_name = label_map.get(str(traffic_type), str(traffic_type))
        traffic_by_terminal[terminal_name][traffic_name] += int(movement_count)
        terminal_totals[terminal_name] += int(movement_count)

    terminals = [name for name, _count in sorted(terminal_totals.items(), key=lambda item: item[1], reverse=True)[:8]]
    traffic_labels = sorted({traffic for values in traffic_by_terminal.values() for traffic in values})
    cumulative = np.zeros(len(terminals), dtype=float)

    fig, ax = plt.subplots(figsize=(11, 6))
    for traffic_label in traffic_labels:
        values = np.array([traffic_by_terminal[terminal].get(traffic_label, 0) for terminal in terminals], dtype=float)
        ax.bar(terminals, values, bottom=cumulative, label=traffic_label)
        cumulative += values

    ax.set_title("Traffic type mix across the busiest terminals")
    ax.set_xlabel("Terminal")
    ax.set_ylabel("Movements")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    path = save_figure(fig, output_dir, "04_terminal_traffic_mix.png")
    return PlotArtifact(
        name="terminal_traffic_mix",
        file_path=path,
        description="Stacked bar chart of traffic type composition by terminal.",
    )


def clip_percentiles(values: list[float], lower_pct: float = 1.0, upper_pct: float = 99.0) -> np.ndarray:
    array = np.array(values, dtype=float)
    if array.size == 0:
        return array
    lower_bound, upper_bound = np.percentile(array, [lower_pct, upper_pct])
    return array[(array >= lower_bound) & (array <= upper_bound)]


def plot_passenger_distribution(db_path: Path, table: str, output_dir: Path) -> PlotArtifact:
    query = f"""
        SELECT Direction, NbPaxTotal
        FROM {quote_identifier(table)}
        WHERE NbPaxTotal IS NOT NULL
    """
    rows = fetch_all(db_path, query)
    by_direction: dict[str, list[float]] = defaultdict(list)
    for direction, raw_value in rows:
        numeric = to_float(raw_value)
        if numeric is None or numeric < 0:
            continue
        by_direction[clean_text(direction) or "Inconnu"].append(numeric)

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#55A630", "#FF9F1C", "#6C757D"]
    for index, direction in enumerate(sorted(by_direction)):
        clipped_values = clip_percentiles(by_direction[direction], 1, 99)
        if clipped_values.size == 0:
            continue
        ax.hist(
            clipped_values,
            bins=40,
            alpha=0.55,
            label=f"{direction} (median={median(clipped_values):.0f})",
            color=colors[index % len(colors)],
        )

    ax.set_title("Passenger-count distribution after trimming extreme outliers")
    ax.set_xlabel("NbPaxTotal")
    ax.set_ylabel("Flights")
    ax.grid(alpha=0.2)
    ax.legend()
    path = save_figure(fig, output_dir, "05_passenger_distribution.png")
    return PlotArtifact(
        name="passenger_distribution",
        file_path=path,
        description="Passenger histograms by direction with 1st-99th percentile clipping.",
    )


def plot_delay_and_taxi_distribution(db_path: Path, table: str, output_dir: Path) -> PlotArtifact:
    query = f"""
        SELECT DelayMainReasonDuration, TaxiDurationMinutes, Direction
        FROM {quote_identifier(table)}
        WHERE DelayMainReasonDuration IS NOT NULL
           OR TaxiDurationMinutes IS NOT NULL
    """
    rows = fetch_all(db_path, query)
    delay_values: list[float] = []
    taxi_by_direction: dict[str, list[float]] = defaultdict(list)
    for raw_delay, raw_taxi, raw_direction in rows:
        delay_value = to_float(raw_delay)
        if delay_value is not None and delay_value >= 0:
            delay_values.append(delay_value)

        taxi_value = to_float(raw_taxi)
        if taxi_value is not None and -60 <= taxi_value <= 120:
            taxi_by_direction[clean_text(raw_direction) or "Inconnu"].append(taxi_value)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

    clipped_delay_values = clip_percentiles(delay_values, 0, 99.5)
    axes[0].hist(clipped_delay_values, bins=50, color="#D1495B", alpha=0.85)
    axes[0].set_title("Delay distribution")
    axes[0].set_xlabel("DelayMainReasonDuration (minutes)")
    axes[0].set_ylabel("Flights")
    axes[0].grid(alpha=0.2)

    ordered_directions = sorted(taxi_by_direction)
    taxi_samples = [clip_percentiles(taxi_by_direction[direction], 1, 99) for direction in ordered_directions]
    axes[1].boxplot(taxi_samples, tick_labels=ordered_directions, showfliers=False)
    axes[1].set_title("Taxi-time spread by direction")
    axes[1].set_xlabel("Direction")
    axes[1].set_ylabel("TaxiDurationMinutes")
    axes[1].grid(alpha=0.2)

    path = save_figure(fig, output_dir, "06_delay_and_taxi_distribution.png")
    return PlotArtifact(
        name="delay_and_taxi_distribution",
        file_path=path,
        description="Delay histogram and taxi-time boxplot by direction.",
    )


def plot_aircraft_vs_passengers(db_path: Path, table: str, output_dir: Path, limit: int = 15) -> PlotArtifact:
    query = f"""
        SELECT IdAircraftType, NbPaxTotal
        FROM {quote_identifier(table)}
        WHERE IdAircraftType IS NOT NULL
          AND TRIM(CAST(IdAircraftType AS TEXT)) NOT IN ('', 'NULL')
          AND NbPaxTotal IS NOT NULL
    """
    rows = fetch_all(db_path, query)
    counts: dict[str, int] = defaultdict(int)
    passenger_totals: dict[str, float] = defaultdict(float)

    for raw_aircraft, raw_pax in rows:
        aircraft = clean_text(raw_aircraft)
        pax = to_float(raw_pax)
        if aircraft is None or pax is None or pax < 0:
            continue
        counts[aircraft] += 1
        passenger_totals[aircraft] += pax

    busiest_aircraft = sorted(counts, key=counts.get, reverse=True)[:limit]
    avg_passengers = [passenger_totals[aircraft] / counts[aircraft] for aircraft in busiest_aircraft]
    movement_counts = [counts[aircraft] for aircraft in busiest_aircraft]

    fig, ax = plt.subplots(figsize=(11, 6))
    scatter = ax.scatter(
        avg_passengers,
        movement_counts,
        s=np.array(movement_counts) / 10,
        c=np.array(avg_passengers),
        cmap="viridis",
        alpha=0.8,
        edgecolors="black",
        linewidths=0.5,
    )
    for aircraft, avg_pax, movement_count in zip(busiest_aircraft, avg_passengers, movement_counts):
        ax.annotate(aircraft, (avg_pax, movement_count), fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.set_title("Aircraft types: average passengers vs movement volume")
    ax.set_xlabel("Average NbPaxTotal")
    ax.set_ylabel("Movement count")
    ax.grid(alpha=0.2)
    fig.colorbar(scatter, ax=ax, label="Average passengers")
    path = save_figure(fig, output_dir, "07_aircraft_vs_passengers.png")
    return PlotArtifact(
        name="aircraft_vs_passengers",
        file_path=path,
        description="Bubble chart for the busiest aircraft types and their average passenger load.",
    )


def plot_missingness_overview(
    db_path: Path,
    table: str,
    output_dir: Path,
    columns: list[str],
) -> PlotArtifact:
    missing_rates: list[float] = []
    labels: list[str] = []

    for column in columns:
        values = fetch_column_values(db_path, table, column)
        total_count = len(values)
        missing_count = sum(1 for value in values if is_missing(value))
        labels.append(column)
        missing_rates.append((missing_count / total_count) * 100 if total_count else 0.0)

    order = np.argsort(missing_rates)
    ordered_labels = [labels[index] for index in order]
    ordered_rates = [missing_rates[index] for index in order]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(ordered_labels, ordered_rates, color="#8D99AE")
    ax.set_title("Missing-value rate for key analysis columns")
    ax.set_xlabel("Missing values (%)")
    ax.set_ylabel("Column")
    ax.grid(axis="x", alpha=0.2)
    path = save_figure(fig, output_dir, "08_missingness_overview.png")
    return PlotArtifact(
        name="missingness_overview",
        file_path=path,
        description="Missing-rate comparison for the core columns used in the EDA plots.",
    )


def apply_plot_style() -> None:
    plt.style.use("ggplot")
    plt.rcParams.update(
        {
            "axes.facecolor": "#F8F9FA",
            "figure.facecolor": "white",
            "axes.titleweight": "bold",
            "axes.edgecolor": "#ADB5BD",
            "grid.color": "#DEE2E6",
            "font.size": 10,
        }
    )


def main() -> None:
    args = parse_args()
    db_path = args.db_path
    output_dir = args.output_dir
    table = args.table

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    apply_plot_style()

    row_count = int(
        fetch_one(
            db_path,
            f"SELECT COUNT(*) FROM {quote_identifier(table)}",
        )[0]
    )

    artifacts = [
        plot_monthly_volume(db_path, table, output_dir),
        plot_hourly_heatmaps(db_path, table, output_dir),
        plot_top_airlines(db_path, table, output_dir),
        plot_terminal_traffic_mix(db_path, table, output_dir),
        plot_passenger_distribution(db_path, table, output_dir),
        plot_delay_and_taxi_distribution(db_path, table, output_dir),
        plot_aircraft_vs_passengers(db_path, table, output_dir),
        plot_missingness_overview(db_path, table, output_dir, SELECTED_PROFILE_COLUMNS),
    ]

    ensure_metadata_tables(db_path)
    column_profiles = build_column_profiles(db_path, table, SELECTED_PROFILE_COLUMNS)
    run_id = persist_metadata(
        db_path=db_path,
        table=table,
        output_dir=output_dir,
        row_count=row_count,
        artifacts=artifacts,
        column_profiles=column_profiles,
    )

    print(f"Generated {len(artifacts)} plot(s) in {output_dir.resolve()}")
    print(f"Saved EDA metadata to database.db with run_id={run_id}")
    for artifact in artifacts:
        print(f"- {artifact.name}: {artifact.file_path.resolve()}")


if __name__ == "__main__":
    main()
