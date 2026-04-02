import os
from datetime import timedelta
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from google.cloud import bigquery
from google.cloud.bigquery.table import Row
from jours_feries_france import JoursFeries
from tqdm.auto import tqdm
from vacances_scolaires_france import SchoolHolidayDates

from pipeline_config import PIPELINE_CONFIG
from utils import (
    find_csv_file,
    get_season,
    haversine_km,
    normalize_airport_code_series,
    normalize_country_code_series,
    normalize_direction_value,
    pick_column,
)


CONFIG = PIPELINE_CONFIG
PATHS = CONFIG.dataset_paths
COLUMNS = CONFIG.columns
FEATURES = CONFIG.features
REFERENCE = CONFIG.reference
CSV_ENCODING = "utf-8-sig"


class DatasetFiles:
    @staticmethod
    def ensure_directories() -> None:
        PATHS.data_dir.mkdir(exist_ok=True)
        PATHS.weather_dir.mkdir(exist_ok=True)
        PATHS.holidays_dir.mkdir(exist_ok=True)
        PATHS.world_events_dir.mkdir(exist_ok=True)

    @classmethod
    def has_original_dataset(cls) -> bool:
        return PATHS.original_dataset.exists()

    @classmethod
    def load_original_dataset(cls) -> pd.DataFrame:
        if not cls.has_original_dataset():
            raise FileNotFoundError(f"Original dataset not found at {PATHS.original_dataset}")
        return pd.read_csv(PATHS.original_dataset, low_memory=False)

    @classmethod
    def load_main_dataset(cls) -> pd.DataFrame:
        if not PATHS.main_dataset.exists():
            raise FileNotFoundError(f"Main dataset not found at {PATHS.main_dataset}")
        return pd.read_csv(PATHS.main_dataset, low_memory=False)

    @classmethod
    def save_original_dataset(cls, dataframe: pd.DataFrame) -> Path:
        cls.ensure_directories()
        dataframe.to_csv(PATHS.original_dataset, index=False, encoding=CSV_ENCODING)
        return PATHS.original_dataset

    @classmethod
    def save_main_dataset(cls, dataframe: pd.DataFrame) -> Path:
        cls.ensure_directories()
        dataframe.to_csv(PATHS.main_dataset, index=False, encoding=CSV_ENCODING)
        return PATHS.main_dataset


class BigQueryDatasetDownloader:
    @staticmethod
    def _require_env(name: str) -> str:
        value = os.getenv(name)
        if not value:
            raise RuntimeError(f"Missing required environment variable: {name}")
        return value

    @classmethod
    def ensure_original_dataset(cls) -> Path:
        DatasetFiles.ensure_directories()
        if DatasetFiles.has_original_dataset():
            return PATHS.original_dataset

        project_id = cls._require_env("PROJECT_ID")
        dataset_id = cls._require_env("DATASET_ID")
        table_id = cls._require_env("TABLE_ID")

        service_account_key_path = os.getenv("SERVICE_ACCOUNT_KEY_PATH") or os.getenv("YOUR_SERVICE_ACCOUNT_KEY_PATH")
        if service_account_key_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_key_path

        table_ref = f"`{project_id}.{dataset_id}.{table_id}`"
        query = f"SELECT * FROM {table_ref};"
        client = bigquery.Client(project=project_id)
        query_job = client.query(query)
        rows: Iterator[Row] = query_job.result()
        total_rows = getattr(rows, "total_rows", 0) or 0

        dataset_rows: list[dict[str, object]] = []
        for row in tqdm(rows, total=total_rows or None, desc="Downloading dataset", unit="row"):
            dataset_rows.append(dict(row.items()))

        dataset = pd.DataFrame(dataset_rows)

        return DatasetFiles.save_original_dataset(dataset)


class AirportReferenceService:
    @staticmethod
    def load_country_mapping() -> pd.DataFrame:
        if not PATHS.world_events_airports.exists():
            return pd.DataFrame(columns=["airport_code", "country_code"])

        return (
            pd.read_parquet(PATHS.world_events_airports, columns=["iata_code", "iso_country"])
            .rename(columns={"iata_code": "airport_code", "iso_country": "country_code"})[
                ["airport_code", "country_code"]
            ]
            .assign(
                airport_code=lambda dataframe: normalize_airport_code_series(dataframe["airport_code"]),
                country_code=lambda dataframe: normalize_country_code_series(dataframe["country_code"]),
            )
            .dropna(subset=["airport_code", "country_code"])
            .drop_duplicates()
        )

    @staticmethod
    def load_airports_with_coordinates() -> pd.DataFrame:
        airports_path = find_csv_file(PATHS.holidays_dir, ["upply-airports.csv", "airports.csv"])
        if airports_path is None:
            return pd.DataFrame(columns=["airport_code", "latitude", "longitude"])

        airports_raw = pd.read_csv(airports_path, sep=None, engine="python")
        code_column = pick_column(
            airports_raw.columns,
            ["code", "airportprevious", "airport_code", "airportcode", "iata", "iata_code"],
        )
        latitude_column = pick_column(airports_raw.columns, ["latitude", "lat"])
        longitude_column = pick_column(airports_raw.columns, ["longitude", "lon", "lng"])
        if code_column is None or latitude_column is None or longitude_column is None:
            return pd.DataFrame(columns=["airport_code", "latitude", "longitude"])

        return (
            airports_raw.rename(
                columns={
                    code_column: "airport_code",
                    latitude_column: "latitude",
                    longitude_column: "longitude",
                }
            )[["airport_code", "latitude", "longitude"]]
            .assign(
                airport_code=lambda dataframe: normalize_airport_code_series(dataframe["airport_code"]),
                latitude=lambda dataframe: pd.to_numeric(dataframe["latitude"], errors="coerce"),
                longitude=lambda dataframe: pd.to_numeric(dataframe["longitude"], errors="coerce"),
            )
            .dropna(subset=["airport_code", "latitude", "longitude"])
            .drop_duplicates(subset=["airport_code"])
        )


class WeatherFeatureService:
    @staticmethod
    def ensure_weather_dataset() -> pd.DataFrame:
        DatasetFiles.ensure_directories()
        if PATHS.weather_dataset.exists():
            return pd.read_csv(PATHS.weather_dataset, low_memory=False)

        weather_dataset = WeatherFeatureService._build_weather_dataset()
        weather_dataset.to_csv(PATHS.weather_dataset, index=False, encoding=CSV_ENCODING)
        return weather_dataset

    @staticmethod
    def _build_weather_dataset() -> pd.DataFrame:
        archive_end_date = pd.Timestamp.now(tz=REFERENCE.weather_timezone).date() - timedelta(days=2)
        weather_params = {
            "latitude": REFERENCE.weather_api_latitude,
            "longitude": REFERENCE.weather_api_longitude,
            "daily": ",".join(FEATURES.weather_columns),
            "timezone": REFERENCE.weather_timezone,
        }

        archive_response = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={**weather_params, "start_date": "2023-01-01", "end_date": archive_end_date.isoformat()},
            timeout=60,
        )
        archive_response.raise_for_status()

        forecast_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={**weather_params, "past_days": 30, "forecast_days": 16},
            timeout=60,
        )
        forecast_response.raise_for_status()

        archive_daily = archive_response.json().get("daily")
        forecast_daily = forecast_response.json().get("daily")
        if archive_daily is None or forecast_daily is None:
            raise RuntimeError("Open-Meteo response is missing the 'daily' payload.")

        weather_dataset = pd.concat(
            [pd.DataFrame(archive_daily), pd.DataFrame(forecast_daily)],
            ignore_index=True,
        )
        return weather_dataset.drop_duplicates(subset=["time"], keep="first").sort_values("time").reset_index(drop=True)

    @staticmethod
    def apply(flights: pd.DataFrame, weather_dataset: pd.DataFrame) -> pd.DataFrame:
        processed = flights.copy()
        processed[COLUMNS.scheduled_datetime] = pd.to_datetime(processed[COLUMNS.scheduled_datetime], errors="coerce")
        processed["_weather_date"] = processed[COLUMNS.scheduled_datetime].dt.strftime("%Y-%m-%d")

        processed = processed.drop(
            columns=[
                column_name
                for column_name in processed.columns
                if column_name in FEATURES.weather_columns
                or any(column_name.startswith(f"{name}_") for name in FEATURES.weather_columns)
            ],
            errors="ignore",
        )

        weather_lookup = weather_dataset.copy()
        weather_lookup["time"] = pd.to_datetime(weather_lookup["time"], errors="coerce").dt.strftime("%Y-%m-%d")
        merged = processed.merge(weather_lookup, left_on="_weather_date", right_on="time", how="left")
        return merged.drop(columns=["_weather_date", "time"], errors="ignore")


class HolidayFeatureService:
    @classmethod
    def apply(cls, flights: pd.DataFrame) -> pd.DataFrame:
        airports = AirportReferenceService.load_country_mapping()
        holidays = cls._load_destination_holidays_dataset()

        processed = flights.copy().drop(
            columns=[
                "day_of_week",
                "is_weekend",
                "date",
                "date_only",
                "season",
                "remote_country",
                "leg_country",
                "origin_country",
                "dest_country",
                "is_domestic",
                "is_route_domestic",
                "has_stopover",
                "is_fr_public_holiday",
                "is_bridge_day",
                "is_fr_school_holiday_zone_a",
                "is_fr_school_holiday_zone_b",
                "is_fr_school_holiday_zone_c",
                "is_first_last_day_of_school_holiday",
                "is_origin_public_holiday",
                "is_origin_school_holiday",
                "is_dest_public_holiday",
                "is_dest_school_holiday",
                "is_any_day_off",
                "scheduled_date",
                "days_until_next_day_off",
                "days_until_next_workday",
            ],
            errors="ignore",
        )

        processed[COLUMNS.scheduled_datetime] = pd.to_datetime(processed[COLUMNS.scheduled_datetime], errors="coerce")
        processed["day_of_week"] = processed[COLUMNS.scheduled_datetime].dt.dayofweek
        processed["is_weekend"] = processed["day_of_week"].isin([5, 6]).astype(int)
        processed["date"] = processed[COLUMNS.scheduled_datetime].dt.normalize()
        processed["date_only"] = processed[COLUMNS.scheduled_datetime].dt.date
        processed["season"] = processed[COLUMNS.scheduled_datetime].dt.month.apply(get_season)

        processed = cls._add_route_country_features(processed, airports)
        processed = cls._add_france_holiday_features(processed)
        processed = cls._add_foreign_holiday_features(processed, holidays)
        return processed.drop(columns=["date", "date_only"], errors="ignore")

    @staticmethod
    def _load_destination_holidays_dataset() -> pd.DataFrame:
        holidays_path = find_csv_file(
            PATHS.holidays_dir,
            ["openholidays_2023_2027.csv", "openholidays.csv", "open_holidays_2023_2027.csv"],
        )
        if holidays_path is None:
            return pd.DataFrame(columns=["country_code", "type", "start_date", "end_date"])

        holidays_raw = pd.read_csv(holidays_path, sep=None, engine="python")
        country_column = pick_column(holidays_raw.columns, ["country_iso_code", "country_code", "dest_country"])
        type_column = pick_column(holidays_raw.columns, ["holiday_type", "type"])
        start_column = pick_column(holidays_raw.columns, ["start_date", "startdate", "date_start"])
        end_column = pick_column(holidays_raw.columns, ["end_date", "enddate", "date_end"])
        if None in {country_column, type_column, start_column, end_column}:
            return pd.DataFrame(columns=["country_code", "type", "start_date", "end_date"])

        return holidays_raw.rename(
            columns={
                country_column: "country_code",
                type_column: "type",
                start_column: "start_date",
                end_column: "end_date",
            }
        )[["country_code", "type", "start_date", "end_date"]]

    @staticmethod
    def _add_route_country_features(flights: pd.DataFrame, airports: pd.DataFrame) -> pd.DataFrame:
        processed = flights.copy()
        if COLUMNS.airport_previous not in processed.columns:
            processed[COLUMNS.airport_previous] = pd.NA
        if COLUMNS.airport_origin not in processed.columns:
            processed[COLUMNS.airport_origin] = pd.NA

        processed[COLUMNS.airport_previous] = normalize_airport_code_series(processed[COLUMNS.airport_previous])
        processed[COLUMNS.airport_origin] = normalize_airport_code_series(processed[COLUMNS.airport_origin])

        previous_lookup = airports.rename(
            columns={"airport_code": COLUMNS.airport_previous, "country_code": "remote_country"}
        )
        origin_lookup = airports.rename(columns={"airport_code": COLUMNS.airport_origin, "country_code": "leg_country"})
        processed = processed.merge(previous_lookup, on=COLUMNS.airport_previous, how="left")
        processed = processed.merge(origin_lookup, on=COLUMNS.airport_origin, how="left")

        direction = (
            processed[COLUMNS.direction].map(normalize_direction_value)
            if COLUMNS.direction in processed.columns
            else pd.Series(pd.NA, index=processed.index, dtype="object")
        )
        is_arrival = direction.eq("arrivee").fillna(False)
        is_departure = direction.eq("depart").fillna(False)

        processed[COLUMNS.origin_country] = pd.Series(pd.NA, index=processed.index, dtype="object")
        processed.loc[is_arrival, COLUMNS.origin_country] = processed.loc[is_arrival, "remote_country"]
        processed.loc[is_departure, COLUMNS.origin_country] = REFERENCE.lyon_country_code

        processed[COLUMNS.destination_country] = pd.Series(pd.NA, index=processed.index, dtype="object")
        processed.loc[is_arrival, COLUMNS.destination_country] = REFERENCE.lyon_country_code
        processed.loc[is_departure, COLUMNS.destination_country] = processed.loc[is_departure, "remote_country"]

        is_domestic = pd.Series(pd.NA, index=processed.index, dtype="Int64")
        known_leg_country = processed["leg_country"].notna()
        is_domestic.loc[known_leg_country] = (
            processed.loc[known_leg_country, "leg_country"].eq(REFERENCE.lyon_country_code).astype("Int64")
        )
        processed["is_domestic"] = is_domestic

        is_route_domestic = pd.Series(pd.NA, index=processed.index, dtype="Int64")
        known_remote_country = processed["remote_country"].notna()
        is_route_domestic.loc[known_remote_country] = (
            processed.loc[known_remote_country, "remote_country"].eq(REFERENCE.lyon_country_code).astype("Int64")
        )
        processed["is_route_domestic"] = is_route_domestic

        has_stopover = pd.Series(pd.NA, index=processed.index, dtype="Int64")
        stopover_mask = processed[COLUMNS.airport_previous].notna() & processed[COLUMNS.airport_origin].notna()
        has_stopover.loc[stopover_mask] = (
            processed.loc[stopover_mask, COLUMNS.airport_previous]
            != processed.loc[stopover_mask, COLUMNS.airport_origin]
        ).astype("Int64")
        processed["has_stopover"] = has_stopover
        return processed

    @staticmethod
    def _add_france_holiday_features(flights: pd.DataFrame) -> pd.DataFrame:
        processed = flights.copy()
        years = sorted(processed[COLUMNS.scheduled_datetime].dt.year.dropna().astype(int).unique())

        france_public_holidays: set[object] = set()
        for year in years:
            france_public_holidays.update(JoursFeries.for_year(year, zone="Métropole").values())
        processed["is_fr_public_holiday"] = processed["date_only"].isin(france_public_holidays).astype(int)

        holiday_timestamps = set(pd.to_datetime(list(france_public_holidays)))
        flight_dates = pd.to_datetime(processed["date_only"])
        previous_day_is_holiday = (flight_dates - pd.Timedelta(days=1)).isin(holiday_timestamps)
        next_day_is_holiday = (flight_dates + pd.Timedelta(days=1)).isin(holiday_timestamps)
        processed["is_bridge_day"] = (
            ((processed["day_of_week"] == 4) & previous_day_is_holiday)
            | ((processed["day_of_week"] == 0) & next_day_is_holiday)
        ).astype(int)

        school_holidays = SchoolHolidayDates()
        zone_a_days: set[object] = set()
        zone_b_days: set[object] = set()
        zone_c_days: set[object] = set()
        for year in years:
            zone_a_days.update(school_holidays.holidays_for_year_and_zone(year, "A").keys())
            zone_b_days.update(school_holidays.holidays_for_year_and_zone(year, "B").keys())
            zone_c_days.update(school_holidays.holidays_for_year_and_zone(year, "C").keys())

        processed["is_fr_school_holiday_zone_a"] = processed["date_only"].isin(zone_a_days).astype(int)
        processed["is_fr_school_holiday_zone_b"] = processed["date_only"].isin(zone_b_days).astype(int)
        processed["is_fr_school_holiday_zone_c"] = processed["date_only"].isin(zone_c_days).astype(int)

        zone_bc_timestamps = set(pd.to_datetime(list(zone_b_days | zone_c_days)))
        boundary_days = {
            date
            for date in zone_bc_timestamps
            if (date - pd.Timedelta(days=1)) not in zone_bc_timestamps
            or (date + pd.Timedelta(days=1)) not in zone_bc_timestamps
        }
        processed["is_first_last_day_of_school_holiday"] = (
            pd.to_datetime(processed["date_only"]).isin(boundary_days).astype(int)
        )
        return processed

    @staticmethod
    def _add_foreign_holiday_features(flights: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
        if holidays.empty:
            processed = flights.copy()
            for column_name in [
                "is_origin_public_holiday",
                "is_origin_school_holiday",
                "is_dest_public_holiday",
                "is_dest_school_holiday",
            ]:
                processed[column_name] = 0
            return processed

        normalized_holidays = holidays.copy()
        normalized_holidays["country_code"] = normalize_country_code_series(normalized_holidays["country_code"])
        normalized_holidays["type"] = normalized_holidays["type"].astype("string").str.strip().str.lower()
        normalized_holidays["type"] = normalized_holidays["type"].replace(
            {"public_holiday": "public", "school_holiday": "school"}
        )
        normalized_holidays["start_date"] = pd.to_datetime(normalized_holidays["start_date"], errors="coerce")
        normalized_holidays["end_date"] = pd.to_datetime(normalized_holidays["end_date"], errors="coerce")
        normalized_holidays = normalized_holidays[["country_code", "type", "start_date", "end_date"]].dropna()
        normalized_holidays["date"] = [
            list(pd.date_range(start_date, end_date, freq="D"))
            for start_date, end_date in zip(normalized_holidays["start_date"], normalized_holidays["end_date"])
        ]

        holiday_days = normalized_holidays.explode("date")[["country_code", "type", "date"]].drop_duplicates()
        public_holiday_days = holiday_days[holiday_days["type"] == "public"][["country_code", "date"]].drop_duplicates()
        school_holiday_days = holiday_days[holiday_days["type"] == "school"][["country_code", "date"]].drop_duplicates()

        processed = HolidayFeatureService._add_country_holiday_features(
            flights,
            public_holiday_days,
            school_holiday_days,
            country_column=COLUMNS.origin_country,
            public_column="is_origin_public_holiday",
            school_column="is_origin_school_holiday",
        )
        return HolidayFeatureService._add_country_holiday_features(
            processed,
            public_holiday_days,
            school_holiday_days,
            country_column=COLUMNS.destination_country,
            public_column="is_dest_public_holiday",
            school_column="is_dest_school_holiday",
        )

    @staticmethod
    def _add_country_holiday_features(
        flights: pd.DataFrame,
        public_holiday_days: pd.DataFrame,
        school_holiday_days: pd.DataFrame,
        *,
        country_column: str,
        public_column: str,
        school_column: str,
    ) -> pd.DataFrame:
        processed = flights.merge(
            public_holiday_days.rename(columns={"country_code": country_column}).assign(**{public_column: 1}),
            on=[country_column, "date"],
            how="left",
        )
        processed = processed.merge(
            school_holiday_days.rename(columns={"country_code": country_column}).assign(**{school_column: 1}),
            on=[country_column, "date"],
            how="left",
        )

        france_mask = processed[country_column].eq(REFERENCE.lyon_country_code).fillna(False)
        processed.loc[france_mask, public_column] = processed.loc[france_mask, "is_fr_public_holiday"]
        processed.loc[france_mask, school_column] = processed.loc[france_mask, "is_fr_school_holiday_zone_a"]
        processed[public_column] = processed[public_column].fillna(0).astype(int)
        processed[school_column] = processed[school_column].fillna(0).astype(int)
        return processed


class WorldEventRiskService:
    @staticmethod
    def apply(flights: pd.DataFrame) -> pd.DataFrame:
        processed = flights.copy().drop(
            columns=[
                "risk_score",
                "nbDaysSinceRiskStarted",
                "nbDaysSinceSafeStarted",
                "world_event_country_code",
                "world_event_date",
            ],
            errors="ignore",
        )

        required_columns = {COLUMNS.airport_previous, COLUMNS.scheduled_datetime}
        if not required_columns.issubset(processed.columns):
            processed["risk_score"] = 0.0
            processed["nbDaysSinceRiskStarted"] = np.nan
            processed["nbDaysSinceSafeStarted"] = np.nan
            return processed

        if not PATHS.world_events_airports.exists() or not PATHS.world_events_risk.exists():
            processed["risk_score"] = 0.0
            processed["nbDaysSinceRiskStarted"] = np.nan
            processed["nbDaysSinceSafeStarted"] = np.nan
            return processed

        airports = pd.read_parquet(PATHS.world_events_airports, columns=["iata_code", "iso_country"])
        risk_events = pd.read_parquet(
            PATHS.world_events_risk,
            columns=[
                "iso_country",
                "Date",
                "SmoothedRisk",
                "nbDaysSinceRiskStarted",
                "nbDaysSinceSafeStarted",
            ],
        )

        airports["iata_code"] = normalize_airport_code_series(airports["iata_code"])
        airports["iso_country"] = normalize_country_code_series(airports["iso_country"])
        risk_events["iso_country"] = normalize_country_code_series(risk_events["iso_country"])
        risk_events["Date"] = pd.to_datetime(risk_events["Date"], errors="coerce").dt.normalize()
        risk_events["SmoothedRisk"] = pd.to_numeric(risk_events["SmoothedRisk"], errors="coerce")
        risk_events["nbDaysSinceRiskStarted"] = pd.to_numeric(risk_events["nbDaysSinceRiskStarted"], errors="coerce")
        risk_events["nbDaysSinceSafeStarted"] = pd.to_numeric(risk_events["nbDaysSinceSafeStarted"], errors="coerce")

        airport_lookup = airports.rename(
            columns={"iata_code": COLUMNS.airport_previous, "iso_country": "world_event_country_code"}
        ).drop_duplicates(subset=[COLUMNS.airport_previous], keep="first")
        risk_lookup = (
            risk_events.rename(
                columns={
                    "iso_country": "world_event_country_code",
                    "Date": "world_event_date",
                    "SmoothedRisk": "risk_score",
                }
            )
            .dropna(subset=["world_event_country_code", "world_event_date"])
            .drop_duplicates(subset=["world_event_country_code", "world_event_date"], keep="last")
        )

        processed[COLUMNS.airport_previous] = normalize_airport_code_series(processed[COLUMNS.airport_previous])
        processed["world_event_date"] = pd.to_datetime(
            processed[COLUMNS.scheduled_datetime], errors="coerce"
        ).dt.normalize()
        processed = processed.merge(airport_lookup, on=COLUMNS.airport_previous, how="left")
        processed = processed.merge(risk_lookup, on=["world_event_country_code", "world_event_date"], how="left")
        processed["risk_score"] = pd.to_numeric(processed["risk_score"], errors="coerce").fillna(0.0)
        processed["nbDaysSinceRiskStarted"] = pd.to_numeric(processed["nbDaysSinceRiskStarted"], errors="coerce")
        processed["nbDaysSinceSafeStarted"] = pd.to_numeric(processed["nbDaysSinceSafeStarted"], errors="coerce")
        return processed.drop(columns=["world_event_country_code", "world_event_date"], errors="ignore")


class PreprocessingFeatureService:
    @classmethod
    def apply(cls, flights: pd.DataFrame) -> pd.DataFrame:
        processed = flights.copy()
        processed[COLUMNS.scheduled_datetime] = pd.to_datetime(processed[COLUMNS.scheduled_datetime], errors="coerce")
        processed = cls.add_cyclical_datetime_features(processed)
        processed = cls.add_day_off_countdown_features(processed)
        processed = cls.add_lag_and_rolling_features(processed)
        processed = cls.add_distance_features(processed)
        return processed

    @staticmethod
    def add_cyclical_datetime_features(flights: pd.DataFrame) -> pd.DataFrame:
        processed = flights.copy()
        scheduled = processed[COLUMNS.scheduled_datetime]
        processed["day_of_week"] = scheduled.dt.dayofweek
        processed["is_weekend"] = processed["day_of_week"].isin([5, 6]).astype(int)
        processed["month_of_year"] = scheduled.dt.month
        processed["hour_of_day"] = scheduled.dt.hour + scheduled.dt.minute / 60.0 + scheduled.dt.second / 3600.0
        processed["hour_sin"] = np.sin(2 * np.pi * processed["hour_of_day"] / 24.0)
        processed["hour_cos"] = np.cos(2 * np.pi * processed["hour_of_day"] / 24.0)
        processed["day_of_week_sin"] = np.sin(2 * np.pi * processed["day_of_week"] / 7.0)
        processed["day_of_week_cos"] = np.cos(2 * np.pi * processed["day_of_week"] / 7.0)
        processed["month_sin"] = np.sin(2 * np.pi * (processed["month_of_year"] - 1) / 12.0)
        processed["month_cos"] = np.cos(2 * np.pi * (processed["month_of_year"] - 1) / 12.0)
        return processed

    @classmethod
    def add_day_off_countdown_features(cls, flights: pd.DataFrame) -> pd.DataFrame:
        processed = flights.copy().drop(
            columns=["scheduled_date", "is_any_day_off", "days_until_next_day_off", "days_until_next_workday"],
            errors="ignore",
        )
        cls._ensure_day_off_columns_exist(processed)

        if COLUMNS.origin_country not in processed.columns:
            processed[COLUMNS.origin_country] = np.nan
        if COLUMNS.destination_country not in processed.columns:
            processed[COLUMNS.destination_country] = np.nan

        processed["scheduled_date"] = processed[COLUMNS.scheduled_datetime].dt.normalize()
        processed[COLUMNS.origin_country] = processed[COLUMNS.origin_country].fillna("__UNKNOWN__")
        processed[COLUMNS.destination_country] = processed[COLUMNS.destination_country].fillna("__UNKNOWN__")
        processed["is_any_day_off"] = (
            processed[list(FEATURES.day_off_columns)].fillna(0).astype(int).max(axis=1).astype(int)
        )

        daily_calendar = (
            processed[[COLUMNS.origin_country, COLUMNS.destination_country, "scheduled_date", "is_any_day_off"]]
            .dropna(subset=["scheduled_date"])
            .groupby([COLUMNS.origin_country, COLUMNS.destination_country, "scheduled_date"], as_index=False)[
                "is_any_day_off"
            ]
            .max()
            .sort_values([COLUMNS.origin_country, COLUMNS.destination_country, "scheduled_date"])
            .reset_index(drop=True)
        )

        countdown_frames = [
            cls.compute_group_day_off_countdowns(group)
            for _, group in daily_calendar.groupby([COLUMNS.origin_country, COLUMNS.destination_country], sort=False)
        ]
        if countdown_frames:
            countdowns = pd.concat(countdown_frames, ignore_index=True)
            processed = processed.merge(
                countdowns,
                on=[COLUMNS.origin_country, COLUMNS.destination_country, "scheduled_date"],
                how="left",
            )
        else:
            processed["days_until_next_day_off"] = np.nan
            processed["days_until_next_workday"] = np.nan

        processed[COLUMNS.origin_country] = processed[COLUMNS.origin_country].replace("__UNKNOWN__", np.nan)
        processed[COLUMNS.destination_country] = processed[COLUMNS.destination_country].replace("__UNKNOWN__", np.nan)
        return processed

    @staticmethod
    def _ensure_day_off_columns_exist(flights: pd.DataFrame) -> None:
        missing_columns = [
            column_name for column_name in FEATURES.day_off_columns if column_name not in flights.columns
        ]
        if missing_columns:
            raise RuntimeError(f"Missing day-off columns for preprocessing: {missing_columns}")

    @staticmethod
    def compute_group_day_off_countdowns(group: pd.DataFrame) -> pd.DataFrame:
        processed = group.sort_values("scheduled_date").reset_index(drop=True).copy()
        dates = pd.to_datetime(processed["scheduled_date"]).dt.date.tolist()
        is_day_off = processed["is_any_day_off"].astype(bool).tolist()

        next_day_off_date = None
        next_workday_date = None
        days_until_next_day_off = [np.nan] * len(processed)
        days_until_next_workday = [np.nan] * len(processed)

        for index in range(len(processed) - 1, -1, -1):
            current_date = dates[index]
            if is_day_off[index]:
                days_until_next_day_off[index] = 0
                if next_workday_date is not None:
                    days_until_next_workday[index] = (next_workday_date - current_date).days
                next_day_off_date = current_date
            else:
                days_until_next_workday[index] = 0
                if next_day_off_date is not None:
                    days_until_next_day_off[index] = (next_day_off_date - current_date).days
                next_workday_date = current_date

        processed["days_until_next_day_off"] = days_until_next_day_off
        processed["days_until_next_workday"] = days_until_next_workday
        return processed[
            [
                COLUMNS.origin_country,
                COLUMNS.destination_country,
                "scheduled_date",
                "days_until_next_day_off",
                "days_until_next_workday",
            ]
        ]

    @staticmethod
    def add_lag_and_rolling_features(flights: pd.DataFrame) -> pd.DataFrame:
        processed = flights.copy().drop(columns=list(FEATURES.lag_columns), errors="ignore")
        required_columns = {COLUMNS.target, COLUMNS.scheduled_datetime, COLUMNS.flight_number}
        if not required_columns.issubset(processed.columns):
            for column_name in FEATURES.lag_columns:
                processed[column_name] = np.nan
            return processed

        processed["_temp_date"] = pd.to_datetime(processed[COLUMNS.scheduled_datetime], errors="coerce").dt.normalize()
        daily_history = (
            processed.dropna(subset=[COLUMNS.flight_number])
            .groupby([COLUMNS.flight_number, "_temp_date"])[COLUMNS.target]
            .mean()
            .reset_index()
            .rename(columns={COLUMNS.target: "_pax"})
            .sort_values([COLUMNS.flight_number, "_temp_date"])
            .reset_index(drop=True)
        )

        for n_days, column_name in [(1, "pax_lag_1"), (7, "pax_lag_7"), (30, "pax_lag_30")]:
            lag_lookup = daily_history[[COLUMNS.flight_number, "_temp_date", "_pax"]].copy()
            lag_lookup["_temp_date"] = lag_lookup["_temp_date"] + pd.Timedelta(days=n_days)
            lag_lookup = lag_lookup.rename(columns={"_pax": column_name})
            processed = processed.merge(lag_lookup, on=[COLUMNS.flight_number, "_temp_date"], how="left")

        daily_history["_rolling_7"] = daily_history.groupby(COLUMNS.flight_number)["_pax"].transform(
            lambda series: series.shift(1).rolling(window=7, min_periods=1).mean()
        )
        daily_history["_ewm_7"] = daily_history.groupby(COLUMNS.flight_number)["_pax"].transform(
            lambda series: series.shift(1).ewm(span=7, adjust=False, min_periods=1).mean()
        )
        daily_history["_diff_1"] = daily_history.groupby(COLUMNS.flight_number)["_pax"].transform(
            lambda series: series.shift(1).diff(1)
        )
        daily_history["_diff_7"] = daily_history.groupby(COLUMNS.flight_number)["_pax"].transform(
            lambda series: series.shift(1).diff(7)
        )

        rolling_lookup = daily_history[
            [COLUMNS.flight_number, "_temp_date", "_rolling_7", "_ewm_7", "_diff_1", "_diff_7"]
        ].rename(
            columns={
                "_rolling_7": "pax_rolling_7",
                "_ewm_7": "pax_ewm_7",
                "_diff_1": "pax_diff_1",
                "_diff_7": "pax_diff_7",
            }
        )
        processed = processed.merge(rolling_lookup, on=[COLUMNS.flight_number, "_temp_date"], how="left")
        return processed.drop(columns=["_temp_date"], errors="ignore")

    @staticmethod
    def add_distance_features(flights: pd.DataFrame) -> pd.DataFrame:
        processed = flights.copy().drop(columns=["flight_distance_km"], errors="ignore")
        if COLUMNS.airport_previous not in processed.columns:
            processed["flight_distance_km"] = np.nan
            return processed

        airports = AirportReferenceService.load_airports_with_coordinates()
        if airports.empty:
            processed["flight_distance_km"] = np.nan
            return processed

        lyon_row = airports.loc[airports["airport_code"] == REFERENCE.lyon_airport_code]
        lyon_latitude = float(lyon_row["latitude"].iloc[0]) if not lyon_row.empty else REFERENCE.lyon_latitude
        lyon_longitude = float(lyon_row["longitude"].iloc[0]) if not lyon_row.empty else REFERENCE.lyon_longitude

        remote_lookup = airports.rename(
            columns={"airport_code": COLUMNS.airport_previous, "latitude": "_remote_lat", "longitude": "_remote_lon"}
        )
        processed = processed.merge(remote_lookup, on=COLUMNS.airport_previous, how="left")

        has_coordinates = processed["_remote_lat"].notna() & processed["_remote_lon"].notna()
        distances = pd.Series(np.nan, index=processed.index)
        if has_coordinates.any():
            distances.loc[has_coordinates] = haversine_km(
                processed.loc[has_coordinates, "_remote_lat"],
                processed.loc[has_coordinates, "_remote_lon"],
                lyon_latitude,
                lyon_longitude,
            )
        processed["flight_distance_km"] = distances
        return processed.drop(columns=["_remote_lat", "_remote_lon"], errors="ignore")


def load_main_dataset() -> None:
    BigQueryDatasetDownloader.ensure_original_dataset()


def load_weather_data() -> None:
    BigQueryDatasetDownloader.ensure_original_dataset()
    weather_dataset = WeatherFeatureService.ensure_weather_dataset()
    source = DatasetFiles.load_main_dataset() if PATHS.main_dataset.exists() else DatasetFiles.load_original_dataset()
    DatasetFiles.save_main_dataset(WeatherFeatureService.apply(source, weather_dataset))


def load_holiday_data() -> None:
    BigQueryDatasetDownloader.ensure_original_dataset()
    source = DatasetFiles.load_main_dataset() if PATHS.main_dataset.exists() else DatasetFiles.load_original_dataset()
    DatasetFiles.save_main_dataset(HolidayFeatureService.apply(source))


def load_world_event_data() -> None:
    BigQueryDatasetDownloader.ensure_original_dataset()
    source = DatasetFiles.load_main_dataset() if PATHS.main_dataset.exists() else DatasetFiles.load_original_dataset()
    DatasetFiles.save_main_dataset(WorldEventRiskService.apply(source))


def apply_preprocessing(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> None:
    source_path = (
        Path(input_path)
        if input_path is not None
        else (PATHS.main_dataset if PATHS.main_dataset.exists() else PATHS.original_dataset)
    )
    destination_path = Path(output_path) if output_path is not None else PATHS.main_dataset
    flights = pd.read_csv(source_path, low_memory=False)
    processed = PreprocessingFeatureService.apply(flights)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(destination_path, index=False, encoding=CSV_ENCODING)


if __name__ == "__main__":
    load_dotenv()

    BigQueryDatasetDownloader.ensure_original_dataset()
    weather_dataset = WeatherFeatureService.ensure_weather_dataset()

    flights = DatasetFiles.load_original_dataset()
    flights = WeatherFeatureService.apply(flights, weather_dataset)
    flights = HolidayFeatureService.apply(flights)
    flights = WorldEventRiskService.apply(flights)
    flights = PreprocessingFeatureService.apply(flights)

    DatasetFiles.save_main_dataset(flights)
