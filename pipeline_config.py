from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True)
class DatasetPaths:
    data_dir: Path
    weather_dir: Path
    holidays_dir: Path
    world_events_dir: Path
    original_dataset: Path
    main_dataset: Path
    weather_dataset: Path
    world_events_airports: Path
    world_events_risk: Path

    @classmethod
    def for_project_root(cls, project_root: Path) -> "DatasetPaths":
        data_dir = project_root / "data"
        weather_dir = data_dir / "weather"
        holidays_dir = data_dir / "holidays"
        world_events_dir = data_dir / "world_events"
        return cls(
            data_dir=data_dir,
            weather_dir=weather_dir,
            holidays_dir=holidays_dir,
            world_events_dir=world_events_dir,
            original_dataset=data_dir / "original_dataset.csv",
            main_dataset=data_dir / "main_dataset.csv",
            weather_dataset=weather_dir / "weather.csv",
            world_events_airports=world_events_dir / "clean_airports.parquet",
            world_events_risk=world_events_dir / "smoothed_risk_events.parquet",
        )


@dataclass(frozen=True, slots=True)
class ModelPaths:
    predictions_dir: Path
    model_configs_dir: Path
    hyperparameters_results: Path

    @classmethod
    def for_project_root(cls, project_root: Path) -> "ModelPaths":
        return cls(
            predictions_dir=project_root / "predictions",
            model_configs_dir=project_root / "model_configs",
            hyperparameters_results=project_root / "hyperparameters.json",
        )


@dataclass(frozen=True, slots=True)
class ColumnNames:
    id: str = "IdMovement"
    row_id: str = "row_number"
    scheduled_datetime: str = "LTScheduledDatetime"
    target: str = "NbPaxTotal"
    pmr_target: str = "PMR"
    pmr_arrival: str = "OzionPHMRPaxArrival"
    pmr_departure: str = "OzionPHMRPaxDeparture"
    flight_number: str = "FlightNumberNormalized"
    direction: str = "Direction"
    airport_previous: str = "AirportPrevious"
    airport_origin: str = "AirportOrigin"
    origin_country: str = "origin_country"
    destination_country: str = "dest_country"


@dataclass(frozen=True, slots=True)
class FeatureNames:
    day_off_columns: tuple[str, ...] = (
        "is_fr_public_holiday",
        "is_fr_school_holiday_zone_a",
        "is_origin_public_holiday",
        "is_origin_school_holiday",
        "is_dest_public_holiday",
        "is_dest_school_holiday",
        "is_weekend",
    )
    weather_columns: tuple[str, ...] = (
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
        "windspeed_10m_max",
    )
    lag_columns: tuple[str, ...] = (
        "pax_lag_1",
        "pax_lag_7",
        "pax_lag_30",
        "pax_rolling_7",
        "pax_ewm_7",
        "pax_diff_1",
        "pax_diff_7",
    )
    prophet_columns: tuple[str, ...] = (
        "daily_seats",
        "daily_forecast_pax",
        "daily_forecast_load_factor",
        "relative_seat_share_day",
        "forecast_pax_per_seat_day",
    )


@dataclass(frozen=True, slots=True)
class ReferenceValues:
    lyon_country_code: str = "FR"
    lyon_airport_code: str = "LYS"
    lyon_latitude: float = 45.725556
    lyon_longitude: float = 4.8414
    weather_api_latitude: float = 45.7589
    weather_api_longitude: float = 4.8414
    weather_timezone: str = "Europe/Paris"


@dataclass(frozen=True, slots=True)
class PlotConfig:
    max_points: int = 80_000


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    dataset_paths: DatasetPaths
    model_paths: ModelPaths
    columns: ColumnNames = ColumnNames()
    features: FeatureNames = FeatureNames()
    reference: ReferenceValues = ReferenceValues()
    plot: PlotConfig = PlotConfig()
    null_like_strings: frozenset[str] = frozenset({"", "NULL", "NONE", "NAN", "NAT"})

    @classmethod
    def for_project_root(cls, project_root: Path) -> "PipelineConfig":
        return cls(
            dataset_paths=DatasetPaths.for_project_root(project_root),
            model_paths=ModelPaths.for_project_root(project_root),
        )


PIPELINE_CONFIG = PipelineConfig.for_project_root(PROJECT_ROOT)
