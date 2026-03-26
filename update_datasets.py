import requests
import os, pandas as pd
from pathlib import Path
from utils import ProgressBar
from dotenv import load_dotenv

try:
    from google.cloud import bigquery
    from google.cloud.bigquery.table import Row
except ImportError:
    bigquery = None
    Row = Any

try:
    from utils import ProgressBar
except ImportError:

    class ProgressBar:
        def __init__(self, total: int, update_every: int = 1):
            self.total = total
            self.update_every = update_every

        def start(self) -> None:
            return

        def increment(self) -> None:
            return

        def finish(self) -> None:
            return


DAY_OFF_COLUMNS = [
    "is_fr_public_holiday",
    "is_fr_school_holiday_zone_a",
    "is_dest_public_holiday",
    "is_dest_school_holiday",
    "is_weekend",
]


def _normalize_filename(filename: str) -> str:
    return "".join(character for character in filename.lower() if character.isalnum())


def _find_holiday_csv(folder: Path, candidate_names: list[str]) -> Path | None:
    for file_name in candidate_names:
        direct_path = folder / file_name
        if direct_path.exists():
            return direct_path

    candidate_tokens = {_normalize_filename(name) for name in candidate_names}
    for csv_path in folder.glob("*.csv"):
        if _normalize_filename(csv_path.name) in candidate_tokens:
            return csv_path

    return None


def _pick_column(columns: pd.Index, candidates: list[str]) -> str | None:
    lowered = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        match = lowered.get(candidate.lower())
        if match is not None:
            return match
    return None


def _load_airports_dataset() -> pd.DataFrame:
    airports_candidates = [
        "upply-airports.csv",
        "uply-airports.csv",
        "upply_airports.csv",
        "airports.csv",
    ]
    airports_path = _find_holiday_csv(HOLIDAYS_FOLDER, airports_candidates)
    if airports_path is None:
        print(
            f"Warning: no airport mapping CSV found in {HOLIDAYS_FOLDER}. "
            "Destination country holiday features will default to 0."
        )
        return pd.DataFrame(columns=["AirportPrevious", "dest_country"])

    airports_raw = pd.read_csv(airports_path, sep=None, engine="python")
    code_column = _pick_column(
        airports_raw.columns,
        ["code", "airportprevious", "airport_code", "airportcode", "iata", "iata_code"],
    )
    country_column = _pick_column(
        airports_raw.columns,
        ["country_code", "country_iso_code", "dest_country", "country"],
    )

    if code_column is None or country_column is None:
        print(
            f"Warning: airports CSV '{airports_path.name}' does not contain expected columns. "
            "Destination country holiday features will default to 0."
        )
        return pd.DataFrame(columns=["AirportPrevious", "dest_country"])

    airports = (
        airports_raw.rename(columns={code_column: "AirportPrevious", country_column: "dest_country"})[
            ["AirportPrevious", "dest_country"]
        ]
        .dropna(subset=["AirportPrevious"])
        .drop_duplicates()
    )
    print(f"Loaded airport mapping from {airports_path}")
    return airports


def _load_destination_holidays_dataset() -> pd.DataFrame:
    holidays_candidates = [
        "openholidays_2023_2027.csv",
        "openholidays.csv",
        "open_holidays_2023_2027.csv",
    ]
    holidays_path = _find_holiday_csv(HOLIDAYS_FOLDER, holidays_candidates)
    if holidays_path is None:
        print(
            f"Warning: no destination holiday CSV found in {HOLIDAYS_FOLDER}. "
            "Destination country holiday features will default to 0."
        )
        return pd.DataFrame(columns=["country_code", "type", "start_date", "end_date"])

    holidays_raw = pd.read_csv(holidays_path, sep=None, engine="python")
    country_column = _pick_column(holidays_raw.columns, ["country_iso_code", "country_code", "dest_country"])
    type_column = _pick_column(holidays_raw.columns, ["holiday_type", "type"])
    start_column = _pick_column(holidays_raw.columns, ["start_date", "startdate", "date_start"])
    end_column = _pick_column(holidays_raw.columns, ["end_date", "enddate", "date_end"])

    if None in [country_column, type_column, start_column, end_column]:
        print(
            f"Warning: destination holidays CSV '{holidays_path.name}' does not contain expected columns. "
            "Destination country holiday features will default to 0."
        )
        return pd.DataFrame(columns=["country_code", "type", "start_date", "end_date"])

    holidays = holidays_raw.rename(
        columns={
            country_column: "country_code",
            type_column: "type",
            start_column: "start_date",
            end_column: "end_date",
        }
    )[["country_code", "type", "start_date", "end_date"]]
    print(f"Loaded destination holidays from {holidays_path}")
    return holidays


def load_main_dataset() -> None:
    """
    Exécute un SELECT * sur une table BigQuery et affiche les résultats.

    Args:
        project_id (str): L'ID du projet GCP.
        dataset_id (str): L'ID du dataset BigQuery.
        table_id (str): L'ID de la table BigQuery.
        service_account_key_path (str): Le chemin vers le fichier JSON de la clé du compte de service.
    """

    if MAIN_DATASET_FILE.exists():
        print(f"\tMain dataset already exists at {MAIN_DATASET_FILE}. Skipping download.")
        return

    # Configure les identifiants du compte de service
    project_id = os.getenv("PROJECT_ID")
    dataset_id = os.getenv("DATASET_ID")
    table_id = os.getenv("TABLE_ID")
    service_account_key_path = os.getenv("SERVICE_ACCOUNT_KEY_PATH")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_key_path

    table_ref = f"`{project_id}.{dataset_id}.{table_id}`"

    query = f"SELECT * FROM {table_ref};"

    # Initialise le client BigQuery
    client = bigquery.Client(project=project_id)

    # Construit la référence complète de la table
    print(f"\tExécution de la requête sur BigQuery: {query}\n")

    query_job = client.query(query)

    print("\tGot response")

    # Récupère les résultats
    rows: Iterator[Row] = query_job.result()

    print(f"\tRequête terminée. {rows.total_rows} lignes récupérées.")

    progress_bar = ProgressBar(rows.total_rows, update_every=max(1, rows.total_rows // 100))
    progress_bar.start()

    dataset_rows = []

    for row in rows:
        dataset_rows.append(dict(row.items()))
        progress_bar.increment()

    progress_bar.finish()

    dataset = pd.DataFrame(dataset_rows)
    dataset.to_csv(MAIN_DATASET_FILE, index=False)
    dataset.to_csv(ORIGINAL_DATASET_FILE, index=False)

    print(f"\tMain dataset saved to {MAIN_DATASET_FILE}")


def load_weather_data():
    if WEATHER_FILE.exists():
        print(f"Weather dataset already exists at {WEATHER_FILE}. Skipping download.")
        weather_dataset = pd.read_csv(WEATHER_FILE)
    else:

        # 1. Define our API endpoints (using JSON)
        # Archive API: From Jan 1, 2023 up to today (it will automatically stop at its most recent available day)
        url_archive = "https://archive-api.open-meteo.com/v1/archive?latitude=45.7589&longitude=4.8414&start_date=2023-01-01&end_date=2026-03-25&daily=precipitation_sum,rain_sum,snowfall_sum,windspeed_10m_max&timezone=Europe%2FParis"

        # Forecast API: We use past_days=30 to ensure we cover any gap, plus forecast_days=16
        url_forecast = "https://api.open-meteo.com/v1/forecast?latitude=45.7589&longitude=4.8414&past_days=30&forecast_days=16&daily=precipitation_sum,rain_sum,snowfall_sum,windspeed_10m_max&timezone=Europe%2FParis"

        # 2. Fetch the data from the web
        print("Fetching Archive data...")
        archive_data = requests.get(url_archive).json()["daily"]

        print("Fetching Forecast data...")
        forecast_data = requests.get(url_forecast).json()["daily"]

        # 3. Convert both into Pandas DataFrames
        df_archive = pd.DataFrame(archive_data)
        df_forecast = pd.DataFrame(forecast_data)

        # 4. Stitch them together
        # We stack them on top of each other. df_archive is first, df_forecast is second.
        weather_dataset = pd.concat([df_archive, df_forecast])

        # 5. The Magic Step: Drop duplicates
        # We drop rows with the same date ('time').
        # keep='first' means if a date exists in both datasets, we keep the Archive version,
        # which is preferred because historical data is finalized and more accurate than past forecasts.
        weather_dataset = weather_dataset.drop_duplicates(subset=["time"], keep="first")

        # 6. Sort chronologically just to be perfectly safe, and reset the row numbers
        weather_dataset = weather_dataset.sort_values("time").reset_index(drop=True)
        weather_dataset.to_csv(WEATHER_FILE, index=False)
        print(f"Weather dataset saved to {WEATHER_FILE}")

    main_dataset = pd.read_csv(MAIN_DATASET_FILE)

    # 2. Convert the main column (in dataframe) to datetime so we can manipulate it
    main_dataset["LTScheduledDatetime"] = pd.to_datetime(
        main_dataset["LTScheduledDatetime"], format="%Y-%m-%d %H:%M:%S"
    )

    # 3. Extract JUST the date into a temporary string column (e.g., '2026-04-22')
    main_dataset["Temp_Date_Match"] = main_dataset["LTScheduledDatetime"].dt.strftime("%Y-%m-%d")

    # 4. Standardize the second dataset's 'time' column to the exact same string format.
    # (We convert it to datetime first just in case, then format it as a string)
    weather_dataset["time"] = pd.to_datetime(weather_dataset["time"]).dt.strftime("%Y-%m-%d")

    # Ensure weather columns are refreshed cleanly if this function is rerun.
    weather_features = {"precipitation_sum", "rain_sum", "snowfall_sum", "windspeed_10m_max"}
    weather_columns_to_drop = [
        column_name
        for column_name in main_dataset.columns
        if column_name in weather_features
        or (column_name.endswith("_x") and column_name[:-2] in weather_features)
        or (column_name.endswith("_y") and column_name[:-2] in weather_features)
    ]
    if weather_columns_to_drop:
        main_dataset = main_dataset.drop(columns=weather_columns_to_drop)

    # 5. Merge them!
    df_merged = pd.merge(main_dataset, weather_dataset, left_on="Temp_Date_Match", right_on="time", how="left")

    # 6. Clean up (drop the temporary column and the redundant 'time' column)
    df_merged = df_merged.drop(columns=["Temp_Date_Match", "time"])

    # 7. Save it back to CSV
    df_merged.to_csv(MAIN_DATASET_FILE, index=False)

    print(f"Success! Saved {len(weather_dataset)} days of continuous weather data.")


def load_holiday_data():
    from jours_feries_france import JoursFeries
    from vacances_scolaires_france import SchoolHolidayDates

    # =========================
    # 1. Download data
    # =========================

    flights = pd.read_csv(MAIN_DATASET_FILE)
    airports = _load_airports_dataset()
    holidays = _load_destination_holidays_dataset()

    derived_columns = [
        "day_of_week",
        "is_weekend",
        "date",
        "date_only",
        "season",
        "dest_country",
        "is_fr_public_holiday",
        "is_fr_school_holiday_zone_a",
        "is_dest_public_holiday",
        "is_dest_school_holiday",
        "is_any_day_off",
        "scheduled_date",
        "days_until_next_day_off",
        "days_until_next_workday",
    ]
    existing_derived = [column_name for column_name in derived_columns if column_name in flights.columns]
    if existing_derived:
        flights = flights.drop(columns=existing_derived)

    flights["LTScheduledDatetime"] = pd.to_datetime(flights["LTScheduledDatetime"], errors="coerce")

    # Number of day of the week : 0=monday ... 6=sunday
    flights["day_of_week"] = flights["LTScheduledDatetime"].dt.dayofweek

    # Weekend : 1 if saturday or sunday, otherwise 0
    flights["is_weekend"] = flights["day_of_week"].isin([5, 6]).astype(int)

    # Date used to join
    flights["date"] = flights["LTScheduledDatetime"].dt.normalize()
    flights["date_only"] = flights["LTScheduledDatetime"].dt.date

    def get_season(month):
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "autumn"

    flights["season"] = flights["LTScheduledDatetime"].dt.month.apply(get_season)

    # =========================
    # 3. MAPPING AIRPORT -> COUNTRY
    # =========================

    if not airports.empty and "AirportPrevious" in flights.columns:
        flights = flights.merge(airports, on="AirportPrevious", how="left")
    else:
        flights["dest_country"] = np.nan

    # =========================
    # 4. FRANCE : PUBLIC HOLIDAY
    # =========================

    years = sorted(flights["LTScheduledDatetime"].dt.year.dropna().astype(int).unique())

    fr_public_holidays = set()
    for y in years:
        holidays_dict = JoursFeries.for_year(y, zone="Métropole")
        fr_public_holidays.update(holidays_dict.values())

    flights["is_fr_public_holiday"] = flights["date_only"].isin(fr_public_holidays).astype(int)

    # =========================
    # 5. FRANCE : SCHOOL HOLIDAY
    # =========================
    # Lyon = zone A

    school_holidays = SchoolHolidayDates()

    fr_school_holidays_zone_a = set()
    for y in years:
        year_holidays = school_holidays.holidays_for_year_and_zone(y, "A")
        fr_school_holidays_zone_a.update(year_holidays.keys())

    flights["is_fr_school_holiday_zone_a"] = flights["date_only"].isin(fr_school_holidays_zone_a).astype(int)

    # =========================
    # 6. DESTINATION : HOLIDAYS
    # =========================

    if holidays.empty:
        dest_public = pd.DataFrame(columns=["dest_country", "date", "is_dest_public_holiday"])
        dest_school = pd.DataFrame(columns=["dest_country", "date", "is_dest_school_holiday"])
    else:
        holidays["type"] = holidays["type"].replace({"public_holiday": "public", "school_holiday": "school"})

        holidays["start_date"] = pd.to_datetime(holidays["start_date"], errors="coerce")
        holidays["end_date"] = pd.to_datetime(holidays["end_date"], errors="coerce")

        holidays = holidays.dropna(subset=["country_code", "type", "start_date", "end_date"])
        holidays["date"] = holidays.apply(
            lambda row: pd.date_range(row["start_date"], row["end_date"], freq="D"), axis=1
        )
        holiday_days = holidays.explode("date")[["country_code", "type", "date"]].drop_duplicates()

        # destination country : public holiday
        dest_public = (
            holiday_days[holiday_days["type"] == "public"][["country_code", "date"]]
            .rename(columns={"country_code": "dest_country"})
            .drop_duplicates()
            .assign(is_dest_public_holiday=1)
        )

        # destination country : school holiday
        dest_school = (
            holiday_days[holiday_days["type"] == "school"][["country_code", "date"]]
            .rename(columns={"country_code": "dest_country"})
            .drop_duplicates()
            .assign(is_dest_school_holiday=1)
        )

    flights = flights.merge(dest_public, on=["dest_country", "date"], how="left")

    flights = flights.merge(dest_school, on=["dest_country", "date"], how="left")

    flights["is_dest_public_holiday"] = flights["is_dest_public_holiday"].fillna(0).astype(int)
    flights["is_dest_school_holiday"] = flights["is_dest_school_holiday"].fillna(0).astype(int)

    flights = flights.drop(columns=["date", "date_only"])

    # =========================
    # 7. EXPORT
    # =========================

    flights.to_csv(DATA_FOLDER / "main_dataset.csv", index=False, encoding="utf-8-sig")

    # os.system("pip uninstall jours_feries_france vacances_scolaires_france -y")


def merge_datasets():
    load_weather_data()
    load_holiday_data()
    apply_preprocessing()


def apply_preprocessing(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> None:
    input_path = Path(input_path or MAIN_DATASET_FILE)
    output_path = Path(output_path or MAIN_DATASET_FILE)

    flights = pd.read_csv(input_path)
    flights["LTScheduledDatetime"] = pd.to_datetime(flights["LTScheduledDatetime"], errors="coerce")

    flights = add_cyclical_datetime_features(flights)
    flights = add_day_off_countdown_features(flights)

    flights.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Preprocessed dataset saved to {output_path}")


def add_cyclical_datetime_features(flights: pd.DataFrame) -> pd.DataFrame:
    processed = flights.copy()
    scheduled = processed["LTScheduledDatetime"]

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


def add_day_off_countdown_features(flights: pd.DataFrame) -> pd.DataFrame:
    processed = flights.copy()
    ensure_day_off_columns_exist(processed)

    processed["scheduled_date"] = processed["LTScheduledDatetime"].dt.normalize()
    processed["dest_country"] = processed["dest_country"].fillna("__UNKNOWN__")

    day_off_flags = processed[DAY_OFF_COLUMNS].fillna(0).astype(int)
    processed["is_any_day_off"] = day_off_flags.max(axis=1).astype(int)

    daily_calendar = (
        processed[["dest_country", "scheduled_date", "is_any_day_off"]]
        .dropna(subset=["scheduled_date"])
        .groupby(["dest_country", "scheduled_date"], as_index=False)["is_any_day_off"]
        .max()
        .sort_values(["dest_country", "scheduled_date"])
        .reset_index(drop=True)
    )

    countdown_frames = []
    for _, group in daily_calendar.groupby("dest_country", sort=False):
        countdown_frames.append(compute_group_day_off_countdowns(group))

    if countdown_frames:
        countdowns = pd.concat(countdown_frames, ignore_index=True)
        processed = processed.merge(
            countdowns,
            on=["dest_country", "scheduled_date"],
            how="left",
        )
    else:
        processed["days_until_next_day_off"] = np.nan
        processed["days_until_next_workday"] = np.nan

    processed["dest_country"] = processed["dest_country"].replace("__UNKNOWN__", np.nan)
    return processed


def ensure_day_off_columns_exist(flights: pd.DataFrame) -> None:
    missing_columns = [column_name for column_name in DAY_OFF_COLUMNS if column_name not in flights.columns]
    if missing_columns:
        raise RuntimeError(f"Missing day-off columns for preprocessing: {missing_columns}")


def compute_group_day_off_countdowns(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("scheduled_date").reset_index(drop=True).copy()
    dates = pd.to_datetime(group["scheduled_date"]).dt.date.tolist()
    is_day_off = group["is_any_day_off"].astype(bool).tolist()

    next_day_off_date = None
    next_workday_date = None
    days_until_next_day_off = [np.nan] * len(group)
    days_until_next_workday = [np.nan] * len(group)

    for index in range(len(group) - 1, -1, -1):
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

    group["days_until_next_day_off"] = days_until_next_day_off
    group["days_until_next_workday"] = days_until_next_workday
    return group[["dest_country", "scheduled_date", "days_until_next_day_off", "days_until_next_workday"]]


# --- Configuration et exécution du script ---
if __name__ == "__main__":
    load_dotenv()

    DATA_FOLDER = Path("data/")
    DATA_FOLDER.mkdir(exist_ok=True)

    WEATHER_FOLDER = DATA_FOLDER / "weather"
    WEATHER_FOLDER.mkdir(exist_ok=True)

    HOLIDAYS_FOLDER = DATA_FOLDER / "holidays"
    HOLIDAYS_FOLDER.mkdir(exist_ok=True)

    ORIGINAL_DATASET_FILE = DATA_FOLDER / "original_dataset.csv"
    MAIN_DATASET_FILE = DATA_FOLDER / "main_dataset.csv"
    WEATHER_FILE = WEATHER_FOLDER / "weather.csv"

    # if input("This operation is about to delete and redownload the entire dataset. Proceed? (Y/N): ").lower() != "y":
    #     quit()

    #
    load_main_dataset()
    #
    merge_datasets()
