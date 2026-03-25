import requests
import os, pandas as pd
from pathlib import Path
from utils import ProgressBar
from dotenv import load_dotenv
from typing import Iterator
from google.cloud import bigquery
from google.cloud.bigquery.table import Row


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
    service_account_key_path = os.getenv("YOUR_SERVICE_ACCOUNT_KEY_PATH")

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

    # 5. Merge them!
    df_merged = pd.merge(main_dataset, weather_dataset, left_on="Temp_Date_Match", right_on="time", how="left")

    # 6. Clean up (drop the temporary column and the redundant 'time' column)
    df_merged = df_merged.drop(columns=["Temp_Date_Match", "time"])

    # 7. Save it back to CSV
    df_merged.to_csv(MAIN_DATASET_FILE, index=False)

    print(f"Success! Saved {len(weather_dataset)} days of continuous weather data.")


def load_holiday_data():

    os.system("pip install jours_feries_france vacances_scolaires_france")

    from jours_feries_france import JoursFeries
    from vacances_scolaires_france import SchoolHolidayDates

    # =========================
    # 1. Download data
    # =========================

    flights = pd.read_csv(MAIN_DATASET_FILE)
    airports = pd.read_csv(HOLIDAYS_FOLDER / "upply-airports.csv", sep=";")
    holidays = pd.read_csv(HOLIDAYS_FOLDER / "openholidays_2023_2027.csv")

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

    airports = airports.rename(columns={"code": "AirportPrevious", "country_code": "dest_country"})

    airports = airports[["AirportPrevious", "dest_country"]].drop_duplicates()

    flights = flights.merge(airports, on="AirportPrevious", how="left")

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

    holidays = holidays.rename(columns={"country_iso_code": "country_code", "holiday_type": "type"})

    holidays["type"] = holidays["type"].replace({"public_holiday": "public", "school_holiday": "school"})

    holidays["start_date"] = pd.to_datetime(holidays["start_date"], errors="coerce")
    holidays["end_date"] = pd.to_datetime(holidays["end_date"], errors="coerce")

    holidays = holidays[["country_code", "type", "start_date", "end_date"]].dropna(
        subset=["country_code", "type", "start_date", "end_date"]
    )

    holidays["date"] = holidays.apply(lambda row: pd.date_range(row["start_date"], row["end_date"], freq="D"), axis=1)

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

    os.system("pip uninstall jours_feries_france vacances_scolaires_france -y")


def merge_datasets():
    load_weather_data()
    load_holiday_data()


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

    if input("This operation is about to delete and redownload the entire dataset. Proceed? (Y/N): ").lower() != "y":
        quit()

    #
    # MAIN_DATASET_FILE.unlink(missing_ok=True)
    # load_main_dataset()
    #
    # WEATHER_FILE.unlink(missing_ok=True)
    # load_weather_data()
    #
    merge_datasets()
