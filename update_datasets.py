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

    progress_bar = ProgressBar(rows.total_rows, update_every=rows.total_rows // 100)
    progress_bar.start()

    dataset_rows = []

    for row in rows:
        dataset_rows.append(dict(row.items()))
        progress_bar.increment()

    progress_bar.finish()

    pd.DataFrame(dataset_rows).to_csv(MAIN_DATASET_FILE, index=False)
    print(f"\tMain dataset saved to {MAIN_DATASET_FILE}")


def load_weather_data():

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
    df_combined = pd.concat([df_archive, df_forecast])

    # 5. The Magic Step: Drop duplicates
    # We drop rows with the same date ('time').
    # keep='first' means if a date exists in both datasets, we keep the Archive version,
    # which is preferred because historical data is finalized and more accurate than past forecasts.
    df_combined = df_combined.drop_duplicates(subset=["time"], keep="first")

    # 6. Sort chronologically just to be perfectly safe, and reset the row numbers
    df_combined = df_combined.sort_values("time").reset_index(drop=True)

    # 7. Export to your final, continuous CSV
    df_combined.to_csv(WEATHER_FILE, index=False)

    print(f"Success! Saved {len(df_combined)} days of continuous weather data.")


def merge_datasets():
    # 1. Load both datasets
    main_dataset = pd.read_csv(MAIN_DATASET_FILE)
    weather_dataset = pd.read_csv(WEATHER_FILE)

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
    df_merged.to_csv(DATA_FOLDER / "merged_dataset.csv", index=False)


# --- Configuration et exécution du script ---
if __name__ == "__main__":
    load_dotenv()

    DATA_FOLDER = Path("data/")
    DATA_FOLDER.mkdir(exist_ok=True)

    WEATHER_FOLDER = DATA_FOLDER / "weather"
    WEATHER_FOLDER.mkdir(exist_ok=True)

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
