import os
from pathlib import Path
from dotenv import load_dotenv
from typing import cast, Iterator
from google.cloud import bigquery
from google.cloud.bigquery.table import Row
from tools.sqlite_manager import SqliteManager
from utils import ProgressBar


def load_main_dataset() -> Iterator[Row]:
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

    query = f"SELECT * FROM {table_ref} limit 10;"

    # Initialise le client BigQuery
    client = bigquery.Client(project=project_id)

    # Construit la référence complète de la table
    print(f"\tExécution de la requête sur BigQuery: {query}\n")

    query_job = client.query(query)

    print("\tGot response")

    # Récupère les résultats
    rows: Iterator[Row] = query_job.result()

    print(f"\tRequête terminée. {rows.total_rows} lignes récupérées.")

    if not os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, "w") as file:
            file.write("")

        with open(DATA_FOLDER / "creation_query.sql") as file:
            creation_query = file.read()

        with SqliteManager(DATABASE_FILE, True) as (db, connection):
            db.execute(creation_query)

    with SqliteManager(DATABASE_FILE, True) as (db, connection):

        progress_bar = ProgressBar(rows.total_rows)
        progress_bar.start()

        for row in rows:
            row = cast(Row, row)
            keys = row.keys()
            values = [str(value) if value is not None else "NULL" for value in row.values()]

            query = f"insert into {table_id} ({", ".join(keys)}) values ({", ".join("?" * len(keys))})"

            db.execute(query, values)

            progress_bar.increment()

        progress_bar.finish()


def load_weather_data():
    import requests

    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive?latitude=45.7589&longitude=4.8414&start_date=2023-01-01&end_date=2026-03-24&daily=weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,snowfall_sum,windspeed_10m_max&timezone=Europe%2FParis&format=csv"
    )

    with open("testing.csv", "w") as file:
        file.write("\n".join(response.content.decode("utf-8").split("\n")[3:]))


# --- Configuration et exécution du script ---
if __name__ == "__main__":
    load_dotenv()

    DATA_FOLDER = Path("data/")
    DATA_FOLDER.mkdir(exist_ok=True)

    DATABASE_FILE = DATA_FOLDER / "database.db"
    # ---
    load_main_dataset()
    # load_weather_data()
    ...
