import os
import urllib.request
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import requests
import polars as pl
from google.cloud import bigquery
from google.cloud import bigquery_storage
from urllib.parse import urlparse
from dotenv import load_dotenv
# Import very simple handmade logger to log stuff in txt files
from utils.logger import Logger

# THis par is to silence bigquery's warnings
import warnings
import logging

warnings.filterwarnings("ignore", module="google.cloud")
warnings.filterwarnings("ignore", module="google.auth")

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

logging.getLogger("google.cloud").setLevel(logging.ERROR)
logging.getLogger("google.auth").setLevel(logging.ERROR)
logging.getLogger("google.api_core").setLevel(logging.ERROR)



def download_gdelt_to_parquet(project_id, output_dir, start_date, config_file, logger : Logger):
    
    if os.path.exists(config_file) and os.path.getsize(config_file) > 0:
        with open(config_file, "r") as f:
            saved_date = f.read().strip()
            if saved_date.isdigit():
                start_date = int(saved_date)
                logger.log(f"Config found. Resuming from date: {start_date}")

    level_1_codes = ["181", "195", "204", "2041", "2042"]
    level_2_codes = [
        "152", "154", "180", "182", "1821", "1822", "1823", "183", "1831", "1832", "1833", 
        "184", "185", "186", "190", "191", "192", "193", "194", "196", "200", "201", "202", "203"
    ]
    level_3_codes = [
        "130", "131", "1311", "1312", "1313", "132", "1321", "1322", "1323", "1324", "133", "134", 
        "135", "136", "137", "138", "1381", "1382", "1383", "1384", "1385", "139", "140", "141", 
        "1411", "1412", "1413", "1414", "142", "1421", "1422", "1423", "1424", "143", "1431", "1432", 
        "1433", "1434", "144", "1441", "1442", "1443", "1444", "145", "1451", "1452", "1453", "1454",
        "150", "151", "153", "160", "161", "162", "1621", "1622", "1623", "163", "164", "165", "166", 
        "1661", "1662", "1663", "170", "171", "1711", "1712", "172", "1721", "1722", "1723", "1724", 
        "173", "174", "175"
    ]

    all_codes = level_1_codes + level_2_codes + level_3_codes
    codes_sql_string = ", ".join([f"'{code}'" for code in all_codes])

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    part_date = str(start_date)
    part_date_formatted = f"{part_date[0:4]}-{part_date[4:6]}-{part_date[6:8]}"

    service_account_key_path = os.getenv("SERVICE_ACCOUNT_KEY_PATH") or os.getenv("YOUR_SERVICE_ACCOUNT_KEY_PATH")
    if service_account_key_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_key_path
    
    client = bigquery.Client(project=project_id)
    bqstorage_client = bigquery_storage.BigQueryReadClient(credentials=client._credentials)
    writer = None
    current_date = None

    try :
        query = f"""
            SELECT
                GLOBALEVENTID, SQLDATE, MONTHYEAR,
                Actor1Code, Actor1Name, Actor1CountryCode, Actor1Type1Code, Actor1Type2Code, Actor1Type3Code,
                Actor2Code, Actor2Name, Actor2CountryCode, Actor2Type1Code, Actor2Type2Code, Actor2Type3Code,
                EventCode, EventBaseCode, EventRootCode, QuadClass, NumMentions, AvgTone, GoldSteinScale,
                Actor1Geo_Type, Actor1Geo_FullName, Actor1Geo_CountryCode,
                Actor2Geo_Type, Actor2Geo_FullName, Actor2Geo_CountryCode,
                ActionGeo_Type, ActionGeo_FullName, ActionGeo_CountryCode,
                DATEADDED
            FROM 
                `gdelt-bq.gdeltv2.events_partitioned`
            WHERE 
                _PARTITIONDATE >= '{part_date_formatted}'
                AND SQLDATE >= {start_date}
                AND EventCode IN ({codes_sql_string})
            ORDER BY DATEADDED
        """

        logger.log(f"Running query for data from {start_date} to today...")
        job = client.query(query)
        result = job.result()

        logger.log(f"Data processed: {job.total_bytes_processed / (1024**2):.2f} MB")
        logger.log(f"Data billed: {job.total_bytes_billed / (1024**2):.2f} MB")
        logger.log("Query finished. Streaming results via high-speed Storage API...")

        row_count = 0

        # Batch logic
        for batch in result.to_arrow_iterable(bqstorage_client=bqstorage_client): 
            day_array = pc.divide(batch.column("DATEADDED"), 1000000).cast(pa.int64())
            dates = day_array.to_pylist()
            
            batch_len = len(dates)
            start_idx = 0
            
            while start_idx < batch_len:
                batch_date = dates[start_idx]
                
                # Find the index where the date changes within the current batch
                end_idx = start_idx + 1
                while end_idx < batch_len and dates[end_idx] == batch_date:
                    end_idx += 1
                    
                # Slice out only the rows belonging to this specific date
                chunk = batch.slice(start_idx, end_idx - start_idx)
                
                # If we hit a new date, close the old file and start a new one
                if current_date != batch_date:
                    if writer is not None:
                        writer.close()
                        # Update config file ONLY when a day is fully written and closed
                        with open(config_file, "w") as f:
                            f.write(str(current_date))
                    
                    current_date = batch_date
                    file_path = os.path.join(output_dir, f"gdelt_{current_date}.parquet")
                    writer = pq.ParquetWriter(file_path, batch.schema, compression='snappy')
                
                # Write the sliced chunk
                writer.write_batch(chunk)
                row_count += chunk.num_rows
                start_idx = end_idx

            print(f"Written {row_count:,} rows so far...", end='\r')

    except Exception as e:
        logger.log(f"[ERROR] : {str(e)}")

    writer
    if writer:
        writer.close()
        # Save the very last date processed
        with open(config_file, "w") as f:
            f.write(str(current_date))
            
        logger.log(f"Success! Downloaded {row_count:,} rows to {output_dir}")
        return True
    else:
        logger.log("No data found for the given query.")
        return False


def download_countryinfo_to_parquet(project_id, output_path, config_path, logger : Logger):
    
    # Ensure directories exist for both the parquet and config files
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    service_account_key_path = os.getenv("SERVICE_ACCOUNT_KEY_PATH") or os.getenv("YOUR_SERVICE_ACCOUNT_KEY_PATH")
    if service_account_key_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_key_path
    # Initialize clients
    client = bigquery.Client(project=project_id)
    bqstorage_client = bigquery_storage.BigQueryReadClient(credentials=client._credentials)

    table_id = "gdelt-bq.extra.countryinfo2"
    
    # Check the table's last modified date
    logger.log(f"Checking metadata for {table_id}...")
    table = client.get_table(table_id)
    
    # The .modified attribute returns a datetime object. We convert it to ISO format for easy text storage.
    last_modified_bq = table.modified.isoformat()
    
    # Compare with local config file
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            last_modified_local = f.read().strip()
            
        if last_modified_local == last_modified_bq:
            logger.log(f"Table hasn't changed since the last download ({last_modified_local}).")
            logger.log("Skipping download.")
            return  False
    else:
        logger.log("No local config found. Proceeding with initial download.")

    # If dates don't match (or file is missing), run the query
    logger.log("New data detected! Running query...")
    writer = None

    try:
        # Since countryinfo2 is a reference table, we just select everything.
        query = f"""
            SELECT *
            FROM `{table_id}`
        """
        
        job = client.query(query)
        result = job.result()

        bytes_processed = job.total_bytes_processed
        bytes_billed = job.total_bytes_billed

        logger.log(f"Data processed: {bytes_processed / (1024**2):.2f} MB")
        logger.log(f"Data billed: {bytes_billed / (1024**2):.2f} MB")
        
        logger.log("Query finished. Streaming results via high-speed Storage API...")

        row_count = 0

        # Download and write to Parquet
        for batch in result.to_arrow_iterable(bqstorage_client=bqstorage_client): 
            if writer is None:
                writer = pq.ParquetWriter(output_path, batch.schema, compression='snappy')
            writer.write_batch(batch)
            row_count += batch.num_rows
            logger.log(f"Written {row_count:,} rows so far...", end='\r')
    except Exception as e:
        logger.log(f"[ERROR] : {str(e)}")

    # Save the new modified date to the config file upon success
    if writer:
        writer.close()
        logger.log(f"Success! Downloaded {row_count:,} rows to {output_path}")
        
        with open(config_path, "w") as f:
            f.write(last_modified_bq)
        logger.log(f"Updated config file with new modified date: {last_modified_bq}")
        return True
    else:
        logger.log("No data found for the given query.")
    return False



def download_ourairports_dataset(api_url, data_url, tracking_file, csv_file, parquet_file, logger : Logger):
    """Download / update data from OurAirports.

    Args:
        api_url (str): URL of github API for the project
        data_url (str): URL of the data file in the github project
        tracking_file (str): Path to the configuration file of this download
        csv_file (str): Path to the downloaded .csv file
        parquet_file (str): Path to the .parquet file that will be saved

    Returns:
        bool: True if downloaded or updated, False otherwise
    """

    logger.log("Checking for OurAirports updates...")

    # Get the latest commit SHA from GitHub
    try:
        api_response = requests.get(api_url, timeout=10)
        api_response.raise_for_status()
        latest_commit_sha = api_response.json()['sha']
    except requests.exceptions.RequestException as e:
        logger.log(f"Failed to check GitHub for updates: {e}")
        return False

    # Read the local commit SHA
    local_commit_sha = None
    if os.path.exists(tracking_file):
        with open(tracking_file, 'r') as f:
            local_commit_sha = f.read().strip()

    # Compare and check if the Parquet file actually exists
    if latest_commit_sha == local_commit_sha and os.path.exists(parquet_file):
        logger.log("Your dataset is already up to date! No download needed.")
        return False

    logger.log("New update found (or missing file). Downloading the latest CSV...")
    
    # Download the CSV in chunks
    try:
        # Ensure directories exist
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        os.makedirs(os.path.dirname(tracking_file), exist_ok=True)
        
        data_response = requests.get(data_url, stream=True, timeout=20)
        data_response.raise_for_status()
        
        with open(csv_file, 'wb') as f:
            for chunk in data_response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        logger.log("CSV downloaded. Starting Polars conversion to Parquet...")
        
        # Convert using Polars Lazy API (Memory-Efficient & Crash-Proof)
        # infer_schema_length=0 forces everything to a String type to avoid mixed-type panics
        (
            pl.scan_csv(csv_file, infer_schema_length=0)
            .drop_nulls(subset=['iata_code'])
            .sink_parquet(parquet_file, compression='snappy')
        )
                
        logger.log("Conversion complete! Cleaning up the raw CSV...")
        
        # Delete the original CSV file to save space
        if os.path.exists(csv_file):
            os.remove(csv_file)
            
        # Update tracking file ONLY after a successful download and conversion
        with open(tracking_file, 'w') as f:
            f.write(latest_commit_sha)
            
        logger.log(f"All done! Your updated dataset is ready at: {parquet_file}")

    except Exception as e:
        logger.log(f"An error occurred during the process: {e}")
        return False
    
    return True

def download_file(url, target_folder="data"):
    os.makedirs(target_folder, exist_ok=True)
    
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    
    if not filename:
        filename = "downloaded_file"
    
    save_path = os.path.join(target_folder, filename)

    if os.path.exists(save_path):
        return
    
    # No need for a logger since we will download the CAMEO files only once    
    print(f"Downloading {filename}...")
    
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"Success! Saved to: {save_path}")
    except Exception as e:
        print(f"Failed to download. Error: {e}")


if __name__ == "__main__":
    load_dotenv(override=True)

    # All env
    PROJECT_ID = os.environ["PROJECT_ID"]
    # GDELT env
    PQT_GDELT = os.environ["PQT_GDELT"]
    TRACK_GDELT = os.environ["TRACK_GDELT"]
    # Country env
    PQT_COUNTRYINFO = os.environ["PQT_COUNTRYINFO"]
    TRACK_COUNTRYINFO = os.environ["TRACK_COUNTRYINFO"]

    # Airports env
    API_AIRPORTS = os.environ["API_AIRPORTS"]
    DATA_AIRPORTS = os.environ["DATA_AIRPORTS"]
    TRACK_AIRPORTS = os.environ["TRACK_AIRPORTS"]
    CSV_AIRPORTS = os.environ["CSV_AIRPORTS"]
    PQT_AIRPORTS = os.environ["PQT_AIRPORTS"]

    download_gdelt_to_parquet(
        project_id=PROJECT_ID, 
        output_dir=PQT_GDELT,
        config_file=TRACK_GDELT,
        start_date=20221001
        )

    # download_pipeline()