import os
from dotenv import load_dotenv
from utils.download_conflicts import (download_countryinfo_to_parquet, download_file, 
                                      download_gdelt_to_parquet, 
                                      download_ourairports_dataset)
from utils.transform_conflicts import (enrich_events_with_iso,
                             process_max_risk_events, create_risk_mapping
                             )
from utils.clean_conflicts import smooth_daily_risk
from utils.logger import Logger

# Load dotenv with override
load_dotenv(override=True)

# All env
PROJECT_ID = os.environ["PROJECT_ID"]
# GDELT env
GDELT_COLD_START_DATE = os.environ['GDELT_COLD_START_DATE']
PQT_GDELT = os.environ["PQT_GDELT"]
TRACK_GDELT = os.environ["TRACK_GDELT"]
CAMEO_EVENT=os.environ["CAMEO_EVENT"]
CAMEO_GOLDSTEIN = os.environ["CAMEO_GOLDSTEIN"]
TXT_CAMEO_GOLDSTEIN=os.environ['TXT_CAMEO_GOLDSTEIN']
TXT_CAMEO_EVENT=os.environ['TXT_CAMEO_EVENT']

# Smoothing
RISK_1_COOLDOWN = int(os.environ['RISK_1_COOLDOWN'])
RISK_2_COOLDOWN = int(os.environ['RISK_2_COOLDOWN'])
RISK_3_COOLDOWN = int(os.environ['RISK_3_COOLDOWN'])


# Country env
TRACK_COUNTRYINFO = os.environ["TRACK_COUNTRYINFO"]

# Airports env
API_AIRPORTS = os.environ["API_AIRPORTS"]
DATA_AIRPORTS = os.environ["DATA_AIRPORTS"]
TRACK_AIRPORTS = os.environ["TRACK_AIRPORTS"]
CSV_AIRPORTS = os.environ["CSV_AIRPORTS"]

# Parquet files
PQT_AIRPORTS = os.environ["PQT_AIRPORTS"]
PQT_COUNTRYINFO = os.environ["PQT_COUNTRYINFO"] 
PQT_CLEAN_AIRPORTS = os.environ["PQT_CLEAN_AIRPORTS"]
PQT_DAILY_RISK = os.environ["PQT_DAILY_RISK"]
PQT_CAMEO_MAP = os.environ['PQT_CAMEO_MAP']
PQT_SMOOTHED_RISK = os.environ["PQT_SMOOTHED_RISK"]

# Logging env variables (paths)
LOG_TF_EVENTS_ISO = os.environ["LOG_TF_EVENTS_ISO"]
LOG_DL_CONFLICT = os.environ["LOG_DL_CONFLICT"]
LOG_DL_AIRPORTS = os.environ["LOG_DL_AIRPORTS"]
LOG_DL_COUNTRYINFO = os.environ["LOG_DL_COUNTRYINFO"]
LOG_CONFLICT_PIPELINE = os.environ["LOG_CONFLICT_PIPELINE"]


# Set loggers

DL_GDELT_LOGGER = Logger(LOG_DL_CONFLICT)
DL_COUNTRY_LOGGER = Logger(LOG_DL_COUNTRYINFO)
DL_AIRPORT_LOGGER = Logger(LOG_DL_AIRPORTS)

TF_ISO_LOGGER = Logger(LOG_TF_EVENTS_ISO)

# Set pipelines
def download_pipeline() :

    # Download CAMEO.eventcodes
    download_file(CAMEO_EVENT)

    # Download CAMEO.goldsteinscale
    download_file(CAMEO_GOLDSTEIN)
    
    # Download GDELT
    download_gdelt_to_parquet(
        project_id=PROJECT_ID,
        output_dir=PQT_GDELT,
        start_date=GDELT_COLD_START_DATE,
        config_file=TRACK_GDELT,
        logger=DL_GDELT_LOGGER
    )

    # Download country info
    download_countryinfo_to_parquet(
        project_id=PROJECT_ID,
        output_path=PQT_COUNTRYINFO,
        config_path=TRACK_COUNTRYINFO,
        logger=DL_COUNTRY_LOGGER
    )

    # Download airports
    download_ourairports_dataset(
        api_url=API_AIRPORTS,
        data_url=DATA_AIRPORTS,
        tracking_file=TRACK_AIRPORTS,
        csv_file=CSV_AIRPORTS,
        parquet_file=PQT_AIRPORTS,
        logger=DL_AIRPORT_LOGGER
    )


def transform_pipeline() :

    create_risk_mapping(
        event_file=TXT_CAMEO_EVENT,
        goldstein_file=TXT_CAMEO_GOLDSTEIN,
        output_file=PQT_CAMEO_MAP
    )
    
    # Clean gdelt to daily risk
    process_max_risk_events(
        gdelt_file=PQT_GDELT,
        cameo_file=PQT_CAMEO_MAP,
        output_file=PQT_DAILY_RISK
    )

    enrich_events_with_iso(
        daily_risk_file=PQT_DAILY_RISK,
        country_info_file=PQT_COUNTRYINFO,
        logger=TF_ISO_LOGGER
    )
    

def clean_pipeline():
    # This is where the score diffusion runs
    smooth_daily_risk(
        file_path=PQT_DAILY_RISK,
        output_path=PQT_SMOOTHED_RISK,
        cooldown_l3=RISK_3_COOLDOWN,
        cooldown_l2=RISK_2_COOLDOWN,
        cooldown_l1=RISK_1_COOLDOWN
    )

if __name__ == '__main__' :
    logger = Logger(LOG_CONFLICT_PIPELINE)

    logger.log("Starting download pipeline")
    download_pipeline()
    logger.log("Download pipeline ended")

    logger.log("Starting transformation pipeline")
    transform_pipeline()
    logger.log("Transformation pipeline ended")

    logger.log("Starting cleaning pipeline")
    clean_pipeline()
    logger.log("Cleaning pipeline ended")

