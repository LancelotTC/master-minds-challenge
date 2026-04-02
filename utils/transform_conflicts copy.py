import os
import polars as pl
from utils.logger import Logger
from utils.file_manager import get_files_since_date

HIGH_MEDIA_BIAS = {
    'US', 'UK', 'AS', 'CA', 'NZ', 'EI', 'FR', 'GM', 'CH', 'JA', 
    'SP', 'IT', 'BR', 'MX', 'IN', 'AR', 'NL', 'BE', 'SW', 'NO', 'DA', 'FI', 'KS'
}
INSTITUTIONAL_ACTORS = ['BUS', 'MED', 'EDU', 'HLH', 'ENV', 'LAB', 'NGO', 'ELI', 'AGR']
MILITANT_GROUPS = ['AAM', 'ALQ', 'ANO', 'ASL', 'DFL', 'FIS', 'GIA', 'GSP', 'HEZ', 'HMS', 'ISJ', 'PLF', 'PLO', 'PLS', 'PMD', 'TAL']
WAR_ACTORS = ['MIL', 'REB', 'INS', 'SEP', 'RAD', 'UIS'] 

def create_risk_mapping(event_file, goldstein_file, output_file):

    df_goldstein = pl.read_csv(goldstein_file, separator='\t', schema_overrides={'CAMEOEVENTCODE': pl.String})
    df_codes = pl.read_csv(event_file, separator='\t', schema_overrides={'CAMEOEVENTCODE': pl.String})
    
    # Left join on CAMEOEVENTCODE
    df = df_goldstein.join(df_codes, on='CAMEOEVENTCODE', how='left')

    # LEVEL 3: DO NOT FLY (Immediate, severe threat to airspace/airports)
    risk_3_codes = ["195", "200", "204", "2041", "2042"]

    # LEVEL 2: DANGER EXISTS (Active ground war, localized intense conflict, terrorism)
    risk_2_codes = [
        "181", "1823",
        "183", "1831", "1832", "1833",
        "190", "191", "192", "193", "194",
        "201", "202", "203"
    ]

    # LEVEL 1: CAUTION (Civil unrest, protests, curfews, state of emergency)
    risk_1_codes = [
        "14", "140", "141", "1411", "1412", "1413", "1414", "142", "1421", "1422", "1423", "1424", 
        "143", "1431", "1432", "1433", "1434", "144", "1441", "1442", "1443", "1444", 
        "145", "1451", "1452", "1453", "1454",
        "17", "170", "171", "1711", "1712", "172", "1721", "1722", "1723", "1724", "173", "174", "175"
    ]
    
    # Process the dataframe using idiomatic Polars method chaining
    df_final = (
        df.with_columns(
            pl.when(pl.col('CAMEOEVENTCODE').is_in(risk_3_codes)).then(3)
            .when(pl.col('CAMEOEVENTCODE').is_in(risk_2_codes)).then(2)
            .when(pl.col('CAMEOEVENTCODE').is_in(risk_1_codes)).then(1)
            .otherwise(0).alias('Level')
        )
        .with_columns(
            pl.format("Level {}", pl.col('Level')).alias('Risk_Level')
        )
        .rename({
            'CAMEOEVENTCODE': 'CAMEO_CODE', 
            'GOLDSTEINSCALE': 'GoldenSteinScale'
        })
        .select([
            'CAMEO_CODE', 'GoldenSteinScale', 'Risk_Level', 'EVENTDESCRIPTION', 'Level'
        ])
    )

    df_final.write_parquet(output_file, compression='snappy')
    

def enrich_events_with_iso(daily_risk_file, country_info_file, logger : Logger):
    
    # 1. Load the datasets
    events = pl.read_parquet(daily_risk_file).with_columns(
        pl.col("ActionGeo_CountryCode").cast(pl.Utf8)
    )
    fips_to_iso = pl.read_parquet(country_info_file)

    # 2. Prep the standard countryinfo data
    fips_to_iso = fips_to_iso.select([
        pl.col("fips").alias("ActionGeo_CountryCode"), 
        pl.col("iso").alias("iso_country"),
        pl.col("iso3").alias("iso3_country")
    ])

    # 3. Create a manual lookup table for the missing FIPS 10-4 codes
    missing_fips_df = pl.DataFrame({
        "ActionGeo_CountryCode": [
            "DQ", "GZ", "HQ", "IP", "JN", "JQ", "KQ", "LQ", 
            "MQ", "OC", "OS", "PF", "PG", "RB", "TE", "WQ"
        ],
        "iso_country": [
            "UM", "PS", "UM", "CP", "SJ", "UM", "UM", "UM", 
            "UM", "XZ", "XZ", "XX", "XX", "RS", "TF", "UM"
        ],
        "iso3_country": [
            "UMI", "PSE", "UMI", "CPT", "SJM", "UMI", "UMI", "UMI", 
            "UMI", "XZZ", "XZZ", "XXX", "XXX", "SRB", "ATF", "UMI"
        ]
    })

    # Concatenate the manual mappings to the bottom of the main lookup table
    fips_to_iso = pl.concat([fips_to_iso, missing_fips_df])

    # 4. Join the datasets
    events = events.join(
        fips_to_iso,
        on='ActionGeo_CountryCode',
        how='left'
    )

    # 5. Save the enriched dataset
    events.write_parquet(daily_risk_file, compression='snappy')
    logger.log(f"Successfully added ISO codes (including manual FIPS fixes) and saved to {daily_risk_file}")


# Risks events to daily risk functions

def clean_gdelt_source(lf: pl.LazyFrame) -> pl.LazyFrame:
    needed_cols = [
        "GLOBALEVENTID", "SQLDATE", "EventCode", "QuadClass", "NumMentions", "GoldSteinScale",
        "Actor1Code", "Actor2Code", "Actor1Type1Code", "Actor2Type1Code", "Actor1Type2Code", "Actor2Type2Code",
        "Actor1Geo_CountryCode", "Actor2Geo_CountryCode", "ActionGeo_CountryCode"
    ]
    
    return (
        lf.select(needed_cols)
        .with_columns([
            pl.col(c).cast(pl.Utf8).fill_null("") for c in [
                "Actor1Type1Code", "Actor2Type1Code", "Actor1Type2Code", "Actor2Type2Code", "Actor1Code", "Actor2Code", "ActionGeo_CountryCode"
            ]
        ])
        .with_columns([
            pl.col("NumMentions").cast(pl.Int32).fill_null(0),
            pl.col("GoldSteinScale").cast(pl.Float32).fill_null(0.0)
        ])
        .filter(
            (pl.col("ActionGeo_CountryCode") != "") &
            (pl.col("QuadClass").is_in([3, 4]))
        )
    )

def refine_airline_risk(lf: pl.LazyFrame) -> pl.LazyFrame:
    # 1. IDENTIFY LOCAL ACTORS
    actor1_is_local = pl.col("Actor1Geo_CountryCode") == pl.col("ActionGeo_CountryCode")
    actor2_is_local = pl.col("Actor2Geo_CountryCode") == pl.col("ActionGeo_CountryCode")
    has_local_actor = actor1_is_local | actor2_is_local

    is_institutional_dispute = (
        pl.col("Actor1Type1Code").is_in(INSTITUTIONAL_ACTORS) | pl.col("Actor1Type2Code").is_in(INSTITUTIONAL_ACTORS) |
        pl.col("Actor2Type1Code").is_in(INSTITUTIONAL_ACTORS) | pl.col("Actor2Type2Code").is_in(INSTITUTIONAL_ACTORS) |
        pl.col("Actor1Code").str.contains("(?i)BUS|MED|EDU|HLH|ENV|LAB|NGO|ELI|AGR") |
        pl.col("Actor2Code").str.contains("(?i)BUS|MED|EDU|HLH|ENV|LAB|NGO|ELI|AGR")
    )

    is_protest_code = pl.col("EventCode").str.starts_with("14")

    local_actor_is_state_without_type = (
        (actor1_is_local & (pl.col("Actor1Type1Code") == "") & (pl.col("Actor1Type2Code") == "") & (pl.col("Actor1Code") != "") & ~is_institutional_dispute) |
        (actor2_is_local & (pl.col("Actor2Type1Code") == "") & (pl.col("Actor2Type2Code") == "") & (pl.col("Actor2Code") != "") & ~is_institutional_dispute)
    )

    local_actor_is_kinetic = (
        (actor1_is_local & (pl.col("Actor1Type1Code").is_in(WAR_ACTORS) | pl.col("Actor1Type2Code").is_in(WAR_ACTORS) | pl.col("Actor1Code").is_in(MILITANT_GROUPS))) |
        (actor2_is_local & (pl.col("Actor2Type1Code").is_in(WAR_ACTORS) | pl.col("Actor2Type2Code").is_in(WAR_ACTORS) | pl.col("Actor2Code").is_in(MILITANT_GROUPS)))
    )

    is_major_combat = pl.col("EventCode").str.starts_with("19") | pl.col("EventCode").str.starts_with("20")

    # 2. DEFINING THE SHIELDS
    is_police_action = (
        pl.col("Actor1Type1Code").is_in(['COP', 'JUD']) | pl.col("Actor1Type2Code").is_in(['COP', 'JUD']) |
        pl.col("Actor2Type1Code").is_in(['COP', 'JUD']) | pl.col("Actor2Type2Code").is_in(['COP', 'JUD'])
    )
    
    is_domestic_crime = (
        pl.col("Actor1Type1Code").is_in(['CRM']) | pl.col("Actor1Type2Code").is_in(['CRM']) |
        pl.col("Actor2Type1Code").is_in(['CRM']) | pl.col("Actor2Type2Code").is_in(['CRM'])
    )

    has_armed_opposition = (
        pl.col("Actor1Type1Code").is_in(WAR_ACTORS) | pl.col("Actor1Type2Code").is_in(WAR_ACTORS) |
        pl.col("Actor2Type1Code").is_in(WAR_ACTORS) | pl.col("Actor2Type2Code").is_in(WAR_ACTORS) |
        pl.col("Actor1Code").is_in(MILITANT_GROUPS) | 
        pl.col("Actor2Code").is_in(MILITANT_GROUPS)
    )
    
    is_unopposed_domestic_military = (
        ((pl.col("Actor1Type1Code") == "MIL") | (pl.col("Actor1Type2Code") == "MIL") | 
         (pl.col("Actor2Type1Code") == "MIL") | (pl.col("Actor2Type2Code") == "MIL")) &
        ~has_armed_opposition &
        (
            (pl.col("Actor1Geo_CountryCode") == pl.col("Actor2Geo_CountryCode")) |
            (pl.col("Actor1Geo_CountryCode") == "") | (pl.col("Actor2Geo_CountryCode") == "")
        )
    )

    is_missing_actor1 = (pl.col("Actor1Code") == "") & (pl.col("Actor1Type1Code") == "") & (pl.col("Actor1Type2Code") == "")
    is_missing_actor2 = (pl.col("Actor2Code") == "") & (pl.col("Actor2Type1Code") == "") & (pl.col("Actor2Type2Code") == "")
    is_single_actor_event = is_missing_actor1 | is_missing_actor2
    
    has_militant_actor = pl.col("Actor1Code").is_in(MILITANT_GROUPS) | pl.col("Actor2Code").is_in(MILITANT_GROUPS)

    is_extreme_event = is_major_combat & (pl.col("GoldSteinScale") <= -9.0)

    is_severe_aviation_combat = (
        pl.col("EventCode").is_in(["194", "195"]) | 
        pl.col("EventCode").str.starts_with("20")
    )
    is_standard_ground_combat = pl.col("EventCode").is_in(["190", "191", "192", "193", "196"])
    
    # Combined for legacy checks
    is_major_combat = is_severe_aviation_combat | is_standard_ground_combat

    # 3. FIX: THE NEW MENTION THRESHOLD LOGIC
    mention_threshold = (
        pl.when(pl.col("Actor1Code").is_in(MILITANT_GROUPS)).then(10)
        
        # Priority 1: High Media Hubs always need a huge signal to break through noise.
        .when(pl.col("ActionGeo_CountryCode").is_in(list(HIGH_MEDIA_BIAS))).then(250)
        
        # Priority 2: State vs State (No MIL tag). If two countries are fighting but no military is identified,
        # it requires massive global coverage (150 mentions) to be real. This kills the Portugal false positives.
        .when(is_major_combat & local_actor_is_state_without_type).then(40)
        
        # Priority 3: Confirmed Kinetic. If the MIL or REB tag is actually there, trust it with a low threshold.
        # This saves your Ukraine row from earlier!
        .when(is_extreme_event & local_actor_is_kinetic).then(20)
        
        .otherwise(40)
    )

    # 4. FINAL RISK PIPELINE
    return lf.with_columns(
        RiskLevel = pl.when(pl.col("NumMentions") < mention_threshold).then(0)
        .when((pl.col("RiskLevel") >= 2) & ~has_local_actor).then(0)
        .when((pl.col("RiskLevel") >= 2) & is_single_actor_event & ~has_militant_actor).then(0)
        .when(is_domestic_crime).then(0)
        .when(is_institutional_dispute & ~has_armed_opposition).then(
            # Allow protests (14x) to stay at Risk 1, but force lawsuits/arrests/fights to 0
            pl.when(is_protest_code).then(pl.min_horizontal(pl.col("RiskLevel"), 1))
            .otherwise(0)
        )
        .when(is_police_action & ~has_armed_opposition).then(pl.min_horizontal(pl.col("RiskLevel"), 1))
        .when(
            is_unopposed_domestic_military & 
            pl.col("ActionGeo_CountryCode").is_in(list(HIGH_MEDIA_BIAS))
        ).then(pl.min_horizontal(pl.col("RiskLevel"), 1))
        .when(
            (pl.col("RiskLevel") >= 2) & 
            ~(local_actor_is_kinetic | (is_major_combat & local_actor_is_state_without_type))
        ).then(1) 
        .otherwise(pl.col("RiskLevel").fill_null(0))
    )

def process_max_risk_events(gdelt_file, cameo_file, output_file ,temp_file, config_file, logger : Logger):

    config_found = False
    if os.path.exists(config_file) and os.path.getsize(config_file) > 0:
        with open(config_file, "r") as f:
            saved_date = f.read().strip()
            if saved_date.isdigit():
                config_file=True
                start_date = int(saved_date)
                logger.log(f"Config found. Resuming from date: {start_date}")


    risk_map = pl.read_parquet(cameo_file).select([
        pl.col("CAMEO_CODE").alias("EventCode"), 
        pl.col("Level").alias("RiskLevel")
    ])

    # If config is set -> get all files that were not seen or not complete
    files_path = os.path.join(gdelt_file, '*.parquet')
    if config_file :
        files_path = get_files_since_date(gdelt_file, start_date)
        
    
    query = (
        pl.scan_parquet(files_path)
        .pipe(clean_gdelt_source)
        .join(risk_map.lazy(), on="EventCode", how="left")
        .pipe(refine_airline_risk)
        .group_by(["SQLDATE", "ActionGeo_CountryCode"])
        .agg(
            pl.all().sort_by(["RiskLevel", "NumMentions"], descending=True).first()
        )
    )

    if config_file :
        # Everyday run
        print("Executing Pipeline...")
        query.sink_parquet(temp_file)
        

        # Load temporary parquet & current daily parquet
        old_data = pl.scan_parquet(output_file)
        new_data = pl.scan_parquet(temp_file)
        temp_output = output_file + '.tmp'

        # Update the data with the new data
        (
            pl.concat([old_data, new_data], how="vertical_relaxed")
            .unique(subset=["SQLDATE", "ActionGeo_CountryCode"], keep="last")
            .sink_parquet(temp_output)
        )

        # Replace temp_output file with the real output_file & remove temp file
        os.replace(temp_output, output_file)
        os.remove(temp_file)

    else :
        # Cold start (once)
        print("Executing Pipeline...")
        query.sink_parquet(output_file)

if __name__ == "__main__":

    # Download / update data 
    # download_pipeline()

    # Transforms
    enrich_events_with_iso()