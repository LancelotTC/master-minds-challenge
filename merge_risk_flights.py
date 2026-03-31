import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)


def add_risk_scores_to_dataset(
    flight_df: pd.DataFrame, iata_col: str, date_col: str, risk_val_col: str = "risk_score"
) -> pd.DataFrame:
    """
    Enriches flight_df with a risk score from a parquet file.
    """
    # airports_path = os.environ["PQT_CLEAN_AIRPORTS"]
    # risk_path = os.environ["PQT_SMOOTHED_RISK"]

    airports_path = "data/clean_airports.parquet"
    risk_path = "data/smoothed_risk_events.parquet"

    if not airports_path or not risk_path:
        return flight_df.assign(risk_score=0.0)

    # 1. Load Reference Data
    # Only load necessary columns to keep it fast
    df_airports = pd.read_parquet(airports_path, columns=["iata_code", "iso_country"])
    df_risk = pd.read_parquet(risk_path, columns=["iso_country", "Date", risk_val_col])

    # 2. Data Sanitization (The "Bulletproof" part)
    # Standardize IATA
    df_airports["iata_code"] = df_airports["iata_code"].str.upper().str.strip()

    # Standardize Dates to just the date (YYYY-MM-DD) to avoid time mismatches
    df_risk["Date"] = pd.to_datetime(df_risk["Date"]).dt.date

    # Prepare the flight dataset copy
    res_df = flight_df.copy()
    res_df[iata_col] = res_df[iata_col].str.upper().str.strip()
    res_df[date_col] = pd.to_datetime(res_df[date_col]).dt.date

    # 3. Merge Chain
    # Map IATA -> Country
    res_df = res_df.merge(df_airports, left_on=iata_col, right_on="iata_code", how="left")

    # Map (Country + Date) -> Risk Value
    res_df = res_df.merge(df_risk, left_on=["iso_country", date_col], right_on=["iso_country", "Date"], how="left")

    # 4. Final Cleanup
    # Ensure the result column is always named 'risk_score' for consistency
    if risk_val_col != "risk_score" and risk_val_col in res_df.columns:
        res_df.rename(columns={risk_val_col: "risk_score"}, inplace=True)

    res_df["risk_score"] = res_df["risk_score"].fillna(0.0)

    # Drop intermediate join columns
    drop_cols = ["iata_code", "iso_country", "Date"]
    res_df.drop(columns=[c for c in drop_cols if c in res_df.columns and c not in flight_df.columns], inplace=True)

    return res_df


if __name__ == "__main__":
    flights_df = pd.read_csv("data/mouvements_aero_insa.csv")
    df = add_risk_scores_to_dataset(flights_df, "AirportPrevious", "LTScheduledTime", "SmoothedRisk")
    high_risk_flights = df[df["risk_score"] > 0]
    print(high_risk_flights[["AirportPrevious"]])
