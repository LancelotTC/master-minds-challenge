import pandas as pd
from jours_feries_france import JoursFeries
from vacances_scolaires_france import SchoolHolidayDates

# =========================
# 1. Download data
# =========================

flights = pd.read_csv("./csv_export/modified_dataset.csv")
airports = pd.read_csv("upply-airports.csv", sep=";")
holidays = pd.read_csv("openholidays_2023_2027.csv")

flights["LTScheduledDatetime"] = pd.to_datetime(
    flights["LTScheduledDatetime"],
    errors="coerce"
)

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

airports = airports.rename(columns={
    "code": "AirportPrevious",
    "country_code": "dest_country"
})

airports = airports[["AirportPrevious", "dest_country"]].drop_duplicates()

flights = flights.merge(
    airports,
    on="AirportPrevious",
    how="left"
)

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

flights["is_fr_school_holiday_zone_a"] = (
    flights["date_only"].isin(fr_school_holidays_zone_a).astype(int)
)

# =========================
# 6. DESTINATION : HOLIDAYS
# =========================

holidays = holidays.rename(columns={
    "country_iso_code": "country_code",
    "holiday_type": "type"
})

holidays["type"] = holidays["type"].replace({
    "public_holiday": "public",
    "school_holiday": "school"
})

holidays["start_date"] = pd.to_datetime(holidays["start_date"], errors="coerce")
holidays["end_date"] = pd.to_datetime(holidays["end_date"], errors="coerce")

holidays = holidays[[
    "country_code",
    "type",
    "start_date",
    "end_date"
]].dropna(subset=["country_code", "type", "start_date", "end_date"])

holidays["date"] = holidays.apply(
    lambda row: pd.date_range(row["start_date"], row["end_date"], freq="D"),
    axis=1
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

flights = flights.merge(
    dest_public,
    on=["dest_country", "date"],
    how="left"
)

flights = flights.merge(
    dest_school,
    on=["dest_country", "date"],
    how="left"
)

flights["is_dest_public_holiday"] = flights["is_dest_public_holiday"].fillna(0).astype(int)
flights["is_dest_school_holiday"] = flights["is_dest_school_holiday"].fillna(0).astype(int)

flights = flights.drop(columns=["date", "date_only"])

# =========================
# 7. EXPORT
# =========================

flights.to_csv("./csv_export/mouvements_aero_insa_with_holidays.csv", index=False, encoding="utf-8-sig")
