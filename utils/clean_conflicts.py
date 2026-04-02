import polars as pl

def smooth_daily_risk(file_path, output_path, cooldown_l3=30, cooldown_l2=14, cooldown_l1=2):
    """
    Applies Cascading Tiered Cooldowns (Risk Decay) to daily aviation risk using Polars.
    Risk steps down gradually (e.g., L3 -> L2 -> L1 -> 0) instead of dropping instantly.
    """
    # 1. Load Data
    df = pl.read_parquet(file_path)

    # 2. Format Dates and drop null countries
    df = df.with_columns(
        pl.col('SQLDATE').cast(pl.String).str.strptime(pl.Date, format='%Y%m%d').alias('Date')
    ).drop_nulls(subset=['iso_country'])
    
    # Extract unique country mapping
    country_map = (
        df.select(['iso_country', 'iso3_country'])
        .drop_nulls()
        .unique(subset=['iso_country'])
    )
    
    # 3. Get the absolute maximum risk per country per day
    daily_max = df.group_by(['iso_country', 'Date']).agg(
        pl.col('RiskLevel').max()
    )
    
    # 4. Generate a full date grid to fill in missing days
    min_date = daily_max.select(pl.col('Date').min()).item()
    max_date = daily_max.select(pl.col('Date').max()).item()
    
    all_dates = pl.DataFrame({
        'Date': pl.date_range(min_date, max_date, "1d", eager=True)
    })
    
    countries = daily_max.select('iso_country').unique()
    grid = countries.join(all_dates, how="cross")
    
    # 5. Join original max data onto the complete timeline
    daily_full = (
        grid.join(daily_max, on=['iso_country', 'Date'], how='left')
        .with_columns(pl.col('RiskLevel').fill_null(0))
        .join(country_map, on='iso_country', how='left')
        .sort(['iso_country', 'Date'])
    )
    
    # 6. Create binary indicator flags
    daily_full = daily_full.with_columns(
        Is_L3 = (pl.col('RiskLevel') == 3).cast(pl.Int32),
        Is_L2 = (pl.col('RiskLevel') == 2).cast(pl.Int32),
        Is_L1 = (pl.col('RiskLevel') == 1).cast(pl.Int32),
    )
    
    # Helper to apply the rolling maximum
    def apply_cooldown(col_name, days):
        if days <= 0:
            return pl.col(col_name)
        return pl.col(col_name).rolling_max(window_size=days + 1, min_periods=1)

    # 7. Calculate cascading window cooldowns per country
    daily_full = daily_full.with_columns(
        L3_base    = apply_cooldown('Is_L3', cooldown_l3).over('iso_country'),
        L2_base    = apply_cooldown('Is_L2', cooldown_l2).over('iso_country'),
        L1_base    = apply_cooldown('Is_L1', cooldown_l1).over('iso_country'),
        L2_from_L3 = apply_cooldown('Is_L3', cooldown_l3 + cooldown_l2).over('iso_country'),
        L1_from_L3 = apply_cooldown('Is_L3', cooldown_l3 + cooldown_l2 + cooldown_l1).over('iso_country'),
        L1_from_L2 = apply_cooldown('Is_L2', cooldown_l2 + cooldown_l1).over('iso_country'),
    )
    
    # 8. Apply priority logic for the smoothed risk level
    daily_full = daily_full.with_columns(
        SmoothedRisk = pl.when(pl.col('L3_base') == 1).then(3)
        .when((pl.col('L2_base') == 1) | (pl.col('L2_from_L3') == 1)).then(2)
        .when((pl.col('L1_base') == 1) | (pl.col('L1_from_L2') == 1) | (pl.col('L1_from_L3') == 1)).then(1)
        .otherwise(0)
    )
    
    # 9. Track run-length logic for consecutive active/safe days
    daily_full = daily_full.with_columns(
        is_risk = pl.col('SmoothedRisk') > 0
    )
    
    daily_full = daily_full.with_columns(
        state_changed = (
            pl.col('is_risk') != pl.col('is_risk').shift(1).over('iso_country')
        ).fill_null(True)  # The first row of a group triggers a state change
    )
    
    daily_full = daily_full.with_columns(
        block_id = pl.col('state_changed').cast(pl.Int32).cum_sum().over('iso_country')
    )
    
    daily_full = daily_full.with_columns(
        consecutive_days = pl.col('Date').rank(method='ordinal').over(['iso_country', 'block_id'])
    )
    
    daily_full = daily_full.with_columns(
        nbDaysSinceRiskStarted = pl.when(pl.col('is_risk')).then(pl.col('consecutive_days')).otherwise(-1),
        nbDaysSinceSafeStarted = pl.when(~pl.col('is_risk')).then(pl.col('consecutive_days')).otherwise(-1)
    )

    # 10. Filter down to final deliverables
    final_df = daily_full.select([
        'iso_country', 
        'iso3_country', 
        pl.col('Date').cast(pl.Datetime), 
        'RiskLevel', 
        'SmoothedRisk',
        'nbDaysSinceRiskStarted',
        'nbDaysSinceSafeStarted'
    ])
    
    # Write back to parquet
    final_df.write_parquet(output_path, compression='snappy')
    
    return final_df

if __name__ == "__main__":
    processed_df = smooth_daily_risk(
        'data/daily_max_risk_events.parquet', 
        'data/smoothed_risk_events_pl.parquet', 
        cooldown_l3=30, 
        cooldown_l2=14,  
        cooldown_l1=2   
    )