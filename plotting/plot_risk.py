import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_daily_risk_map(df, iso3_column='iso3_country', risk_column='SmoothedRisk', target_date=None):
    """
    Plots a world map colored by aviation risk level for a specific date,
    using 3-letter ISO country codes.
    """
    
    df['Date'] = pd.to_datetime(df['Date'])
    
    if target_date is None:
        target_date = df['Date'].max().date()
    else:
        target_date = pd.to_datetime(target_date).date()
        
    df_today = df[df['Date'].dt.date == target_date].copy()
    print(f"Plotting risk map for: {target_date} ({len(df_today)} countries found)")

    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    print("Downloading base map from Natural Earth...")
    world = gpd.read_file(url)

    world.columns = world.columns.str.lower()
    
    if 'name' not in world.columns and 'admin' in world.columns:
        world['name'] = world['admin']

    world.loc[world['name'] == 'France', 'iso_a3'] = 'FRA'
    world.loc[world['name'] == 'Norway', 'iso_a3'] = 'NOR'
        
    world_risk = world.merge(df_today, how='left', left_on='iso_a3', right_on=iso3_column)
    
    world_risk[risk_column] = world_risk[risk_column].fillna(0).astype(int)
    
    custom_cmap = ListedColormap(['#A9A9A9', '#FFD700', '#FF8C00', '#DC143C'])
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    world_risk.plot(
        column=risk_column, 
        cmap=custom_cmap, 
        linewidth=0.3, 
        ax=ax, 
        edgecolor='black',
        vmin=0, 
        vmax=3,
        legend=True,
        legend_kwds={'shrink': 0.5, 'ticks': [0, 1, 2, 3]}
    )
    
    ax.set_title(f"Global Aviation Risk Levels ({target_date})", fontdict={'fontsize': 20}, pad=20)
    ax.set_axis_off()
    
    plt.show()

if __name__ == "__main__":

    data = pd.read_parquet('data/smoothed_risk_events.parquet')
    
    plot_daily_risk_map(data, iso3_column='iso3_country', risk_column='SmoothedRisk', target_date='2026-04-01')