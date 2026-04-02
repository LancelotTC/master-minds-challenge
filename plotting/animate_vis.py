import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation

def animate_risk_map(df, iso3_column='iso3_country', risk_column='SmoothedRisk', save_path='aviation_risk_animation.mp4'):
    """
    Creates an animation of aviation risk levels changing over time.
    """
    df['Date'] = pd.to_datetime(df['Date'])
    all_dates = sorted(df['Date'].unique())
    
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    print("Downloading base map...")
    world = gpd.read_file(url)
    world.columns = world.columns.str.lower()
    
    if 'name' not in world.columns and 'admin' in world.columns:
        world['name'] = world['admin']
    world.loc[world['name'] == 'France', 'iso_a3'] = 'FRA'
    world.loc[world['name'] == 'Norway', 'iso_a3'] = 'NOR'

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    custom_cmap = ListedColormap(['#A9A9A9', '#FFD700', '#FF8C00', '#DC143C'])
    
    def update(frame):
        ax.clear()  # Clear previous frame
        target_date = all_dates[frame]
        df_frame = df[df['Date'] == target_date].copy()
        
        # Merge data for this specific date
        world_risk = world.merge(df_frame, how='left', left_on='iso_a3', right_on=iso3_column)
        world_risk[risk_column] = world_risk[risk_column].fillna(0).astype(int)
        
        world_risk.plot(
            column=risk_column, 
            cmap=custom_cmap, 
            linewidth=0.3, 
            ax=ax, 
            edgecolor='black',
            vmin=0, 
            vmax=3
        )
        
        # Formatting per frame
        readable_date = pd.to_datetime(target_date).strftime('%Y-%m-%d')
        ax.set_title(f"Global Aviation Risk Levels: {readable_date}", fontsize=20, pad=20)
        ax.set_axis_off()
        

    print(f"Generating animation for {len(all_dates)} days...")
    anim = FuncAnimation(fig, update, frames=len(all_dates), interval=50) # 200ms per frame
    
   
    anim.save(save_path, writer='ffmpeg', fps=5)
    plt.close()
    print(f"Animation saved to: {save_path}")

if __name__ == "__main__":

    data = pd.read_parquet('data/smoothed_risk_events.parquet')
    
    animate_risk_map(data)