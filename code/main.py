#Import libraries
from colorama import Fore
import configuration_settings as cs
import data_processing as dp
import visualization as vis
import prediction as pred
import numpy as np

if __name__ == "__main__":
    dp.ERA5_data_downloader()

    #User Input for Area of Interest
    print("Input Area of Interest extend lat[-90,90], lon[-180,180]")
    try:
        lat_min = float(input("Latitude Min : "))
        lat_max = float(input("Latitude Max : "))
        lon_min = float(input("Longitude Min : "))
        lon_max = float(input("Longitude Max : "))
        print(f"\nArea of Interest: Lat[{lat_min}, {lat_max}], Lon[{lon_min}, {lon_max}]")
    except ValueError:
        print(f"{Fore.RED} Wrong input! Please insert a number!{Fore.RESET}")
        print(f"\nArea of Interest: Lat[{lat_min}, {lat_max}], Lon[{lon_min}, {lon_max}]")

    df_ERA, df_CSR = dp.Load_slice_conv_dataset(lat_min, lat_max, lon_min, lon_max)

# Regrid ERA5 and merge
    if  df_ERA is not None and df_CSR is not None:
        # Compute GRACE resolution
        g_lats = np.sort(df_CSR["lat"].unique()) 
        g_lons = np.sort(df_CSR["lon"].unique())

        dlat = np.diff(g_lats)
        dlon = np.diff(g_lons)

        step_lat = dlat[dlat > 0].min()
        step_lon = dlon[dlon > 0].min()

        step = float(min(step_lat, step_lon))
        #Regrid ERA5 
        df_CSR["lat_r"] = (df_CSR["lat"] / step).round() * step
        df_CSR["lon_r"] = (df_CSR["lon"] / step).round() * step

        df_ERA["lat_r"] = (df_ERA["lat"] / step).round() * step
        df_ERA["lon_r"] = (df_ERA["lon"] / step).round() * step

        df_ERA = df_ERA.groupby(["year", "month", "lat_r", "lon_r"]).mean().reset_index()
        #Merge 2 dataframes
        merged = pd.merge(
            df_CSR,
            df_ERA,
            on=["year", "month", "lat_r", "lon_r"],
            how="inner",
            suffixes=("_grace", "_era")
        )

        # Remove rows with year 2025 and above. Training is going to be done up to 2024. 
        # 2025 and on is going to be used for testing
        merged= merged[merged['year'] <= 2024]
        print(f"Most recent year in training dataset: {merged['year'].max()}")

        data_out = ['time','lat_r','lat_era','lon_r','lon_era']
        merged = merged.drop(columns = data_out, errors='ignore')
# We are ready for ML

