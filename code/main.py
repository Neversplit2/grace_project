#Import libraries
from colorama import Fore
import configuration_settings as cs
import data_processing as dp
import visualization as vis
import training as tr
import numpy as np
import pandas as pd
import sys 

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

    df_ERA, df_CSR, ds_ERA_sliced = dp.Load_slice_conv_dataset(lat_min, lat_max, lon_min, lon_max)

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

        df_ERA2=df_ERA.copy()
        df_ERA2["lat_r"] = (df_ERA2["lat"] / step).round() * step
        df_ERA2["lon_r"] = (df_ERA2["lon"] / step).round() * step

        df_ERA2 = df_ERA2.groupby(["year", "month", "lat_r", "lon_r"]).mean().reset_index()
        #Merge 2 dataframes
        merged = pd.merge(
            df_CSR,
            df_ERA2,
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
        merged.rename(columns={"lon_grace": "lon", "lat_grace": "lat"}, inplace=True)
# We are ready for ML

    #ERA5 Map

    feature_list = list(ds_ERA_sliced.data_vars)
    # Creating numeric list of ds_era5_merged features 
    print("\n Available Features ")
    for i, feature in enumerate(feature_list, 0):
        print(f" [{i}] {feature}")
    print(f"{Fore.CYAN}\n Choose a number from the list above to create a raster plot for the corresponding feature.{Fore.RESET}")
    # User chooses a number according to list
    try:
        era5_feature = int(input("Pick a number from the list: ")) 
    except ValueError:
        print(f" {Fore.RED} Invalid input! Terminating programm.{Fore.RESET}")
        sys.exit()
    ERA_var_to_plot = feature_list[era5_feature]
    #print(ERA_var_to_plot)

    folder_name_ERA = str(input(f"{Fore.CYAN}Insert folder name you want to save the plot {Fore.RESET}"))
    title_ERA = str(input(f"{Fore.CYAN}Insert ERA5 map title  {Fore.RESET}"))
    extension_ERA = str(input(f"{Fore.CYAN}Insert .extension (eg. .jpg) {Fore.RESET}"))

    output_ERA = vis.dynamic_t(folder_name_ERA, title_ERA, extension_ERA)
    #print(output_ERA)
