#Import libraries
#Download data
import cdsapi
#Computing
import numpy as np
import xarray as xr
import pandas as pd
import configuration_settings as cs
from colorama import Fore
from scipy.interpolate import griddata
import os, joblib, sys

#ERA5 data 
def ERA5_data_downloader():
    file_ERA_path = cs.DATA_DIR / cs.ERA5_FILE
    # In case data not existing
    client = cdsapi.Client()
    dataset = "reanalysis-era5-single-levels-monthly-means"
    if file_ERA_path.exists():
        print(f"{Fore.GREEN} ERA5 data found :) {Fore.RESET}")
    else: 
        print(f"{Fore.RED} ERA5 data missing! We will use default API to download {Fore.RESET}")
        request = {
            "product_type": ["monthly_averaged_reanalysis"],
            "variable": [
                "2m_temperature",
                "total_precipitation",
                "total_evaporation",
                "potential_evaporation",
                "sub_surface_runoff",
                "surface_runoff",
                "evaporation_from_bare_soil",
                "volumetric_soil_water_layer_1",
                "volumetric_soil_water_layer_2",
                "volumetric_soil_water_layer_3",
                "volumetric_soil_water_layer_4",
                "leaf_area_index_high_vegetation",
                "leaf_area_index_low_vegetation"
            ],
            "year": [
                "2002", "2003", "2004",
                "2005", "2006", "2007",
                "2008", "2009", "2010",
                "2011", "2012", "2013",
                "2014", "2015", "2016",
                "2017", "2018", "2019",
                "2020", "2021", "2022",
                "2023", "2024", "2025"
            ],
            "month": [
                "01", "02", "03",
                "04", "05", "06",
                "07", "08", "09",
                "10", "11", "12"
            ],
            "time": ["00:00"],
            "data_format": "netcdf",
            "download_format": "zip"
        }
        # str((DATA_DIR / "era5_data.zip")): Take the data folder and download era5 in a folder named "era5_data.zip"
        client.retrieve(dataset, request).download(str(cs.DATA_DIR / "ERA5_data.zip")) 
        # I might automate this unzip later
        print(f"{Fore.YELLOW} ACTION REQUIRED: Please manually unzip 'era5_data.zip' in your data folder.{Fore.RESET}")

# This is a helping function, that i am gonna call later in another function 
def slice_nc(ds, ds_name, lat_min, lat_max, lon_min, lon_max):
    print(f"\n{Fore.CYAN}Slicing {ds_name}...{Fore.RESET}")

    # 1. Identify Coordinate Names
    possible_lat_names = ['lat', 'latitude']
    possible_lon_names = ['lon', 'longitude']
    
    lat_name = next((n for n in ds.coords if n.lower() in possible_lat_names), None)
    lon_name = next((n for n in ds.coords if n.lower() in possible_lon_names), None)
    
    if not lat_name or not lon_name:
        raise ValueError(f"ERROR: Could not find lat/lon columns in {ds_name}")

    # 2. Handle Longitude Conversion (0-360 -> -180 to 180)
    if ds[lon_name].max() > 180:
        print("  Converting Longitude to -180, 180 format")
        ds = ds.assign_coords({lon_name: (((ds[lon_name] + 180) % 360) - 180)})
        ds = ds.sortby(lon_name)

    # 3. Handle Latitude Slicing Order
    # If data is 90->-90 (descending), slice must be (max, min)
    lat_values = ds[lat_name].values
    if lat_values[0] > lat_values[-1]: 
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)

    # 4. Perform the Slice
    ds_sliced = ds.sel({
        lat_name: lat_slice, 
        lon_name: slice(lon_min, lon_max)
    })
    
    print(f"  {ds_name} sliced. New shape: {ds_sliced[lat_name].size}x{ds_sliced[lon_name].size}")
    return ds_sliced

#This is also a helping function that converts xarrays dataset to pandas dataframe.
def conv_2_df(ds,name):
    print(f"{Fore.CYAN} Converting {name} to dataframe {Fore.RESET}")
    df = ds.to_dataframe().reset_index() 

    possible_time_names = ['time', 'valid_time']
    time_col = next((col for col in df.columns if col in possible_time_names), None)
    if time_col is None:
        print(f"{Fore.RED} ERROR! No time column found {Fore.RESET}")
        return None 
    df["year"] = df[time_col].dt.year
    df["month"] = df[time_col].dt.month
    return df

# In this function I call slice_nc and conv_2_df 
def Load_slice_conv_dataset(lat_min, lat_max, lon_min, lon_max):
    
    file_ERA_path = cs.DATA_DIR / cs.ERA5_FILE
    file_CSR_path = cs.DATA_DIR / cs.GRACE_CSR_FILE

    if not file_ERA_path.exists():
        print(f"{Fore.RED} ERROR! ERA5 data not found \n Run downloader first")
        return None #In order to stop the function
    if not file_CSR_path.exists():
        print(f"{Fore.RED} ERROR! GRACE CSR data not found!")
        return None
    try:
        ds1 = xr.open_dataset(file_ERA_path) #Maybe i can create a small function that opens nc files (super easy)
        ds_ERA_sliced = slice_nc(ds1, "ERA5 data", lat_min, lat_max, lon_min, lon_max)
        ds1.close() # Close original file to free memory

        ds2 = xr.open_dataset(file_CSR_path)
        ds_CSR_sliced = slice_nc(ds2, "GRACE CSR data", lat_min, lat_max, lon_min, lon_max)

        #Fixing time Grace
        ds_CSR_sliced["time"] = pd.to_datetime(
            ds_CSR_sliced.time.values, origin="2002-01-01", unit="D"
        )
        # Drop time_bounds column, if it exists
        ds_CSR_sliced = ds_CSR_sliced.drop_vars("time_bounds", errors="ignore")
        ds2.close()

        df_ERA = conv_2_df(ds_ERA_sliced,"ERA5")
        #(I could leave it out of the func)
        df_ERA = df_ERA.rename(columns={"longitude": "lon", "latitude": "lat"})
        df_ERA = df_ERA.drop(columns=['valid_time', 'number', 'expver'])

        df_CSR = conv_2_df(ds_CSR_sliced,"GRACE")

        return df_ERA, df_CSR, ds_ERA_sliced, ds_CSR_sliced
    # We save the error message in variable e and print it
    except Exception as e:
        print(f"{Fore.RED} Error during processing: {e}{Fore.RESET}")
        return None, None, None, None
    
#dataset_CSR = df_CSR 
def Era_2_Csr(dataset_CSR, dataset_ERA):
        # Compute GRACE resolution
        g_lats = np.sort(dataset_CSR["lat"].unique()) 
        g_lons = np.sort(dataset_CSR["lon"].unique())

        dlat = np.diff(g_lats)
        dlon = np.diff(g_lons)

        step_lat = dlat[dlat > 0].min()
        step_lon = dlon[dlon > 0].min()

        step = float(min(step_lat, step_lon))
        #Regrid ERA5 
        dataset_CSR["lat_r"] = (dataset_CSR["lat"] / step).round() * step
        dataset_CSR["lon_r"] = (dataset_CSR["lon"] / step).round() * step

        df_ERA2=dataset_ERA.copy()
        df_ERA2["lat_r"] = (df_ERA2["lat"] / step).round() * step
        df_ERA2["lon_r"] = (df_ERA2["lon"] / step).round() * step

        df_ERA2 = df_ERA2.groupby(["year", "month", "lat_r", "lon_r"]).mean().reset_index()
        #Merge 2 dataframes
        print(f"{Fore.CYAN} Merging ERA5 & Grace dataframes {Fore.RESET}")
        merged = pd.merge(
            dataset_CSR,
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
        
        return merged

#dataset_CSR = df_CSR, dataset_ERA = df_ERA 
def CSR_interp(dataset_CSR, dataset_ERA):
    grace_df2= dataset_CSR.copy()
    data_off = ['time','lat_r','lon_r']
    grace_df2 = grace_df2.drop(columns= data_off, errors='ignore')
    target_lats = np.sort(dataset_ERA["lat"].unique())
    target_lons = np.sort(dataset_ERA["lon"].unique())
    #indexing ='' Cartesian (‘xy’, default) or matrix (‘ij’) 
    grid_lat, grid_lon = np.meshgrid(target_lats, target_lons, indexing='ij')

    grace_regridded_list = []

    for (year, month), group in grace_df2.groupby(['year', 'month']):
        # Source Points (Old GRACE Grid)
        points = group[['lat', 'lon']].values
        values = group['lwe_thickness'].values
        # Interpolate onto the ERA5 Grid
        # method='nearest' ensures we just copy the closest value (Big Tile logic)
        grid_z = griddata(points, values, (grid_lat, grid_lon), method='nearest')
        # Store result
        grace_regridded_list.append(pd.DataFrame({
            'year': year,
            'month': month,
            'lat': grid_lat.flatten(),
            'lon': grid_lon.flatten(),
            'lwe_thickness': grid_z.flatten()
        }))

    CSR_on_ERA_grid = pd.concat(grace_regridded_list)
    
    return CSR_on_ERA_grid

# dataset =df_ERA , model = full_model_path, year = map_year, month = map_month
def prediction(dataset, model, year, month):
    target_ym = f"{year}-{month:02d}"
    input_ERA_data = dataset[(dataset["year"] == year) & (dataset["month"] == month)].copy()
    if input_ERA_data.empty:
        print(Fore.RED + f" Error: ERA5 data not found for {target_ym}.")
        sys.exit()
    else:
        
        try:
            required_features = model.feature_names_in_
        except AttributeError:
            print(f"{Fore.RED}Error!{Fore.RESET}")
            sys.exit()
            #print("Using default features")
            #required_features = features

        missing_feats = [c for c in required_features if c not in input_ERA_data.columns]
        if missing_feats:
            raise KeyError(f"Missing required features in ERA5 input_data: {missing_feats}")

        X_pred = input_ERA_data[required_features]
        input_ERA_data["lwe_pred"] = model.predict(X_pred)
        return input_ERA_data

