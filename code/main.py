#Import libraries
from colorama import Fore
import configuration_settings as cs
import data_processing as dpr
import visualization as vis
import training as tr
import numpy as np
import pandas as pd
import sys, os 

if __name__ == "__main__":
    dpr.ERA5_data_downloader()

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

    df_ERA, df_CSR, ds_ERA_sliced, ds_CSR_sliced = dpr.Load_slice_conv_dataset(lat_min, lat_max, lon_min, lon_max)

# Regrid ERA5 and merge
    if  df_ERA is not None and df_CSR is not None:
        print(f"{Fore.CYAN} Regriding ERA5 to CSR Grace Resolution {Fore.RESET}")
        merged = dpr.Era_2_Csr(df_CSR, df_ERA)
# We are ready for ML
    choice = input(f"{Fore.CYAN}Would you like to run the rfe? Write YES/NO {Fore.RESET}").strip().upper()
    if choice == "YES":
#RFE
        print(f"{Fore.CYAN}Starting Random Feature Selection (RFE){Fore.RESET}")
        try:
            model_RFE = (input(f"{Fore.CYAN}Choose the model you want to use: (XGBoost/RF) {Fore.RESET}"))
        except ValueError:
            print(f"{Fore.RED}Error! Please enter 'XGBoost' or 'RF'.{Fore.RESET}")
            sys.exit()
        try: 
            n_features_to_select = int(input("Insert the number of features you want to select : ")) 
            print(f"The RFE will be performed with: {n_features_to_select} features")
        except ValueError:
            print("Invalid input!")
            sys.exit()

        rfe, selected_features, x = tr.rfe(merged,model_RFE,n_features_to_select)
    
        print(f"{Fore.CYAN}Creating rfe feature ranking plot{Fore.RESET}")
    
        folder_name_rfe = str(input(f"{Fore.CYAN}Insert folder name you want to save the plot {Fore.RESET}"))
        title_rfe = str(input(f"{Fore.CYAN}Insert title  {Fore.RESET}"))
        extension_rfe = str(input(f"{Fore.CYAN}Insert .extension (eg. .jpg) {Fore.RESET}"))

        output_rfe = vis.dynamic_t(folder_name_rfe, title_rfe, extension_rfe)

        vis.rfe_plot(rfe, x, output_rfe)
    
    choice2 = input(f"{Fore.CYAN}Would you like to create ERA5 feature map? Write YES/NO {Fore.RESET}").strip().upper()
    if choice2 == "YES":
    #ERA5 Map
        print(f"{Fore.CYAN}Creating ERA5 feature map{Fore.RESET}")
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

        try:
            basin_name = input("Enter Basin Name for the Title (e.g. Amazon, Lake Victoria): ").strip()
        except ValueError:
            print(Fore.RED + " Invalid input! Terminating programm.")
            sys.exit()
        try:
            map_year = int(input("Enter year: "))
            map_month = int(input("Enter month: "))
        except ValueError:
            print(Fore.RED + " Invalid input! Terminating programm.")
            sys.exit() 
        
        folder_name_ERA = str(input(f"{Fore.CYAN}Insert folder name you want to save the plot {Fore.RESET}"))
        title_ERA = str(input(f"{Fore.CYAN}Insert ERA5 map title  {Fore.RESET}"))
        extension_ERA = str(input(f"{Fore.CYAN}Insert .extension (eg. .jpg) {Fore.RESET}"))

        output_ERA = vis.dynamic_t(folder_name_ERA, title_ERA, extension_ERA)
        #print(output_ERA)
        vis.ERA_plot(ds_ERA_sliced, map_year, map_month, ERA_var_to_plot, basin_name, output_ERA)

    #GRACE comparison Map
    print(f"{Fore.CYAN}Creating Comparison grace raw/predicted/diff  map{Fore.RESET}")
    try:
        basin_name = input("Enter Basin Name for the Title (e.g. Amazon, Lake Victoria): ").strip()
    except ValueError:
        print(Fore.RED + " Invalid input! Terminating programm.")
        sys.exit()
    try:
        map_year = int(input("Enter year: "))
        map_month = int(input("Enter month: "))
    except ValueError:
        print(Fore.RED + " Invalid input! Terminating programm.")
        sys.exit()

    models_dir = cs.MODELS_DIR
    
    # Select trained Model
    print(f"\nScanning directory: {models_dir} ")
    try:
        available_models = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
        if not available_models:
            print(Fore.RED + " Error: No .pkl files found in the models directory!")
            sys.exit()
    
        print(" Available Models:")
        for i, model_file in enumerate(available_models):
            print(f"  [{i+1}] {model_file}")

        try:
            selection = int(input(f"\nSelect Model Number (1-{len(available_models)}): "))
            if 1 <= selection <= len(available_models):
                selected_model_name = available_models[selection - 1]
                full_model_path = os.path.join(models_dir, selected_model_name)
                print(f"  Selected: {selected_model_name}")
            else:
                print(Fore.RED + "  Invalid number selected. Exiting.")
                sys.exit()
        except ValueError:
            print(Fore.RED +"  Invalid input. Please enter a number.")
            sys.exit()

    except FileNotFoundError:
        print(Fore.RED + f"  Error: Directory '{models_dir}' not found.")
        sys.exit() 

    folder_name_CSR = str(input(f"{Fore.CYAN}Insert folder name you want to save the plot {Fore.RESET}"))
    title_CSR = str(input(f"{Fore.CYAN}Insert map title  {Fore.RESET}"))
    extension_CSR = str(input(f"{Fore.CYAN}Insert .extension (eg. .jpg) {Fore.RESET}"))
   
    output_CSR = vis.dynamic_t(folder_name_CSR, title_CSR, extension_CSR)
    var_to_plot = "lwe_thickness"

    df_CSR_on_ERA_grid = dpr.CSR_interp(df_CSR, df_ERA)
   
    vis.CSR_plot(full_model_path, map_year, map_month, output_CSR, ds_CSR_sliced, df_CSR_on_ERA_grid, df_ERA, var_to_plot, basin_name)

#stats real vs predicted lwe
    print(f"{Fore.CYAN}Creating Comparison grace raw/predicted statistical analysis plot{Fore.RESET}")

    try:
        start_year = int(input("Enter starting year: "))
        end_year = int(input("Enter ending year: "))
    except ValueError:
        print(Fore.RED + " Invalid input! Terminating programm.")
        sys.exit()
    df_pred_4_stats = dpr.cont_prediction(df_ERA,full_model_path, start_year, end_year)
    # merging 2 dfs
    merged_ev_stats = pd.merge(df_CSR_on_ERA_grid, df_pred_4_stats,on=["year", "month", "lat", "lon"],
    how="inner",)

    print(f"{Fore.CYAN}Choose lat, lon for plot{Fore.RESET}")
    try:
        target_lat = float(input("Enter latitude: "))
        target_lon = float(input("Enter longtitude: "))
    except ValueError:
        print(Fore.RED + " Invalid input! Terminating programm.")
        sys.exit()
    merged_ev_stats_cl = dpr.cleaner_4_stats(merged_ev_stats, target_lat, target_lon)
    print(merged_ev_stats_cl.head(10))

    folder_name_stats = str(input(f"{Fore.CYAN}Insert folder name you want to save the plot {Fore.RESET}"))
    title_stats = str(input(f"{Fore.CYAN}Insert map title  {Fore.RESET}"))
    extension_stats = str(input(f"{Fore.CYAN}Insert .extension (eg. .jpg) {Fore.RESET}"))
   
    output_stats = vis.dynamic_t(folder_name_stats, title_stats, extension_stats)
    vis.model_eval_plot(merged_ev_stats_cl, output_stats)
