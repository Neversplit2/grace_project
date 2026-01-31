import configuration_settings as cs
import data_processing as dpr
from colorama import Fore
import sys
import os, joblib
import cartopy.crs as ccrs 
import cartopy.feature as cfeature 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


#So i want to create a function for the dynamic approach

def dynamic_t(directory, filename, extension=""):
    #I am adding the ../ in order to create the filepath
    directory = f"../{directory}"
    # Check if the chosen folder exists else terminate
    #try:
     #   os.makedirs(directory, exist_ok= True)
    ##   print(f" {Fore.RED} Wrong directory name!.{Fore.RESET}")
      #  sys.exit()
    if not os.path.exists(directory):
        print(f" {Fore.RED} Error: The directory '{directory}' does not exist!{Fore.RESET}")
        sys.exit()
    # Ensure extension has a dot
    if not extension.startswith("."):
        extension = f".{extension}"
        
    output_path = os.path.join(directory, f"{filename}{extension}")
    i=0
    while os.path.exists(output_path):
        output_path = os.path.join(directory,f"{filename}_{i}{extension}")
        i=i+1
    return output_path.replace("\\", "/")   
 
#RFE_ranking
#rfe = rfe , x = x, output = output_rfe
def rfe_plot(rfe, x, output):
    df_ranking = pd.DataFrame()
    # Creating column "Feature" in order to save the rfe feature names participating 
    df_ranking["Feature"] = x.columns
    # Creating column "Rank" in order to save the rfe feature ranking
    df_ranking["Rank"] = rfe.ranking_ 
    # Sorting df_ranking based on column "Rank", in ascending order (True)
    df_ranking = df_ranking.sort_values(by=['Rank'], ascending=True)
    #print(df_ranking)

    # Creating figure
    plt.figure(figsize=(4, 5), dpi=200) 
    # hue = controls the colour by grouping the input data
    # creating the barplot graph
    sns.barplot(data=df_ranking, x="Rank", y="Feature", hue="Rank",palette="coolwarm", legend=False)

    #Formatting
    plt.title(f"RFE Feature Ranking (Top {rfe.n_features_} Selected)", fontsize=11)
    plt.xlabel("Rank (1 = Selected, Higher = Eliminated Early)", fontsize=8)
    plt.ylabel("Features", fontsize=8)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    #Changing feature fontsize
    plt.yticks(fontsize=6)
    # Show the plot

    plt.savefig(output)
    plt.show()

#ERA5 map
#Jekinaei i sinartisi
def ERA_plot(dataset, year, month, var_to_plot, basin_name, output):
    data_slice = dataset[var_to_plot].sel(valid_time=f'{year}-{month}-01 23:00', method='nearest')
    time_str = data_slice.valid_time.dt.strftime("%m-%Y").item()
    # Find the value that is smaller than 2% of the data
    vmin = data_slice.quantile(0.02).item()
    # Find the value that is smaller than 98% of the data
    vmax = data_slice.quantile(0.98).item()
    plt.figure(figsize=(10, 8))

    ax = plt.axes(projection=ccrs.PlateCarree())
    plot = data_slice.plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="Blues",     
        robust=True,
        vmin=vmin, vmax=vmax,    
        cbar_kwargs={     
            "orientation": "vertical",
            "fraction": 0.03,
            "pad": 0.04,
            "label": f"Values ({var_to_plot})"     
        }
    )

    ax.coastlines(resolution="10m", color="black", linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor='gray')

    ax.add_feature(cfeature.RIVERS, color='lightblue', linewidth=0.8)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False   
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # Title
    ax.set_title(f"{basin_name.upper()} ERA5 {var_to_plot.lower()}  [{time_str}]", fontsize=14, fontweight='bold')

    # Remove default xarray labels
    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f" Raster plot saved as: {output}")
    print("Map")
    plt.show()

#Comparison_GRACE

#model= full_model_path, #year= map_year, month=map_month
#dataset_CSR = ds_CSR_sliced, #dataset_ERA = df_ERA, dataset_diff = pros to parwn to ftiaxnw mesa argotera isws to prosthesw alliws 
#output = output_CSR, dataset_CSR2 = CSR_on_ERA_grid
def CSR_plot2(model, year, month, output, dataset_CSR, dataset_CSR2, dataset_ERA, var_to_plot, basin_name):
    model = joblib.load(model)
    target_ym = f"{year}-{month:02d}"
   
    # -------------------------
    # ERA5 data after prediction
    # -------------------------
   # dataset = , model = full_model_path, year = map_year, month = map_month
    input_ERA_data = dpr.prediction(dataset_ERA, model, year, month, )

        #data_predicted2 is the dataframe that i am gonna use for the lwe_diff
    data_predicted2 = input_ERA_data.copy()

    #ds_pred = input_ERA_data.groupby(["lat", "lon"])[["lwe_pred"]].mean().to_xarray()
    ds_pred = input_ERA_data.set_index(["lat", "lon"])[["lwe_pred"]].to_xarray()
    data_predicted = ds_pred["lwe_pred"]

    t_index = pd.DatetimeIndex(dataset_CSR.time.values)
    mask = (t_index.year == year) & (t_index.month == month)
    
    data_exist = mask.any()    
    #picked_ts = pd.Timestamp(data_actual.time.values)
    #time_str = picked_ts.strftime("%Y-%m")

    time_str = f"{month:02d}-{year}"

    if data_exist == False:
        print(Fore.RED + f"No CSR data found for {month}/{year}{Fore.RESET}! \n {Fore.CYAN} Creating plot from the predicted lwe thickness {Fore.RESET} ")
        #Predicted map
        
        vmin = data_predicted.quantile(0.02).item()
        vmax= data_predicted.quantile(0.98).item()
        
        plt.figure(figsize=(10,8))

        ax = plt.axes(projection=ccrs.PlateCarree())
        plot = data_predicted.plot.pcolormesh(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap="RdBu",
            robust=True,
            cbar_kwargs={"label": "Predicted LWE (cm)", "orientation": "horizontal", "pad": 0.05,"fraction": 0.03},
            vmin=vmin, vmax=vmax
        )
        ax.set_title(f"{basin_name} Predicted LWE\n {time_str}", fontsize=14, fontweight="bold")
        ax.coastlines(resolution="10m")
        ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor='gray')
        ax.add_feature(cfeature.RIVERS, color="lightblue", alpha=0.5)
        ax.gridlines(draw_labels= True, linewidth=0.5, linestyle="--", alpha=0.5, color='gray')

        plt.savefig(output, dpi=300, bbox_inches="tight")
        print(f" Raster plot saved as: {output}")
        print("Map")
        plt.show()

    else:
        # -------------------------
        # lwe_difference
        # -------------------------
        idx = int(np.where(mask)[0][0])
        data_actual = dataset_CSR[var_to_plot].isel(time=idx)
        
        dataset_diff = pd.merge(
            dataset_CSR2,
            data_predicted2,
            on=["year", "month", "lat", "lon"],
            how="inner",
            suffixes=("_grace", "_grace_pred")
        )

        data_out=['t2m', 'tp', 'e', 'pev', 'ssro', 'sro', 'evabs','swvl1', 'swvl2', 'swvl3', 'swvl4', 'lai_hv', 'lai_lv']
        dataset_diff = dataset_diff.drop(columns=data_out, errors='ignore')
        dataset_diff["lwe_difference"] = abs(dataset_diff["lwe_pred"] - dataset_diff["lwe_thickness"])

        input_diff_data = dataset_diff[(dataset_diff['year'] == year) & (dataset_diff['month'] == month)]
        
        ds_diff = input_diff_data.set_index(['lat', 'lon']).to_xarray()
        data_slice_diff = ds_diff['lwe_difference']

        # 4. Handle Time String Manually
        # Since we lost the 'valid_time' column during earlier processing, we construct the string ourselves
        time_str = f"{month:02d}-{year}" 

        # Plot
        vmin = data_actual.quantile(0.02).item()
        vmax = data_actual.quantile(0.98).item()

        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3, figsize=(20, 8),
            subplot_kw={"projection": ccrs.PlateCarree()}
        )

        # Map 1
        data_actual.plot.pcolormesh(
            ax=ax1,
            transform=ccrs.PlateCarree(),
            cmap="RdBu",
            robust=True,
            cbar_kwargs={"label": "LWE (cm)", "orientation": "horizontal", "pad": 0.05, "fraction": 0.03},
            vmin=vmin, vmax=vmax
        )
        ax1.set_title(f"{basin_name} Observed LWE\n{time_str}", fontsize=14, fontweight="bold")
        ax1.coastlines(resolution="10m")
        ax1.add_feature(cfeature.BORDERS, linestyle=":", edgecolor='gray')
        ax1.add_feature(cfeature.RIVERS, color="lightblue", alpha=0.5)
        ax1.gridlines(draw_labels= True, linewidth=0.5, linestyle="--", alpha=0.5, color='gray')

        # Map 2
        data_predicted.plot.pcolormesh(
            ax=ax2,
            transform=ccrs.PlateCarree(),
            cmap="RdBu",
            robust=True,
            cbar_kwargs={"label": "Predicted LWE (cm)", "orientation": "horizontal", "pad": 0.05,"fraction": 0.03},
            vmin=vmin, vmax=vmax
        )
        ax2.set_title(f"{basin_name} Predicted LWE\n {time_str}", fontsize=14, fontweight="bold")
        ax2.coastlines(resolution="10m")
        ax2.add_feature(cfeature.BORDERS, linestyle=":", edgecolor='gray')
        ax2.add_feature(cfeature.RIVERS, color="lightblue", alpha=0.5)
        ax2.gridlines(draw_labels= True, linewidth=0.5, linestyle="--", alpha=0.5, color='gray')

        # Map 3
        # Note: data_slice is now an Xarray DataArray, so .plot works!
        plot = data_slice_diff.plot.pcolormesh(
            ax=ax3,
            transform=ccrs.PlateCarree(),
            cmap="Reds",     
            robust=True,
            vmin=0 , vmax=15,    
            cbar_kwargs={"orientation": "horizontal", "fraction": 0.03,"pad": 0.05, "label": "LWE difference (cm)"}
            )

        ax3.set_title(f"Difference between predicted and raw {basin_name}'s data {time_str}", fontsize=14, fontweight='bold')
        ax3.coastlines(resolution="10m", color="black", linewidth=1)
        ax3.add_feature(cfeature.BORDERS, linestyle=":", edgecolor='gray')
        ax3.add_feature(cfeature.RIVERS, color='lightblue', linewidth=0.8)
        ax3.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

        plt.tight_layout()
        plt.savefig(output, dpi=300, bbox_inches="tight")
        print(f" Map saved to: {output}")
        plt.show()
