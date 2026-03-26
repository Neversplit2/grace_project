import configuration_settings as cs
import data_processing as dpr
from colorama import Fore
import sys
import os, joblib
import cartopy.crs as ccrs 
import cartopy.feature as cfeature 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
import numpy as np
import seaborn as sns
import training as tr

def rfe_plot(rfe, x):
    df_ranking = pd.DataFrame()
    df_ranking["Feature"] = x.columns
    df_ranking["Rank"] = rfe.ranking_ 
    df_ranking = df_ranking.sort_values(by=['Rank'], ascending=True)

    # 1. Set the Dark Theme Style
    plt.style.use('dark_background') # Instantly turns the background black
    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=200) # Slightly wider for better text fit
    fig.patch.set_facecolor('#0b0f19') # Matches your terminal background hex
    ax.set_facecolor('#0b0f19')

    # 2. Custom Neon Palette (Cyan to Magenta)
    # We use a custom gradient that screams "Tech"
    neon_colors = sns.color_palette("husl", len(df_ranking))
    
    sns.barplot(
        data=df_ranking, 
        x="Rank", 
        y="Feature", 
        palette="cool_r", # 'cool' goes from Cyan to Purple
        ax=ax,
        # edgecolor="#00E5FF", # Adds a neon glow effect to the bar edges
        linewidth=1
    )

    # 3. Formatting with Monospace Fonts
    # Using 'monospace' makes it look like code output
    ax.set_title(f" RFE FEATURE RANKING [TOP {rfe.n_features_}]", 
                 fontsize=10, color='#00E5FF', fontfamily='monospace', weight='bold', pad=20)
    
    ax.set_xlabel("Ranking level (OPTIMAL=1)", fontsize=8, color="#FFFFFF", fontfamily='monospace', weight='bold')
    ax.set_ylabel("Features", fontsize=8, color="#FFFFFF", fontfamily='monospace', weight='bold')
    
    # 4. Sci-Fi Grid & Spines
    ax.grid(axis="x", linestyle=":", alpha=0.3, color='#8892B0')
    
    # Remove the top and right borders for a cleaner "HUD" look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#1e293b')
    ax.spines['bottom'].set_color('#1e293b')
    
    # Tick formatting
    ax.tick_params(axis='both', colors='#8892B0', labelsize=7)
   
    # Bold the features selected
    for i, tick in enumerate(ax.get_yticklabels()):
        if df_ranking.iloc[i]["Rank"] == 1:
            tick.set_color("#FFFFFF")
            tick.set_weight("bold")
    plt.tight_layout()
    return fig

#Learning curves
def XGB_learn_curve(X_train, X_test, y_train, y_test):
    curve_steps, train_mae_list, val_mae_list = tr.XGBoost_curves(X_train, X_test, y_train, y_test)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=200) 
    
    ax.plot(curve_steps, train_mae_list, label='Train MAE', linewidth=2)
    ax.plot(curve_steps, val_mae_list, label='Validation MAE', linewidth=2)
    ax.set_xlabel('Number of Trees', fontsize=14)
    ax.set_ylabel('MAE', fontsize=14)
    ax.set_title('XGBoost Training Curve', fontsize=16)
    
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, linewidth=0.5)

    return fig

def RF_learn_curve(X_train, X_test, y_train, y_test):
    curve_steps, train_mae_list, val_mae_list= tr.RF_curves(X_train, X_test, y_train, y_test)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    
    ax.figure(figsize=(10, 6), dpi=200)
    ax.plot(curve_steps, train_mae_list, label='Train MAE', linewidth=2)
    ax.plot(curve_steps, val_mae_list, label='Validation MAE', linewidth=2)
    ax.set_xlabel('Number of Trees', fontsize=14)
    ax.set_ylabel('MAE', fontsize=14)
    ax.set_title('Random Forest Training Curve', fontsize=16)
    
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, linewidth=0.5)

    return fig

def ERA_plot(dataset, year, month, var_to_plot, basin_name):
    data_slice = dataset[var_to_plot].sel(valid_time=f'{year}-{month}-01 23:00', method='nearest')
    time_str = data_slice.valid_time.dt.strftime("%m-%Y").item()
    # Find the value that is smaller than 2% of the data
    vmin = data_slice.quantile(0.02).item()
    # Find the value that is smaller than 98% of the data
    vmax = data_slice.quantile(0.98).item()
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': ccrs.PlateCarree()})

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
    ax.set_title(f"{basin_name.upper()} ERA5 {var_to_plot.lower()}  [{time_str}]", fontfamily='monospace', fontsize=14, fontweight='bold')
    # Remove default xarray labels
    ax.set_xlabel("", fontfamily='monospace')
    ax.set_ylabel("", fontfamily='monospace')

    return fig

def CSR_plot(model, year, month, dataset_CSR, dataset_CSR2, dataset_ERA, var_to_plot, basin_name):
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
        
        fig, ax = plt.subplots(figsize=(10, 8), projection=ccrs.PlateCarree()) 
      
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

        return fig 

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

        ax3.set_title(f"Difference between predicted and raw {basin_name}'s \n data {time_str}", fontsize=14, fontweight='bold')
        ax3.coastlines(resolution="10m", color="black", linewidth=1)
        ax3.add_feature(cfeature.BORDERS, linestyle=":", edgecolor='gray')
        ax3.add_feature(cfeature.RIVERS, color='lightblue', linewidth=0.8)
        ax3.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

        return fig 
    
def model_eval_plot(dataframe):
    # yaxis= lwe_thickness/lwe_pred and x_axis= time (year-month)
    #i need to create the time variable
    plot_dates = pd.to_datetime(dataframe[['year', 'month']].assign(DAY=1))

    r_score, p_value = dpr.corr_pearson(dataframe)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8),)
 
    #Map 1
    #lwe_thickness
    ax1.plot(plot_dates, dataframe["lwe_thickness"], color='blue', linestyle='--', linewidth=2, label="CSR", alpha=0.7, 
             marker='o')

    # Predicted -> Green Solid Line
    ax1.plot(plot_dates, dataframe["lwe_pred"], color='green', linestyle='-', linewidth=2, label="Predicted CSR", alpha=0.7, 
             marker='x')

    ax1.set_title(f"Actual vs Predicted LWE for \n Lat: {dataframe['lat'].iloc[0]:.2f}, Lon: {dataframe['lon'].iloc[0]:.2f} R: {r_score:.4f} ", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Time")
    ax1.set_ylabel("LWE (cm)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Map 2
    ax2.scatter(dataframe["lwe_thickness"], dataframe["lwe_pred"], color='purple', alpha=0.6, label='Data Points', s=50)

    # best fit line
    a, b = np.polyfit(dataframe["lwe_thickness"], dataframe["lwe_pred"], 1)
    # line equation
    regression_line = a * dataframe["lwe_thickness"] + b
    
    ax2.plot(dataframe["lwe_thickness"], regression_line, color='red', linewidth=2, label=f'Trend Line (R={r_score:.2f})')
    
    
    ax2.set_title(f"Correlation Analysis\nScatter Plot & Trend Line", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Actual CSR Values (cm)")
    ax2.set_ylabel("Predicted Values (cm)")
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.5)

    return fig

def feature_importance_pie(model, X_train):
 
    model = joblib.load(model)
    importances = model.feature_importances_
    feature_names = X_train.columns
    
    # Create the DataFrame 
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    df = df.sort_values(by='Importance', ascending=False)

    # --- ENHANCED COLOR LOGIC ---
    # We use a more "vibrant" list of points for the gradient
    # Neon Cyan -> Bright Electric Blue -> Vivid Magenta/Purple
    n_features = len(df)
    custom_colors = ["#00FFFF", "#0080FF", "#8000FF", "#FF00FF"]
    cmap = mcolors.LinearSegmentedColormap.from_list("intense_sci_fi", custom_colors)
    colors = [cmap(i) for i in np.linspace(0, 1, n_features)][::-1]

    # --- PLOTTING ---
    plt.style.use('dark_background') # Base dark theme
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=180) # Higher DPI for sharper lines
    fig.patch.set_facecolor('#0b0f19') # Match your UI background
    
    # Simple Pie Chart
    #wedges = slices , texts = features, autotexts= percentage
    wedges, texts, autotexts =ax.pie(
        df['Importance'], 
        labels=df['Feature'], 
        autopct='%1.1f%%',      # Shows percentage with 1 decimal
        startangle=140,          # Starts the first slice at the top
        pctdistance=0.88,
        colors= colors,
        radius =1.1,
        #textprops={'color': "white"}
        wedgeprops={'linewidth': 0.25, 'edgecolor': '#0b0f19'}
    )

    for text in texts:
        text.set_fontfamily('monospace') # Sci-fi style
        text.set_fontsize(9)
        text.set_weight('bold')          # Makes names stand out
        text.set_color('#8892B0')        # Muted blue-gray to match your theme

    for text in autotexts:
        text.set_fontfamily('monospace') # Sci-fi style
        text.set_fontsize(7)
        text.set_weight('bold')          # Makes names stand out
        text.set_color("#000000")       

    ax.set_title("Feature Importance", color="#00E5FF", fontfamily="monospace", fontsize=14, pad=25)
    
    return fig 
