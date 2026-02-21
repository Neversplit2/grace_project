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
import seaborn as sns
import training as tr

def rfe_plot(rfe, x):
    df_ranking = pd.DataFrame()
    # Creating column "Feature" in order to save the rfe feature names participating 
    df_ranking["Feature"] = x.columns
    # Creating column "Rank" in order to save the rfe feature ranking
    df_ranking["Rank"] = rfe.ranking_ 
    # Sorting df_ranking based on column "Rank", in ascending order (True)
    df_ranking = df_ranking.sort_values(by=['Rank'], ascending=True)

    # 1. Create the figure AND the axes explicitly
    fig, ax = plt.subplots(figsize=(6, 7), dpi=200) 
    
    # 2. Draw the barplot ON our specific axes (notice ax=ax at the end)
    sns.barplot(data=df_ranking, x="Rank", y="Feature", hue="Rank", palette="coolwarm", legend=False, ax=ax)

    # 3. Formatting (we use ax.set_title instead of plt.title)
    ax.set_title(f"RFE Feature Ranking (Top {rfe.n_features_} Selected)", fontsize=12)
    ax.set_xlabel("Rank (1 = Selected, Higher = Eliminated Early)", fontsize=10)
    ax.set_ylabel("Features", fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    
    # Changing feature fontsize using tick_params
    ax.tick_params(axis='y', labelsize=8)
    
    # 4. Return the plot object instead of showing/saving it!
    return fig