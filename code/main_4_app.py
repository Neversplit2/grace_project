import configuration_settings as cs
import data_processing as dpr
import visualization as vis
import training as tr
import numpy as np
import pandas as pd
import sys, os, joblib 

#Pipeline number 1: Data preparation
def pipe_data_prp(GRACE_data, lat_min, lat_max, lon_min, lon_max):
    dpr.ERA5_data_downloader()
    # Slice & merge datasets
    df_ERA, df_CSR, ds_ERA_sliced, ds_CSR_sliced = dpr.Load_slice_conv_dataset(
        GRACE_data, lat_min, lat_max, lon_min, lon_max)
    merged = None
    if  df_ERA is not None and df_CSR is not None:
        merged = dpr.Era_2_Csr(df_CSR, df_ERA)
    #df_CSR_on_ERA_grid is needed for the GRace comparison map
    df_CSR_on_ERA_grid =  dpr.CSR_interp(df_CSR, df_ERA)

    return df_ERA, df_CSR, ds_ERA_sliced, ds_CSR_sliced, merged, df_CSR_on_ERA_grid

#Pipeline number 2: Feature Selection (RFE)
def pipe_RFE(merged, model_type, n_features_to_select ):
    rfe, selected_features, x = tr.rfe(merged, model_type, n_features_to_select)

    return rfe, selected_features, x 

#Pipeline number 3: Model training
def pipe_model_train(selected_features, x, merged, model_type):
    X_train, X_test, y_train, y_test = tr.data_4_train(selected_features, x, merged)

    if model_type == "XGBoost":
        best_model = tr.XGBoost_train(X_train, y_train)
    elif model_type == "RF":
        best_model = tr.RF_train(X_train, y_train)
    else:
        best_model = None
    
    return X_train, X_test, y_train, y_test, best_model

#Pipeline number 4: Statistical Analysis
def pipe_stats(df_ERA, full_model_path, start_year, end_year, df_CSR_on_ERA_grid, target_lat, target_lon):
    df_pred_4_stats = dpr.cont_prediction(df_ERA, full_model_path, start_year, end_year)
    merged_ev_stats = pd.merge(df_CSR_on_ERA_grid, df_pred_4_stats,on=["year", "month", "lat", "lon"],
        how="inner",)
    merged_ev_stats_cl = dpr.cleaner_4_stats(merged_ev_stats, target_lat, target_lon)

    return merged_ev_stats_cl
