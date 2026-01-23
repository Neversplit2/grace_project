#Import libraries
import configuration_settings as cs
import data_processing as dp
from colorama import Fore
import xgboost as xgb
import joblib, sys
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#RFE
#dataset = merged, model = model_RFE 
def rfe(dataset, model, n_features_to_select):
    # Drop NaN values
    df_clean = dataset.dropna()
    print(f"Total samples available: {len(df_clean):,}")

    # PERFORMANCE FIX: Sample data for RFE to speed up feature selection
    max_rfe_samples = 50000  # Adjust as needed (50k-100k is usually sufficient)
    if len(df_clean) > max_rfe_samples:
        print(f"Sampling {max_rfe_samples:,} rows for RFE (faster feature selection)...")
        df_rfe = df_clean.sample(n=max_rfe_samples, random_state=42)
    else:
        print("Using all data for RFE...")
        df_rfe = df_clean

    y = df_clean["lwe_thickness"]
    y_rfe = df_rfe["lwe_thickness"]

    if model == "XGBoost":
        columns_to_select = ['tp', 'sro', 'ssro', 'e', 'swvl1', 'swvl2', 'swvl3', 'swvl4','pev','t2m','evabs','lai_hv', 'lai_lv']
        x = df_clean.loc[:, columns_to_select]
        x_rfe = df_rfe.loc[:, columns_to_select]
        print(f"Features included in training: {x.columns.to_list()}")
        model = xgb.XGBRegressor(objective="reg:squarederror",n_estimators=50, random_state=42, n_jobs=-1)

    elif model == "RF":
        columns_to_select = ['tp', 'sro', 'ssro', 'e', 'swvl1', 'swvl2', 'swvl3', 'swvl4','pev','t2m','evabs','lai_hv', 'lai_lv']
        x = df_clean.loc[:, columns_to_select]
        x_rfe = df_rfe.loc[:, columns_to_select]
        print(f"Features included in training: {x.columns.to_list()}")
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    else:
        print("Invalid model input!")
        print("Please enter 'XGBoost' or 'RF'.")
    
    # RFE Classification (on sampled data for speed)
    print(f"\nRunning RFE on {len(df_rfe):,} samples...")
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select) 
    rfe.fit(x_rfe, y_rfe)

    # Print selected features from RFE
    selected_features = x.columns[rfe.support_].tolist()

    print("\n RFE Results!")
    print(f"{Fore.GREEN} The {n_features_to_select} Best Features are: {Fore.RESET}")
    print(f"{Fore.GREEN}{selected_features}{Fore.RESET}")
    return rfe, selected_features, x