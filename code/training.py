#Import libraries
import configuration_settings as cs
import data_processing as dpr
from colorama import Fore
import xgboost as xgb
import joblib, sys
import numpy as np
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

#training
# selected_features = selected_features, x = x, dataset = merged
def data_4_train(selected_features, x, dataset):
    x_final = x[selected_features]
    print(f"Training will use: \n {x_final.head(1)}")
    target = "lwe_thickness"
    extra_features = ["year", "month", "lon", "lat"]
    base_features = list(x_final.columns)
    # Final feature list
    features = base_features + extra_features
    features = [col for col in features if col != target]
    # Define X and y
    X = dataset[features]
    y = dataset[target]

    # Split into training and validation datasets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )
    return X_train, X_test, y_train, y_test

def XGBoost_tuner(X_train, y_train):
    print("  XGBoost Tuning...")
    param_grid = {
    'n_estimators': [200, 400],
    'max_depth': [6, 10, 15],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.6, 0.8, 1],
    'colsample_bytree': [0.6, 0.8, 1],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [0.5, 1.0]
    }

    final_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    tuner = RandomizedSearchCV(
        final_model,
        param_distributions=param_grid,
        n_iter=12,                     
        cv=kf,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,                      
        verbose=2,
        random_state=42
    )

    tuner.fit(X_train, y_train)
    best_params = tuner.best_params_
    print(" Best XGBoost Parameters:", best_params)
    return best_params

def XGBoost_train(X_train, y_train):
    best_params = XGBoost_tuner(X_train, y_train)
    best_model = xgb.XGBRegressor(
        **best_params,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    best_model.fit(X_train, y_train)
    
    return best_model

def RF_tuner(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100],        
        'max_depth': [6, 8],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt']
    }

    final_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)  # 3-FOLD → faster
    tuner = RandomizedSearchCV(
        final_model,
        param_distributions=param_grid,
        n_iter=8,                     
        cv=kf,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=2,
        random_state=42
    )

    # Fit tuner
    tuner.fit(X_train, y_train)
    best_params = tuner.best_params_
    print("Best Parameters:", best_params)
    return best_params
def RF_train(X_train,y_train):
    best_params = RF_tuner(X_train,y_train)
    best_model = RandomForestRegressor(
        **best_params,
        random_state=42,
        n_jobs=-1
    )
    best_model.fit(X_train, y_train)
    return best_model

#learning curves
#X_train, X_test, y_train, y_test ta exw ypologisei sto main
def data_4_curves(X_train, X_test, y_train, y_test):
    
    max_tuning_samples = 100000 #adjust as needed eg 50000

    if len(X_train) > max_tuning_samples:
        X_tune = X_train.sample(n=max_tuning_samples, random_state=42)
        y_tune = y_train.loc[X_tune.index]
    else:
        X_tune = X_train
        y_tune = y_train

    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_tune, y_tune, test_size=0.2, random_state=42
    )
    return X_train_sub, X_val, y_train_sub, y_val

def XGBoost_curves(X_train, X_test, y_train, y_test):

    X_train_sub, X_val, y_train_sub, y_val = data_4_curves(X_train, X_test, y_train, y_test)

    best_params = XGBoost_tuner(X_train_sub, y_train_sub)
    
    max_trees = best_params['n_estimators']
    curve_steps = np.unique(
        np.linspace(10, max_trees, num=min(max_trees // 10, 20), dtype=int)
    )
    # Lists for storing results
    train_mae_list = []
    val_mae_list = []

    print(f"\nCalculating learning curves for XGBoost (for {max_trees} trees)")

    for n_trees in curve_steps:
        
        # Create a temporary model for this step
        # We use the best_params except n_estimators (which we set manually here)
        # Filter parameters to remove 'n_estimators'
        params_without_n = {k: v for k, v in best_params.items() if k != 'n_estimators'}
        
        xgb_curves = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,             
            n_estimators=int(n_trees), 
            **params_without_n     
        )
        
        # Train on the subset (sub)
        xgb_curves.fit(X_train_sub, y_train_sub)

        # Predictions
        y_train_pred = xgb_curves.predict(X_train_sub)
        y_val_pred = xgb_curves.predict(X_val)

        # Calculating errors
        train_mae = mean_absolute_error(y_train_sub, y_train_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)

        train_mae_list.append(train_mae)
        val_mae_list.append(val_mae)

        print(f"Trees: {n_trees:4d} | Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}")
        
    return curve_steps, train_mae_list, val_mae_list