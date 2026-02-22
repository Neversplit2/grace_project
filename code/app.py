import streamlit as st
import main_4_app as m4p
import vis_4_app as v4p

st.title("Grace LWE project")

st.write("Welcome to the GRACE LWE Prediction interface!")

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("1. Area of Interest")
    lat_min = st.number_input("Latitude Min", value=-17.0) # Used dummy defaults for speed
    lat_max = st.number_input("Latitude Max", value=5.0)
    lon_min = st.number_input("Longitude Min", value=-80.0)
    lon_max = st.number_input("Longitude Max", value=-50.0)

    st.header("2. Dataset")
    grace_data = st.selectbox("GRACE Dataset", ["CSR", "JPL"])

    st.header("3. RFE Settings")
    model_RFE = st.selectbox("Select Model for RFE", ["XGBoost", "RF"])
    n_features = st.number_input("Number of features to select", min_value=1, value=5, step=1)

    st.header("4. Map Settings")
    map_year = st.number_input("Year for Map", min_value=2000, max_value=2023, value=2010)
    map_month = st.number_input("Month for Map", min_value=1, max_value=12, value=5)
    era_var = st.selectbox("ERA5 Variable to Plot", ["t2m", "tp", "e","pev","ssro", "sro", "evabs","swvl1",
                         "swvl2", "swvl3", "swvl4", "lai_hv", "lai_lv"]) 
    basin_name = st.text_input("Basin/Region Name", value="Test Region")

    st.header("5. Load Pre-trained Model")
    st.info("Upload a .pkl file to skip training and generate maps immediately.")
    uploaded_model = st.file_uploader("Upload Model (.pkl)", type=["pkl"])


# --- MAIN EXECUTION ---
# When the user clicks this button, the magic happens
if st.button("🚀 Run Data Prep & RFE", type="primary"):
    
    # We use a try/except block so if your data code crashes, Streamlit shows the error nicely on screen
    try:
        # Step 1: Data Prep
        with st.spinner("Preparing Data... (Downloading and merging)"):
            df_ERA, df_CSR, ds_ERA_sliced, ds_CSR_sliced, merged, df_CSR_on_ERA_grid = m4p.pipe_data_prp(
                grace_data, lat_min, lat_max, lon_min, lon_max
            )
            # --- SAVE TO MEMORY BANK ---
            st.session_state['ds_ERA_sliced'] = ds_ERA_sliced
            st.session_state['ds_CSR_sliced'] = ds_CSR_sliced
            st.session_state['merged'] = merged
            st.session_state['df_ERA'] = df_ERA
            st.session_state['df_CSR_on_ERA_grid'] = df_CSR_on_ERA_grid
            st.success("Data Prepared Successfully!")

        # Step 2: RFE
        if merged is not None:
            with st.spinner(f"Running RFE with {model_RFE}..."):
                # Call main2.py
                rfe, selected_features, x = m4p.pipe_RFE(merged, model_RFE, int(n_features))
                st.success(f"RFE Complete! Found {len(selected_features)} best features.")
                
                # Call vis2.py
                st.subheader("Feature Ranking Results")
                fig = v4p.rfe_plot(rfe, x)
                
                # Show on webpage!
                st.pyplot(fig)
        else:
            st.error("Merge failed. Data is empty.")

    except Exception as e:
        st.error(f"An error occurred during the test: {e}")

st.divider() # Adds a nice visual line

if st.button("🗺️ Generate Maps"):
    if 'ds_ERA_sliced' not in st.session_state or 'merged' not in st.session_state:
        st.warning("⚠️ Please run 'Data Prep & RFE' first to load the datasets!")
        
    # 2. Check if they actually uploaded a model!
    elif uploaded_model is None:
        st.warning("⚠️ Please upload a .pkl model in the sidebar to generate prediction maps!")
        
    else:
        with st.spinner("Drawing Maps..."):
            
            # We display the ERA5 map first in its own column
            col1, col2 = st.columns([1, 2]) # Makes the right side wider if needed
            
            with col1:
                st.subheader("ERA5 Data")
                era_fig = v4p.ERA_plot(st.session_state['ds_ERA_sliced'], map_year, map_month, era_var, basin_name)
                st.pyplot(era_fig)
                
            # Then we display the massive 3-panel CSR plot below it!
            st.divider()
            st.subheader("Model Predictions vs GRACE Observations")
            
            # Call your CSR_plot function!
            # Notice we are passing `uploaded_model` directly to the `model` argument
            csr_fig = v4p.CSR_plot(
                model=uploaded_model, 
                year=map_year, 
                month=map_month, 
                dataset_CSR=st.session_state['ds_CSR_sliced'], 
                dataset_CSR2=st.session_state['df_CSR_on_ERA_grid'], # Assuming your merged df is CSR2
                dataset_ERA=st.session_state['df_ERA'], 
                var_to_plot='lwe_thickness', # Or whatever your exact variable name is
                basin_name=basin_name
            ) 
            
            st.pyplot(csr_fig)
            
