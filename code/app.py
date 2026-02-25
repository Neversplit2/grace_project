import streamlit as st
import main_4_app as m4p
import vis_4_app as v4p
import plotly.graph_objects as go

st.set_page_config(
    page_title="GRACE LWE Predictor", 
    page_icon="🛰️", 
    layout="wide", # This makes it a full-screen dashboard!
    initial_sidebar_state="expanded"
)

st.title("Grace LWE project")

# --- ADD THIS WELCOME SECTION ---
st.markdown("### 🌍 Welcome to the Spatial Prediction Engine")
st.info("👈 **Start by configuring your parameters in the sidebar, then run the pipeline.**")

# Create 3 professional metric cards to fill the empty space
col1, col2, col3 = st.columns(3)
col1.metric(label="Target Variable", value="LWE Thickness", delta="GRACE Satellite")
col2.metric(label="Climate Predictors", value="ERA5 Reanalysis", delta="12+ Features")
col3.metric(label="Analytics Engine", value="Standby", delta="Awaiting Input", delta_color="off")

st.divider()
# --------------------------------

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.title("⚙️Control Panel")

    with st.expander("📍 1. Area of Interest", expanded=False):
        basin_name = st.text_input("Basin/Region Name", value="Test Region", help="This name will appear on your map titles.")
        
        st.markdown("<small>**Latitude Bounds**</small>", unsafe_allow_html=True)
        col_lat1, col_lat2 = st.columns(2)
        with col_lat1:
            lat_min = st.number_input("Min", value=-17.0, step=0.5, format="%.1f", key="lat_min")
        with col_lat2:
            lat_max = st.number_input("Max", value=5.0, step=0.5, format="%.1f", key="lat_max")
            
        st.markdown("<small>**Longitude Bounds**</small>", unsafe_allow_html=True)
        col_lon1, col_lon2 = st.columns(2)
        with col_lon1:
            lon_min = st.number_input("Min", value=-80.0, step=0.5, format="%.1f", key="lon_min")
        with col_lon2:
            lon_max = st.number_input("Max", value=-50.0, step=0.5, format="%.1f", key="lon_max")

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
# --- INTERACTIVE 3D GLOBE ---
st.markdown("### 🌐 Area of Interest Overview")

# 1. Coordinates for the red box
lons = [lon_min, lon_min, lon_max, lon_max, lon_min]
lats = [lat_min, lat_max, lat_max, lat_min, lat_min]

center_lon = (lon_min + lon_max) / 2
center_lat = (lat_min + lat_max) / 2

# 2. Build the Plotly Figure
fig = go.Figure(go.Scattergeo(
    lon=lons,
    lat=lats,
    mode='lines+markers',
    line=dict(width=4, color='red'),   # Standard red line
    marker=dict(size=8, color='red')   # Standard red dots
))

# 3. Style the globe
fig.update_geos(
    projection_type="orthographic",
    showcoastlines=True, coastlinecolor="#FAFAFA",
    showland=True, landcolor="#262730",
    showocean=True, oceancolor="#0E1117",
    projection_rotation=dict(lon=center_lon, lat=center_lat, roll=0)
)

# 4. Layout (REMOVED transparent backgrounds to prevent collapsing)
fig.update_layout(
    height=500,
    margin={"r":0,"t":0,"l":0,"b":0},
    paper_bgcolor="#0E1117", # Paints the background the exact same dark color
    plot_bgcolor="#0E1117"
)

# 5. Display
st.plotly_chart(fig, use_container_width=True)
st.divider()

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
            
