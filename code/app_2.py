import streamlit as st
import main_4_app as m4p
import vis_4_app as v4p
import plotly.graph_objects as go
import math
import numpy as np
import requests
import data_processing as dpr

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="GRACE Spatial Engine", 
    page_icon="🛰️", 
    layout="wide", 
)

# --- SCI-FI MAIN PAGE HEADER ---
#In order to change the default streamlit's fonts and dispay i am using st.markdown 
# unsafe_allow_html = True: Allows streamlit to trust HTML code 
#h1 refers to Heading 1, all the changes are made inside the style = '...'
#p refers to paragraph 
#hr draws a line across the screen (Horizontal Rule)
st.markdown("""
    <h1 style='text-align: center; color: #00E5FF; font-family: monospace; letter-spacing: 2px;'> 
         GRACE LWE SPATIAL ENGINE
    </h1>
    <p style='text-align: center; color: #8892B0; font-size: 1.1rem; font-family: monospace;'>
        SYSTEM DIRECTIVE: NEVERSPLIT | SECURE LINK: ANASTRIA-LAB
    </p>
    <hr style='border: 1px solid rgba(0, 229, 255, 0.3); margin-top: 10px; margin-bottom: 25px;'>
""", unsafe_allow_html = True)

# --- CUSTOM CSS: SCI-FI BUTTONS ---
st.markdown("""
    <style>
/* MAIN PAGE */
       
             /* col1, col2 */
    /* Target the small label text (Primary Target & Climate Predictors) */
    [data-testid="stMetricLabel"] * {
        color: #A0AEC0 !important; /* The Muted Blue-Grey color */
        text-shadow: 0 0 10px rgba(160, 174, 192, 0.5) !important; /* Matching soft ghost glow */
        font-size: 15px !important; /* 20px might be too huge for a label, try 16px first! */
    }
    
    /* Target the main value text (LWE Thickness & ERA5 Reanalysis) */
    [data-testid="stMetricValue"] {
        font-family: "Courier New", Courier, monospace !important; /* The Terminal/Tech look */
        font-weight: 600 !important; /* Makes it bold */
        letter-spacing: 1px !important; /* Spreads the letters out slightly for a cleaner look */
    }       
    /* Target the text inside the Tab buttons */
    button[data-baseweb="tab"] p {
        font-size: 20px !important;
        font-weight: bold !important;
        font-family: 'Courier New', monospace !important;
        transition: all 0.3s ease !important;
    }

    /* Optional: Make the active tab glow cyan to match your buttons */
    button[aria-selected="true"] p {
        color: #00E5FF !important;
        text-shadow: 0 0 10px rgba(0, 229, 255, 0.5) !important;
    }
            
  /* Change the animated underline color for the active tab */
    div[data-baseweb="tab_highlight"] {
        background-color: #00E5FF !important; /* Your Neon Cyan */
        /* Optional: Add a subtle glow to the line itself */
        box-shadow: 0 0 10px rgba(0, 229, 255, 0.5) !important; 
    }

/* BUTTONS */
            /* TAB2 */
    /* 1. Primary Button Styling (The "Data Prep" Button) */
    button[kind="primary"] {
        background-color: transparent !important;
        border: 2px solid #00E5FF !important;
        color: #00E5FF !important;
        border-radius: 6px !important;
        box-shadow: 0 0 10px rgba(0, 229, 255, 0.3) !important;
        transition: all 0.3s ease-in-out !important;
        font-family: monospace !important;
        font-weight: bold !important;
        letter-spacing: 1px !important;
    }
    
    /* Hover Effect for Primary Button (Glows bright cyan and inverts colors!) */
    button[kind="primary"]:hover {
        background-color: #00E5FF !important;
        color: #0E1117 !important; 
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.8) !important;
        transform: scale(1.02) !important;
    }
  
    /* 2. Secondary Button Styling (The "RFE" Button) */
    button[kind="secondary"] {
        background-color: transparent !important;
        border: 1px solid #8892B0 !important;
        color: #8892B0 !important;
        border-radius: 6px !important;
        transition: all 0.3s ease-in-out !important;
        font-family: monospace !important;
    }
    
    /* Hover Effect for Secondary Button (Lights up cyan when hovered) */
    button[kind="secondary"]:hover {
        border: 1px solid #00E5FF !important;
        color: #00E5FF !important;
        box-shadow: 0 0 10px rgba(0, 229, 255, 0.2) !important;
    }
            
    </style>
""", unsafe_allow_html=True)

# --- SYSTEM STATUS METRICS ---
col1, col2, col3 = st.columns(3)

with col1:
    # 1. We just display the metric without the Streamlit delta
    st.metric("Primary Target", "LWE Thickness")
    
    # 2. We inject a custom clickable pill that mimics the Streamlit delta style perfectly
    #border-radius: 8px; thats how we change how sharp is the corners of the rectangle button 
    st.markdown("""
        <a href="https://grace.jpl.nasa.gov/data/get-data/jpl_global_mascons/" target="_blank" style="
            text-decoration: none;
            color: rgb(61, 213, 109); 
            background-color: rgba(61, 213, 109, 0.2); 
            padding: 2px 8px;
            border-radius: 0px 10px 0px 10px;
            font-size: 14px;
            font-weight: 500;
            display: inline-block;
            margin-top: -35px; /* Pulls it tightly up under the text like a real metric */
            transition: opacity 0.2s ease-in-out;
        " onmouseover="this.style.opacity='0.7'" onmouseout="this.style.opacity='1'">
            📡JPL GRACE Satellite Downlink
        </a>
    """, unsafe_allow_html=True)
    st.markdown("""
        <a href="https://grace.jpl.nasa.gov/data/get-data/jpl_global_mascons/" target="_blank" style="
            text-decoration: none;
            color: rgb(61, 213, 109); 
            background-color: rgba(61, 213, 109, 0.2); 
            padding: 2px 8px;
            border-radius: 10px 0px 10px 0px;
            font-size: 14px;
            font-weight: 500;
            display: inline-block;
            margin-top: -15px; /* Pulls it tightly up under the text like a real metric */
            transition: opacity 0.2s ease-in-out;
        " onmouseover="this.style.opacity='0.7'" onmouseout="this.style.opacity='1'">
            📡CSR GRACE Satellite Downlink
        </a>
    """, unsafe_allow_html=True)

with col2:
    st.metric("Climate Predictors", "ERA5 Reanalysis")

    st.markdown("""
        <a href="https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means?tab=download" target="_blank" style="
            text-decoration: none;
           color: rgb(61, 213, 109); 
            background-color: rgba(61, 213, 109, 0.2); 
            padding: 2px 8px;
            border-radius: 10px 0px 10px 0px;
            font-size: 14px;
            font-weight: 500;
            display: inline-block;
            margin-top: -45px; /* Pulls it tightly up under the text like a real metric */
            transition: opacity 0.2s ease-in-out;
        " onmouseover="this.style.opacity='0.7'" onmouseout="this.style.opacity='1'">
            ☁️ERA5-Land Downlink
        </a>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# --- CREATE THE TABS ---
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "⚙️ 1. Setup & Area of interest", 
    "🧠 2. Data Processing", 
    "🦾 3. Model Training",
    "🗺️ 4. Results - Maps"
])

# ==========================================
# TAB 1: SETUP & COORDINATES
# ==========================================
with tab1:
    st.header("Define Area of Interest")

    # We create two columns: Left for inputs, Right for the Globe!
    col_input, col_globe = st.columns([1, 1.5]) 
    
    #Everything under with col_input 
    with col_input:
        
        st.markdown("<small style = 'text-align: center; color: #8892B0; font-size: 1rem; letter-spacing: 2px;'>"
        "**Latitude Bounds**"
        "</small>", unsafe_allow_html=True)

        # Split col_input in 2 extra columns
        col_lat1, col_lat2 = st.columns(2)
        # key = "lat_min" : forces Streamlit to permanently remember whatever number the user types in here
        with col_lat1: lat_min = st.number_input("Min", value=-17.0, step=0.25, format="%.2f", key="lat_min")
        with col_lat2: lat_max = st.number_input("Max", value=5.0, step=0.25, format="%.2f", key="lat_max")
            
        st.markdown("<small style = 'text-align: center; color: #8892B0; font-size: 1rem; letter-spacing: 2px;'>"
        "**Longitude Bounds**"
        "</small>", unsafe_allow_html=True)

        col_lon1, col_lon2 = st.columns(2)
        with col_lon1: lon_min = st.number_input("Min", value=-80.0, step=0.25, format="%.2f", key="lon_min")
        with col_lon2: lon_max = st.number_input("Max", value=-50.0, step=0.25, format="%.2f", key="lon_max")

        basin_name = st.text_input("Basin/Region Name", value="Test Region", help="Region/Basin name will appear on your map titles.")
        grace_data = st.selectbox("GRACE Dataset", ["CSR", "JPL"], help="Chosen dataset will be used for training" )

    with col_globe:
        # --- 3D HOLOGRAM SATELLITE HUD WITH COASTLINES ---
    
        # 2. Fetch and wrap coastlines (Cached so it only downloads once and stays fast!)
        @st.cache_data
        def get_3d_coastlines():
            url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
            try:
                data = requests.get(url).json()
            except:
                return [], [], [] # Safe fallback if offline

            xs, ys, zs = [], [], []
            def add_line(coords):
                for lon, lat in coords:
                    x, y, z = dpr.get_xyz(lon, lat, radius=1.01) # Radius 1.01 hovers just above the surface
                    xs.append(x); ys.append(y); zs.append(z)
                xs.append(None); ys.append(None); zs.append(None) # Breaks the line between continents

            for feature in data.get('features', []):
                geom = feature.get('geometry')
                if not geom: continue
                if geom['type'] == 'Polygon':
                    for poly in geom['coordinates']: add_line(poly)
                elif geom['type'] == 'MultiPolygon':
                    for multipoly in geom['coordinates']:
                        for poly in multipoly: add_line(poly)
            return xs, ys, zs

        # Generate the coastline coordinates
        cx_line, cy_line, cz_line = get_3d_coastlines()

        # Calculate centers
        center_lon = (lon_min + lon_max) / 2
        center_lat = (lat_min + lat_max) / 2

        # 3. Get 3D Coordinates for corners and satellite
        x1, y1, z1 = dpr.get_xyz(lon_min, lat_min)
        x2, y2, z2 = dpr.get_xyz(lon_max, lat_min)
        x3, y3, z3 = dpr.get_xyz(lon_max, lat_max)
        x4, y4, z4 = dpr.get_xyz(lon_min, lat_max)
        
        # SATELLITE HEIGHT: radius=1.6 puts it 60% above the Earth's surface
        sx, sy, sz = dpr.get_xyz(center_lon, center_lat, radius=1.6) 

        # 4. Create the Base "Dark Matter" Earth Sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 15)
        x_sph = np.outer(np.cos(u), np.sin(v))
        y_sph = np.outer(np.sin(u), np.sin(v))
        z_sph = np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig = go.Figure()
        fig.add_trace(go.Surface(
            x=x_sph, y=y_sph, z=z_sph,
            colorscale=[[0, '#0E1117'], [1, '#1A1C23']],
            opacity=1.0, showscale=False, hoverinfo='skip'
        ))

        # 5. Add the Wireframe Coastlines
        if cx_line:
            fig.add_trace(go.Scatter3d(
                x=cx_line, y=cy_line, z=cz_line,
                mode='lines', line=dict(color='#4A5568', width=1.5), # Subtle slate grey coastlines
                hoverinfo='skip'
            ))

        # 6. Target Region Base (Solid Neon Cyan on the surface)
        fig.add_trace(go.Scatter3d(
            x=[x1, x2, x3, x4, x1], y=[y1, y2, y3, y4, y1], z=[z1, z2, z3, z4, z1],
            mode='lines', line=dict(color='#00E5FF', width=5), hoverinfo='skip'
        ))

        # 7. Hologram Pyramid Lines (Faded dashed lines from Satellite to Corners)
        for cx, cy, cz in [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3), (x4,y4,z4)]:
            fig.add_trace(go.Scatter3d(
                x=[sx, cx], y=[sy, cy], z=[sz, cz],
                mode='lines', line=dict(color='rgba(0, 229, 255, 0.4)', width=3, dash='dash'),
                hoverinfo='skip'
            ))

        # 8. The GRACE Satellite Origin Point
        fig.add_trace(go.Scatter3d(
            x=[sx], y=[sy], z=[sz],
            mode='text', # We drop the 'markers' entirely so the white diamond vanishes
            text=['🛰️<br>'], # Satellite icon with the text directly underneath it
            textposition='middle center',
            textfont=dict(color='#00E5FF', size=32), # Massive size so it acts like a real map marker
            hoverinfo='skip'
        ))

        # 9. Layout & Camera Engine
        fig.update_layout(
            height=450, margin=dict(r=0, t=0, l=0, b=0),
            paper_bgcolor="#0E1117", showlegend=False,
            scene=dict(
                xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                camera=dict(eye=dict(x=sx*0.8, y=sy*0.8, z=sz*0.8)) # Camera tracks the target
            )
        )
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: FEATURE ENGINEERING
# ==========================================
with tab2:
    st.header("Recursive Feature Elimination (RFE)")
    st.write("Run the data preparation pipeline to rank the best ERA5 features.")

    if st.button("Data Prep", type="primary"):
        try:
            with st.spinner("Preparing Data... (Downloading and merging)"):
                df_ERA, df_CSR, ds_ERA_sliced, ds_CSR_sliced, merged, df_CSR_on_ERA_grid = m4p.pipe_data_prp(
                    grace_data, lat_min, lat_max, lon_min, lon_max
                )
                st.session_state['ds_ERA_sliced'] = ds_ERA_sliced
                st.session_state['ds_CSR_sliced'] = ds_CSR_sliced
                st.session_state['merged'] = merged
                st.session_state['df_ERA'] = df_ERA
                st.session_state['df_CSR_on_ERA_grid'] = df_CSR_on_ERA_grid
                st.success("Data Prepared Successfully!")
        except Exception as e:
                    st.error(f"An error occurred: {e}")

    col_input1, col_input2 = st.columns([1, 1.5])
    with col_input1:
        col_rfe1, col_rfe2 = st.columns(2)

        with col_rfe1:
            model_RFE = st.selectbox("Select Model for RFE", ["XGBoost", "Random Forest"])
        with col_rfe2:
            n_features = st.number_input("Number of features to select", min_value=1, value=5, step=1)


    if st.button("RFE", type="primary"):
        try:

            if st.session_state['merged'] is not None:
                with st.spinner(f"Running RFE with {model_RFE}..."):
                    rfe, selected_features, x = m4p.pipe_RFE(st.session_state['merged'], model_RFE, int(n_features))
                    st.success(f"RFE Complete! Found {len(selected_features)} best features.")
                    st.subheader("Feature Ranking Results")
                    fig_rfe = v4p.rfe_plot(rfe, x)
                    st.pyplot(fig_rfe)
            else:
                st.error("Merge failed. Data is empty.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# ==========================================
# TAB 3: PREDICTION MAPS
# ==========================================
with tab4:
    st.header("Generate Spatial Predictions")
    st.write("Upload a pre-trained model and configure the temporal settings for the final map outputs.")
    
    col_map1, col_map2, col_map3 = st.columns(3)
    with col_map1:
        map_year = st.number_input("Year for Map", min_value=2000, max_value=2023, value=2010)
    with col_map2:
        map_month = st.number_input("Month for Map", min_value=1, max_value=12, value=5)
    with col_map3:
        era_var = st.selectbox("ERA5 Variable to Plot", ["t2m", "tp", "e","pev","ssro", "sro", "evabs","swvl1", "swvl2", "swvl3", "swvl4", "lai_hv", "lai_lv"]) 
    
    uploaded_model = st.file_uploader("Upload Pre-trained Model (.pkl)", type=["pkl"])

    if st.button("🗺️ Generate Maps", type="primary"):
        if 'ds_ERA_sliced' not in st.session_state or 'merged' not in st.session_state:
            st.warning("⚠️ Please run 'Data Prep & RFE' in Tab 2 first to load the datasets!")
        elif uploaded_model is None:
            st.warning("⚠️ Please upload a .pkl model to generate prediction maps!")
        else:
            with st.spinner("Drawing Maps..."):
                col_plot1, col_plot2 = st.columns([1, 2])
                with col_plot1:
                    st.subheader("ERA5 Data")
                    era_fig = v4p.ERA_plot(st.session_state['ds_ERA_sliced'], map_year, map_month, era_var, basin_name)
                    st.pyplot(era_fig)
                
                st.divider()
                st.subheader("Model Predictions vs GRACE Observations")
                csr_fig = v4p.CSR_plot(
                    model=uploaded_model, 
                    year=map_year, 
                    month=map_month, 
                    dataset_CSR=st.session_state['ds_CSR_sliced'], 
                    dataset_CSR2=st.session_state['df_CSR_on_ERA_grid'], 
                    dataset_ERA=st.session_state['df_ERA'], 
                    var_to_plot='lwe_thickness', 
                    basin_name=basin_name
                ) 
                st.pyplot(csr_fig)