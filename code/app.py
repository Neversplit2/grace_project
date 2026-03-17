import streamlit as st
import main_4_app as m4p
import vis_4_app as v4p
import plotly.graph_objects as go
import numpy as np
import requests, joblib
import data_processing as dpr
import time, io

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="GRACE Spatial Engine", 
    page_icon="🛰️", 
    layout="wide", 
)

# --- SCI-FI MAIN PAGE HEADER ---
#In order to change the default streamlit's fonts and dispay i am using st.markdown 

st.markdown("""
    <h1 style='text-align: center; color: #00E5FF; font-family: monospace; letter-spacing: 2px;'> 
         DOWNSCALING ENGINE FOR GRACE & GRACE-FO LWE DATA
    </h1>
    <div style="text-align: center; color: #626A7F; font-family: monospace; font-size: 14px; letter-spacing: 1px; margin-bottom: 30px;">
        ENGINEERED & DESIGNED BY: 
        <div class="cyber-tooltip">
            NEVERSPLIT
            <span class="tooltip-text">
                <span style="color:#00E5FF; font-weight:bold;">[CREDENTIAL: NEVERSPLIT]</span><br>
                titlos<br>
                <i>personal info</i>
            </span>
        </div> 
        &nbsp;| 
        <div class="cyber-tooltip">
            ANASTRIA-LAB
            <span class="tooltip-text">
                <span style="color:#00E5FF; font-weight:bold;">[CREDENTIAL: ANASTRIA-LAB]</span><br>
                titlos (supervisor oti thes bale).<br>
                <i>personal info </i>
            </span>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- CUSTOM CSS: SCI-FI BUTTONS ---
st.markdown("""
    <style>
/* MAIN PAGE */
            
            /* Header */
    /* =========================================
       3. HOVER TOOLTIP STYLING
       ========================================= */
       
    /* The clickable text itself (The trigger) */
    .cyber-tooltip {
        position: relative;
        display: inline-block;
        color: #A0AEC0; /* Matches your muted sub-header text */
        cursor: crosshair; /* Cool targeting computer mouse pointer! */
        border-bottom: 1px dotted #00E5FF; /* Subtle cyan dotted line */
        transition: color 0.2s ease-in-out;
    }
    
    /* Text lights up cyan when the mouse hits it */
    .cyber-tooltip:hover {
        color: #00E5FF;
    }

    /* The Hidden Pop-Up Box (The payload) */
    .cyber-tooltip .tooltip-text {
        visibility: hidden;
        width: 260px;
        background-color: #0E1117; /* Streamlit Dark background */
        color: #A0AEC0;
        text-align: left;
        border: 1px solid #00E5FF;
        border-radius: 0px 10px 0px 10px;
        padding: 12px;
        position: absolute;
        z-index: 999;
        top: 150%; /* Pops up BELOW the text */
        left: 50%;
        margin-left: -130px; /* Centers the box perfectly */
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.2);
        opacity: 0;
        transition: opacity 0.3s, transform 0.3s;
        transform: translateY(10px); /* Starts lower for a slide-up animation */
        font-family: monospace;
        font-size: 12px;
        line-height: 1.5;
    }

    /* Show the Pop-Up on Hover */
    .cyber-tooltip:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
        transform: translateY(0px); /* Slides smoothly into place */
    }
       
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

/* TAB 3 (TRAINING) */
        /* col_1 */
    /* HELP */
/* The outer container of the tooltip */
    div[data-baseweb="tooltip"] {
        background-color: #0E1117 !important; /* Streamlit dark mode */
        border: 1px solid #00E5FF !important; /* Neon cyan border */
        border-radius: 20px 0px 20px 0px !important;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.2) !important; /* Subtle cyan glow */
        margin-left: 610px !important; /* Move box horizontal */
    }
    
    /* Remove Streamlit's default inner background color */
    div[data-baseweb="tooltip"] > div {
        background-color: transparent !important;
    }
    
    /* The text inside the tooltip */
    div[data-baseweb="tooltip"] .stMarkdown p, 
    div[data-baseweb="tooltip"] .stMarkdown li {
        color: #A0AEC0 !important; /* Muted blue-grey */
        font-family: monospace !important; /* Terminal font */
        font-size: 12px !important; /* Matches Neversplit exactly */
        line-height: 1.5 !important;
    }
    
    /* Make the bold text act like your [CREDENTIAL] tags */
    div[data-baseweb="tooltip"] .stMarkdown strong {
        color: #00E5FF !important; 
        font-weight: bold !important;
        
    }
    /* TAB4 */
        /* number_input text fonts*/
            
    /* Target the input field text */
    input[type="number"] {        
        font-family: monospace !important;
        color: #00E5FF !important; 
        font-size: 1rem !important; 
        letter-spacing: 2px;
    }
    /* Change +- color */
        .stNumberInput button {
            color: #00E5FF !important;
        }
            
    /* Change the color when you hover over them */
        .stNumberInput button:hover {
            background-color: #262730 !important;
        }
    /* 1. Force the default state to your slate-dark color */
        .stNumberInput button {
            background-color: #262730 !important;
            color: #00E5FF !important; /* Keeping your cyan icon color */
        }

    /* 2. KILL the blue box (Focus, Active, and Hover) */
    .stNumberInput button:focus, 
    .stNumberInput button:active, 
    .stNumberInput button:hover,
    .stNumberInput button:focus-visible {
        background-color: #262730 !important; /* Stays dark */
        color: #00E5FF !important;            /* Icon stays cyan */
        outline: none !important;             /* Removes the focus ring */
        box-shadow: none !important;          /* Removes any glow */
    }

    /* 3. Specifically target the internal icon color to prevent theme overrides */
    .stNumberInput button p {
        color: #00E5FF !important;
    }
    
    /* Selectbox text color*/ 
   /* 1. Target the text inside the selectbox (the current selection) */
    div[data-baseweb="select"] div {
        font-family: monospace !important;
        color: #00E5FF !important;
        font-size: 1rem !important;
        letter-spacing: 2px !important;
    }

    /* 2. Target the text of the options inside the dropdown menu */
    div[data-baseweb="popover"] li {
        font-family: monospace !important;
        color: #00E5FF !important;
        font-size: 1rem !important;
        letter-spacing: 2px !important;
    }

    /* 3. Ensure the placeholder text (if any) matches the font but stays muted */
    div[data-baseweb="select"] [data-testid="stWidgetLabel"] {
        font-family: monospace !important;
    }

    </style>
""", unsafe_allow_html=True)

#Live intro text
ticker_text = "WELCOME TO THE GRACE & GRACE-FO DOWNSCALING ENGINE." \
            " • TAB 1: DEFINE AREA - BASIN OF INTEREST AND VALIDATE BOUNDARIES WITH THE LIVE NAVIGATION GLOBE." \
            " • TAB 2: RUN DATA PREPARATION AND RFE ANALYSIS. " \
            "• TAB 3: TRAIN YOUR MODEL (OPTIONAL: FEEL FREE TO SKIP IF YOU ALREADY HAVE A PRE-TRAINED MODEL READY TO USE). " \
            "• TAB 4: THIS IS WHERE MAGIC HAPPENS.. CREATE ANY MAP YOU CAN IMAGINE!" \
            " • TAB 5: PERFORM STATISTICAL ANALYSIS AND EVALUATE YOUR MODEL’S PERFORMANCE."

st.markdown(f"""
    <style>
    .ticker-wrapper {{
        width: 100%;
        overflow: hidden;
        background-color: transparent; 
        border-top: 1px solid rgba(0, 229, 255, 0.3);
        border-bottom: 1px solid rgba(0, 229, 255, 0.3);
        padding: 8px 0;
        margin-bottom: 30px; /* Space before metrics start */
    }}

    .ticker-text {{
        display: inline-block;
        white-space: nowrap;
        padding-left: 100%;
        animation: marquee 70s linear infinite; 
        font-family: 'monospace';
        color: #00E5FF;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0 0 5px rgba(0, 229, 255, 0.5);
    }}

    @keyframes marquee {{
        0%   {{ transform: translate(0, 0); }}
        100% {{ transform: translate(-100%, 0); }}
    }}
    </style>

    <div class="ticker-wrapper">
        <div class="ticker-text">
            {ticker_text}
        </div>
    </div>
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
        <a href="https://www2.csr.utexas.edu/grace/RL06_mascons.html" target="_blank" style="
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
    """, unsafe_allow_html=True)\

with col3:
    st.metric("Engine resolution", "0.1° (~10km)")

    st.markdown("""
        <div style="
            color: rgb(61, 213, 109);
            background-color: rgba(61, 213, 109, 0.2);
            padding: 2px 8px;
            border-radius: 10px 0px 10px 0px;
            font-size: 14px;
            font-weight: 500;
            display: inline-block;
            margin-top: -45px;
            transition: opacity 0.2s ease-in-out;
        ">
            📅 Data Range: 2002 - 2024
        </div>
    """, unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# --- CREATE THE TABS ---
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⚙️ 1. Setup & Area of interest", 
    "🧠 2. Data Processing", 
    "🦾 3. Model Training",
    "🗺️ 4. Maps",
    "📊 5. Statistical Analysis"
])


with tab1:
    st.markdown("<h3 style = 'color: #00E5FF; font-family: monospace; font-weight: 400 !important; letter-spacing: 2px;'>"
    " Define Area Of Interest"
    "</h3>", unsafe_allow_html=True)

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

with tab2:
    st.markdown("<h3 style = 'color: #00E5FF; font-family: monospace; letter-spacing: 2px;'>"
        " Recursive Feature Elimination (RFE) "
        "</h3>", unsafe_allow_html=True)
    #st.write("Run the data preparation pipeline to rank the best ERA5 features.")
    st.markdown("<small style = 'text-align: center; color: #8892B0; font-size: 1rem; letter-spacing: 2px;'>"
        "**Run the data preparation pipeline to rank the best ERA5 features.**"
        "</small>", unsafe_allow_html=True)

    col_1, col_2 = st.columns(2)

    display_screen = col_2.empty()
    with col_1:
        if st.button("Data Prep", type="primary"):

            # Instantly show the terminal on the right before doing any heavy lifting
            terminal_ui = """
            <style>
                /* MAC traffic light look*/
                .term-window {
                    background-color: #121212;
                    border-radius: 10px;
                    border: 1px solid #333;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
                    font-family: 'Fira Code', 'Consolas', 'Courier New', monospace;
                    color: #00E5FF;
                    overflow: hidden;
                    height: 250px;
                    margin-top: -55px; /* Move the window */
                }
                .term-header {
                    background-color: #2b2b2b;
                    padding: 10px;
                    display: flex;
                    align-items: center;
                    border-bottom: 1px solid #111;
                }
                .term-button {
                    width: 12px; height: 12px; border-radius: 50%; margin-right: 8px;
                }
                .close { background-color: #ff5f56; }
                .minimize { background-color: #ffbd2e; }
                .maximize { background-color: #27c93f; }
                .term-body { padding: 15px; font-size: 14px; line-height: 1.5; }

                /* CSS to make lines appear one by one */
                .term-line { opacity: 0; margin: 0; animation: fadeIn 0.1s forwards; }
                .delay-1 { animation-delay: 1s; color: #a3adc2; }
                .delay-2 { animation-delay: 3.2s; }
                .delay-3 { animation-delay: 7.2s; }
                .delay-4 { animation-delay: 10.0s; }
                
                @keyframes fadeIn { to { opacity: 1; } }
                
                /* Blinking cursor effect */
                .cursor-blink { animation: blinker 0.9s infinite; }
                @keyframes blinker { 50% { opacity: 0; } }
            </style>

            <div class="term-window">
                <div class="term-header">
                    <div class="term-button close"></div>
                    <div class="term-button minimize"></div>
                    <div class="term-button maximize"></div>
                </div>
            <div style="background-color: #0b0f19; padding: 20px; border-radius: 8px; 
                        border: 1px solid #1e293b; font-family: 'Courier New', monospace; 
                        color: #00E5FF; height: 250px; box-shadow: inset 0 0 10px #000000;">
                <p class="term-line delay-1">> system start data_prep pipeline</p>
                <p class="term-line delay-2">> Initializing ERA5 Data Pipeline...</p>
                <p class="term-line delay-3">> Downloading and merging spatial grids...</p>
                <p class="term-line delay-4">> Computing parameters... <span class="cursor-blink">█</span></p>
            </div>
            """
            #Whatever i throw at display_screen goes at col2 as i wanted
            display_screen.markdown(terminal_ui, unsafe_allow_html=True)

            try:
                df_ERA, df_CSR, ds_ERA_sliced, ds_CSR_sliced, merged, df_CSR_on_ERA_grid = m4p.pipe_data_prp(
                    grace_data, lat_min, lat_max, lon_min, lon_max
                )
                st.session_state['ds_ERA_sliced'] = ds_ERA_sliced
                st.session_state['ds_CSR_sliced'] = ds_CSR_sliced
                st.session_state['merged'] = merged
                st.session_state['df_ERA'] = df_ERA
                st.session_state['df_CSR_on_ERA_grid'] = df_CSR_on_ERA_grid

                # --- THE SUCCESS TERMINAL ---
                # Shows up instantly (no delays) with the final green success message
                success_terminal = """
                <style>

                    /* MAC traffic light look*/
                .term-window {
                    background-color: #121212;
                    border-radius: 10px;
                    border: 1px solid #333;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
                    font-family: 'Fira Code', 'Consolas', 'Courier New', monospace;
                    color: #00E5FF;
                    overflow: hidden;
                    height: 250px;
                    margin-top: -55px;
                }
                .term-header {
                    background-color: #2b2b2b;
                    padding: 10px;
                    display: flex;
                    align-items: center;
                    border-bottom: 1px solid #111;
                }
                .term-button {
                    width: 12px; height: 12px; border-radius: 50%; margin-right: 8px;
                }
                .close { background-color: #ff5f56; }
                .minimize { background-color: #ffbd2e; }
                .maximize { background-color: #27c93f; }
                .term-body { padding: 15px; font-size: 14px; line-height: 1.5; }

                    .term-line { opacity: 0; margin: 0; animation: fadeIn 0.1s forwards; }
                    .delay-1 { animation-delay: 3.5s; }
                    @keyframes fadeIn { to { opacity: 1; } }

                </style>
                <div class="term-window">
                <div class="term-header">
                    <div class="term-button close"></div>
                    <div class="term-button minimize"></div>
                    <div class="term-button maximize"></div>
                </div>

                <div style="background-color: #0b0f19; padding: 20px; border-radius: 8px; 
                            border: 1px solid #1e293b; font-family: 'Courier New', monospace; 
                            color: #00E5FF; height: 250px; box-shadow: inset 0 0 10px #000000;">
                    <p style="margin: 0; color: #a3adc2;">> system start data_prep pipeline</p>
                    <p style="margin: 0;">> Initializing ERA5 Data Pipeline...</p>
                    <p style="margin: 0;">> Downloading and merging spatial grids...</p>
                    <p style="margin: 0;">> Computing parameters... <span style="color: #00FF00;">Done.</span></p>
                    <p style="margin: 0; color: #00FF00;">> Data Prepared Successfully!</p>
                    <p class="term-line delay-1" style="margin: 0; color: #FF00FF;">> Ready for Feature Ranking. </p>
                </div>
                """
                
                # Overwrite the loading animation with the final success screen
                display_screen.markdown(success_terminal, unsafe_allow_html=True)

            except Exception as e:
                        st.error(f"An error occurred: {e}")

        col_rfe1, col_rfe2 = st.columns(2)

        with col_rfe1:
            model_RFE = st.selectbox("Select Model for RFE", ["XGBoost", "Random Forest"])
        with col_rfe2:
            n_features = st.number_input("Number of features to select", min_value=1, value=5, step=1)

        if st.button("RFE", type="primary"):
            try:
                if st.session_state['merged'] is not None:
                    terminal_rfe = """
                    <style>
                        /* MAC traffic light look*/
                    .term-window {
                        background-color: #121212;
                        border-radius: 10px;
                        border: 1px solid #333;
                        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
                        font-family: 'Fira Code', 'Consolas', 'Courier New', monospace;
                        color: #00E5FF;
                        overflow: hidden;
                        height: 250px;
                        margin-top: -55px;
                    }
                    .term-header {
                        background-color: #2b2b2b;
                        padding: 10px;
                        display: flex;
                        align-items: center;
                        border-bottom: 1px solid #111;
                    }
                    .term-button {
                        width: 12px; height: 12px; border-radius: 50%; margin-right: 8px;
                    }
                    .close { background-color: #ff5f56; }
                    .minimize { background-color: #ffbd2e; }
                    .maximize { background-color: #27c93f; }
                    .term-body { padding: 15px; font-size: 14px; line-height: 1.5; }

                        .term-line { opacity: 0; margin: 0; animation: fadeIn 0.1s forwards; }
                        .delay-1 { animation-delay: 1s; }
                        .delay-2 { animation-delay: 2s; }
                        @keyframes fadeIn { to { opacity: 1; } }

                    </style>
                    <div class="term-window">
                    <div class="term-header">
                        <div class="term-button close"></div>
                        <div class="term-button minimize"></div>
                        <div class="term-button maximize"></div>
                    </div>

                    <div style="background-color: #0b0f19; padding: 20px; border-radius: 8px; 
                                border: 1px solid #1e293b; font-family: 'Courier New', monospace; 
                                color: #00E5FF; height: 250px; box-shadow: inset 0 0 10px #000000;">
                        <p style="margin: 0; color: #a3adc2;">> system start RFE</p>
                        <p class="term-line delay-1" style="margin: 0; color: #FF00FF;">> Initializing RFE Ranking Algorithm...</p>
                        <p class="term-line delay-2" style="margin: 0; color: #FF00FF;">> Processing features... <span class="cursor-blink">█</span></p>
                    </div>
                    """
                    display_screen.markdown(terminal_rfe, unsafe_allow_html=True)
                                   
                    rfe, selected_features, x = m4p.pipe_RFE(st.session_state['merged'], model_RFE, int(n_features))
                    
                    terminal_rfe_done = """
                    <style>
                        /* MAC traffic light look*/
                    .term-window {
                        background-color: #121212;
                        border-radius: 10px;
                        border: 1px solid #333;
                        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
                        font-family: 'Fira Code', 'Consolas', 'Courier New', monospace;
                        color: #00E5FF;
                        overflow: hidden;
                        height: 250px;
                        margin-top: -55px;
                    }
                    .term-header {
                        background-color: #2b2b2b;
                        padding: 10px;
                        display: flex;
                        align-items: center;
                        border-bottom: 1px solid #111;
                    }
                    .term-button {
                        width: 12px; height: 12px; border-radius: 50%; margin-right: 8px;
                    }
                    .close { background-color: #ff5f56; }
                    .minimize { background-color: #ffbd2e; }
                    .maximize { background-color: #27c93f; }
                    .term-body { padding: 15px; font-size: 14px; line-height: 1.5; }

                        .term-line { opacity: 0; margin: 0; animation: fadeIn 0.1s forwards; }
                        @keyframes fadeIn { to { opacity: 1; } }

                    </style>
                    <div class="term-window">
                    <div class="term-header">
                        <div class="term-button close"></div>
                        <div class="term-button minimize"></div>
                        <div class="term-button maximize"></div>
                    </div>

                    <div style="background-color: #0b0f19; padding: 20px; border-radius: 8px; 
                                border: 1px solid #1e293b; font-family: 'Courier New', monospace; 
                                color: #00E5FF; height: 250px; box-shadow: inset 0 0 10px #000000;">
                        <p style="margin: 0; color: #a3adc2;">> system start RFE</p>
                        <p style="margin: 0; color: #FF00FF;">> Initializing RFE Ranking Algorithm...</p>
                        <p style="margin: 0; color: #FF00FF;">> Processing features... <span class="cursor-blink">█</span></p>
                        <p style="margin: 0; color: #00FF00;">> RFE Complete! Found {len(selected_features)} best features. </p>
                    </div>
                    """
                    display_screen.markdown(terminal_rfe_done, unsafe_allow_html=True)

                    # Saving variables
                    st.session_state['rfe'] = rfe
                    st.session_state['selected_features'] = selected_features
                    st.session_state['x'] = x

                    time.sleep(2)
                    display_screen.empty()

                    with display_screen.container():

                        fig_rfe = v4p.rfe_plot(rfe, x)
                        #Move plot up
                        st.markdown("""
                        <style>
                        .stPlotlyChart, .stImage, .stException {
                            margin-top: -100px; /* Adjust this value to move it further up */
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        st.pyplot(fig_rfe, use_container_width=True)
                    
                else:
                    st.error("Merge failed. Data is empty.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

with tab3:
    st.markdown("<h3 style = 'color: #00E5FF; font-family: monospace;  letter-spacing: 2px;'>"
        " Model Training Facility"
        "</h3>", unsafe_allow_html=True)
    
    col_1, col_2 = st.columns([1, 1.5])
    with col_1:
        st.markdown("<p style='color: #A0AEC0; font-family: monospace;'>Algorithm Selection</p>", unsafe_allow_html=True)
        
        model = st.selectbox("Select Model for training", ["XGBoost", "Random Forest"])
        
        if model == "XGBoost":
            variant_options = ["Light", "Heavy"]
            help_text = """
            | **XGBoost Light** (⚡ Fast) | **XGBoost Heavy** (💥 Deep Training) |
            | :--- | :--- |
            | n_estimators: [200, 500] | n_estimators: [100, 200, 500] |
            | max_depth: [6, 10] | max_depth: [10, 20, 6] |
            | learning_rate: [0.05, 0.1] | learning_rate: [0.01, 0.05, 0.1] |
            | subsample: [0.8] | subsample: [0.7, 0.8, 0.9] |
            | colsample_bytree: [0.8] | colsample_bytree: [0.7, 0.8, 1.0] |
            | reg_alpha: [0, 0.1] | reg_alpha: [0, 0.1, 0.5] |
            | reg_lambda: [0.5, 1.0] | reg_lambda: [0.1, 0.5, 1.0] |
            """
        elif model == "Random Forest":
            variant_options = ["Light", "Heavy"]
            help_text = """
            | **Random Forest Light** (⚡ Fast) | **Random Forest Heavy** (💥 Deep Training) |
            | :--- | :--- |
            | n_estimators: [200, 300] | n_estimators: [100, 200, 300] |
            | max_depth: [10, 20] | max_depth: [10, 20] |
            | min_samples_split: [2, 5] | min_samples_split: [2, 5] |
            | min_samples_leaf: [1, 2] | min_samples_leaf: [1, 2] |
            | max_features: ['sqrt'] | max_features: ['sqrt', 'log2'] |
            """

        selected_model = st.selectbox(
        "Select Engine Variant", 
        variant_options,
        help=help_text)

        st.markdown("<br>", unsafe_allow_html=True) # A little breathing room
        
        if st.button("Train Model", type="primary"):
            try:
                if  None not in (merged, selected_features, x):
                    with st.spinner(f"Training {model} {selected_model} model..."):

                        X_train, X_test, y_train, y_test, best_model = m4p.pipe_model_train(selected_features, x, merged, model, selected_model)

                        model_filename = f"{model}_{selected_model}.pkl"
                        buffer = io.BytesIO()
                        joblib.dump(best_model, buffer)
                        buffer.seek(0)

                        # Saving
                        st.session_state['X_train'] = X_train
                        st.session_state['X_test'] = X_test
                        st.session_state['y_train'] = y_train
                        st.session_state['y_test'] = y_test
                        st.session_state['best_model'] = best_model
                
                    st.download_button(label ="📥 Download model", data = buffer, file_name = model_filename, mime = "application/octet-stream",
                    use_container_width = True, help = "Export train model")

                else:
                    st.error("⚠️ SYSTEM ERROR: Missing parameters!")  
            except Exception as e:
                st.error(f"An error occurred: {e}")

with tab4:
    st.markdown("<h3 style = 'color: #00E5FF; font-family: monospace;  letter-spacing: 2px;'>"
        "Generate Spatial Predictions"
        "</h3>", unsafe_allow_html=True)
    
    col1, col2 =st.columns([1, 1.5])
    with col2:
        map_container = st.empty() # This is the placeholder for the map
    
    with col1:
        col_map1, col_map2, col_map3 = st.columns(3)
        with col_map1:
            # st.markdown("<small style = 'text-align: center; color: #8892B0; font-size: 1rem; letter-spacing: 2px;'>"
            # "**Year for Map**"
            # "</small>", unsafe_allow_html=True)
            map_year = st.number_input("Year for Map", min_value=2002, max_value=2024, value=2020)
        with col_map2:
            #label_visibility="collapsed": Hide the label name and vanish the spot remaining. If i had = hidden we would still see the blanc spot
            st.markdown("<small style = 'text-align: center; color: #8892B0; font-size: 1rem; letter-spacing: 2px;'>"
            "**Month for Map**"
            "</small>", unsafe_allow_html=True)
            map_month = st.number_input("Month for Map", label_visibility="collapsed", min_value=1, max_value=12, value=5)
        with col_map3:
            st.markdown("<small style = 'text-align: center; color: #8892B0; font-size: 1rem; letter-spacing: 2px;'>"
            "**ERA5 Variable**"
            "</small>", unsafe_allow_html=True)
            era_var = st.selectbox("ERA5 Variable to Plot", ["t2m", "tp", "e","pev","ssro", "sro", "evabs","swvl1", "swvl2", "swvl3", "swvl4", "lai_hv", "lai_lv"], label_visibility="collapsed") 
       
        #ERA5 feature maps
        if st.button("ERA5 Maps",type = "primary"):
            if 'ds_ERA_sliced' not in st.session_state or 'merged' not in st.session_state:
                st.warning("⚠️ Please run 'Data Prep & RFE' in Tab 2 first to load the datasets!")
            else:
                with st.spinner("Drawing Maps..."):
                    era_fig = v4p.ERA_plot(st.session_state['ds_ERA_sliced'], map_year, map_month, era_var, basin_name)
                with map_container.container():
                    st.markdown("<p style='color: #FF00FF; font-family: monospace;'>[SIGNAL_LOCKED]: ERA5 feature map/p>", unsafe_allow_html=True)
                    st.pyplot(era_fig)
                

        uploaded_model = st.file_uploader("Upload Pre-trained Model (.pkl)", type=["pkl"])

        if st.button("GRACE Comparison Maps", type="primary"):
            if uploaded_model is None:
                st.warning("⚠️ Please upload a .pkl model to generate prediction maps!")
            else:
                #st.subheader("Model Predictions vs GRACE Observations")
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
                with map_container.container():
                    st.markdown("<p style='color: #FF00FF; font-family: monospace;'>[SIGNAL_LOCKED]: GRACE_COMPARISON_MATRIX</p>", unsafe_allow_html=True)
                    st.pyplot(csr_fig)
              