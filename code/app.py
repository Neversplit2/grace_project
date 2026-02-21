import streamlit as st
import main_4_app as m4p
import vis_4_app as v4p
st.title("Grace LWE project")

st.write("Welcome to the GRACE LWE Prediction interface!")

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("1. Area of Interest")
    lat_min = st.number_input("Latitude Min", value=-20.0) # Used dummy defaults for speed
    lat_max = st.number_input("Latitude Max", value=20.0)
    lon_min = st.number_input("Longitude Min", value=-60.0)
    lon_max = st.number_input("Longitude Max", value=10.0)

    st.header("2. Dataset")
    grace_data = st.selectbox("GRACE Dataset", ["CSR", "JPL"])

    st.header("3. RFE Settings")
    model_RFE = st.selectbox("Select Model for RFE", ["XGBoost", "RF"])
    n_features = st.number_input("Number of features to select", min_value=1, value=5, step=1)

# --- MAIN EXECUTION ---
# When the user clicks this button, the magic happens
if st.button("🚀 Run Data Prep & RFE", type="primary"):
    
    # We use a try/except block so if your data code crashes, Streamlit shows the error nicely on screen
    try:
        # Step 1: Data Prep
        with st.spinner("Preparing Data... (Downloading and merging)"):
            df_ERA, df_CSR, ds_ERA_sliced, ds_CSR_sliced, merged = m4p.pipe_data_prp(
                grace_data, lat_min, lat_max, lon_min, lon_max
            )
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