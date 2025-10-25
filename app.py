# app.py
import streamlit as st  # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
import joblib  # pyright: ignore[reportMissingImports]
import plotly.express as px  # pyright: ignore[reportMissingImports]

# Page setup
st.set_page_config(page_title="🏠 Hyderabad House Price Predictor", layout="wide")

# Load model, columns, metrics
model = joblib.load("house_price_model.pkl")
columns = joblib.load("model_columns.pkl")
metrics = joblib.load("model_metrics.pkl")

# Sidebar setup
st.sidebar.title("⚙️ App Settings")
theme = st.sidebar.radio("Choose Theme", ["🌞 Light", "🌙 Dark"])

if theme == "🌙 Dark":
    st.markdown("""<style>body { background-color: #121212; color: white; }</style>""", unsafe_allow_html=True)

st.sidebar.header("📊 Model Performance")
st.sidebar.metric("MAE", f"{metrics['MAE']:.2f}")
st.sidebar.metric("RMSE", f"{metrics['RMSE']:.2f}")
st.sidebar.metric("R² Score", f"{metrics['R2']:.3f}")

page = st.sidebar.radio("Navigation", ["🏡 Predict Price", "📈 Insights Dashboard"])

# ------------------- PREDICTION PAGE -------------------
if page == "🏡 Predict Price":
    st.title("🏠 Hyderabad House Price Prediction App")
    st.write("Enter your house details below to get an estimated price (in Lakhs).")

    # Dropdown list of locations
    locations = sorted([col.replace("location_", "") for col in columns if "location_" in col])
    if not locations:  # If no location columns found, use a default list
        locations = ["Gachibowli", "Madhapur", "Kondapur", "Manikonda", "Miyapur", "Kukatpally", "Hitech City", "Banjara Hills", "Jubilee Hills", "Nallagandla", "Tellapur", "Nizampet", "Chanda Nagar", "Begumpet", "Somajiguda", "Ameerpet", "Lingampally", "Kompally", "Alwal", "Secunderabad", "Attapur", "Kothapet", "L B Nagar", "Uppal", "Nagole", "Abids", "Masab Tank", "Himayatnagar", "Bachupally", "Narsingi", "Bowenpally"]

    col1, col2, col3 = st.columns(3)
    with col1:
        sqft = st.number_input("Total Square Feet", min_value=500, max_value=10000, step=50, value=1500)
    with col2:
        bhk = st.slider("Number of Bedrooms (BHK)", 1, 10, 3)
    with col3:
        bath = st.slider("Number of Bathrooms", 1, 10, 2)

    location = st.selectbox("Select Location in Hyderabad", options=locations, index=0)

    if st.button("🔍 Predict Price"):
        input_df = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)
        input_df.loc[0, ['total_sqft', 'bhk', 'bath']] = [sqft, bhk, bath]

        # Handle location encoding
        loc_col = f'location_{location}'
        if loc_col in input_df.columns:
            input_df.loc[0, loc_col] = 1
        else:
            # If location not found in model, use the first available location column
            location_cols = [col for col in columns if col.startswith('location_')]
            if location_cols:
                input_df.loc[0, location_cols[0]] = 1

        predicted_price = model.predict(input_df)[0]
        st.success(f"💰 Estimated House Price in Hyderabad: ₹{predicted_price:.2f} Lakhs")
        st.progress(min(predicted_price / 500, 1.0))
        st.caption("Progress bar capped at ₹500 Lakhs for display purposes.")

# ------------------- INSIGHTS DASHBOARD -------------------
else:
    st.title("📈 Hyderabad Real Estate Insights Dashboard")

    try:
        df = pd.read_csv("Hyderabad_House_Data.csv")
        df = df.dropna()
        df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]) if isinstance(x, str) else None)
        df = df[df['total_sqft'].apply(lambda x: str(x).replace('.', '', 1).isdigit())]
        df['total_sqft'] = df['total_sqft'].astype(float)
        df = df[df['bath'] < df['bhk'] + 2]
        df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']

        st.subheader("🏘️ Top 10 Most Expensive Locations in Hyderabad")
        top_loc = df.groupby("location")["price_per_sqft"].mean().sort_values(ascending=False).head(10).reset_index()
        fig1 = px.bar(top_loc, x="location", y="price_per_sqft", color="price_per_sqft",
                      color_continuous_scale="Viridis", title="Top 10 Expensive Locations")
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("💧 Price Distribution by Bathrooms")
        fig2 = px.box(df, x="bath", y="price", points="all", color="bath", title="Price Distribution vs Bathrooms")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("📊 Relationship between Sqft and Price")
        # Sample data safely - use all data if less than 500 rows
        sample_size = min(500, len(df))
        if len(df) > 0:
            fig3 = px.scatter(df.sample(sample_size), x="total_sqft", y="price", color="bhk",
                              title="Price vs Total Sqft by BHK")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("No data available for visualization.")

        st.caption("📍 Data Source: Hyderabad_House_Data.csv | Created by Praneeth Yadav 🧠")
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure the Hyderabad_House_Data.csv file exists and has proper data.")
