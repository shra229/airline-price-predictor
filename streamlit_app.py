import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Dynamic Airline Price Predictor",
    page_icon="✈️",
    layout="wide"
)

model = joblib.load("dynamic_pricing_small.pkl")

st.title("✈️ Dynamic Airline Price Predictor")
st.markdown("Enter travel details to get an estimated ticket price and compare it with competitors.")
st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)

left_col, right_col = st.columns([1.2, 1.8])

with left_col:
    st.subheader("Input Travel Details")
    airline = st.selectbox("Airline", ["Indigo", "Air_India", "SpiceJet", "Vistara", "GO_FIRST"])
    source_city = st.selectbox("Source City", ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Chennai"])
    destination_city = st.selectbox("Destination City", ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Chennai"])
    travel_class = st.selectbox("Class", ["Economy", "Business"])
    season = st.selectbox("Season", ["Winter", "Summer", "Monsoon", "Autumn"])
    days_left = st.slider("Days Left to Departure", 0, 60, 30)
    is_holiday = st.selectbox("Is it a Holiday?", [0, 1])
    competitor_avg_price = st.number_input("Competitor Avg Price (₹)", min_value=1000, max_value=20000, value=5000)

    st.markdown("### 💬 Feedback")
    st.text_area("What do you think about this prediction?", placeholder="Your feedback...")

with right_col:
    if st.button("Predict Price"):
        input_data = pd.DataFrame([{
            "airline": airline,
            "source_city": source_city,
            "destination_city": destination_city,
            "class": travel_class,
            "days_left": days_left,
            "is_holiday": is_holiday,
            "season": season,
            "competitor_avg_price": competitor_avg_price,
            "stops": 1,
            "weekday": 2,
            "is_weekend": 0,
            "load_factor": 0.85,
            "duration": 120,
            "departure_time": "Morning",
            "arrival_time": "Afternoon",
            "price_category": "Medium",
            "demand_score": 0.5,
            "total_revenue": 100000,
            "last_minute_booking": 0
        }])

        st.subheader("Input Data Preview")
        st.dataframe(input_data)

        predicted_price = model.predict(input_data)[0]
        st.subheader(f"💰 Estimated Ticket Price: ₹{predicted_price:,.0f}")

        st.markdown("### 📊 Predicted vs Competitor Price")

        plt.style.use("ggplot")
        fig, ax = plt.subplots(figsize=(6, 4))

        bars = ax.bar(
            ["Predicted Price", "Competitor Price"],
            [predicted_price, competitor_avg_price],
            color=["#4c78a8", "#f58518"],
            width=0.5,
            edgecolor="black"
        )

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"₹{int(height):,}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 6),
                        textcoords="offset points",
                        ha='center', fontsize=10, color='black', weight='bold')

        ax.set_title("Price Comparison", fontsize=14, fontweight='bold')
        ax.set_ylabel("Price (₹)", fontsize=12)
        ax.set_ylim(0, max(predicted_price, competitor_avg_price) + 10000)
        ax.tick_params(colors='black', labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')

        st.pyplot(fig)

        diff = predicted_price - competitor_avg_price
        if diff > 0:
            st.info(f"Your price is ₹{diff:,.0f} higher than competitor. Consider adjusting for competitiveness.")
        elif diff < 0:
            st.success(f"Your price is ₹{-diff:,.0f} lower than competitor. Might help boost bookings!")
        else:
            st.info("Price is aligned with competitors.")