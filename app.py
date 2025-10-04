import streamlit as st

# Title of the web app
st.title("Find your XOXOplanet")

# Input fields for user data
orbital_period = st.number_input("Enter Orbital Period (days):", min_value=0.0)
transit_duration = st.number_input("Enter Transit Duration (hours):", min_value=0.0)
planetary_radius = st.number_input("Enter Planetary Radius (Earth radii):", min_value=0.0)

# Button to submit the data
if st.button("Check Exoplanet"):
    # Placeholder for model prediction logic
    # Here you would call your model to predict if it's an exoplanet
    # For now, we'll just simulate a response
    if orbital_period > 10 and transit_duration > 5 and planetary_radius > 1:
        result = "This is likely an Exoplanet!"
    else:
        result = "This is likely NOT an Exoplanet."

    # Display the result
    st.success(result)

# Optional: Add some animation or visualization here

