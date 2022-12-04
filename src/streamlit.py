import streamlit as st
import requests
from PIL import Image

# Load and set images in the first place
header_images = Image.open(r'./src/gambar.jpg')
st.image(header_images)

# Header of the UI
st.title("Company Bankruptcy Prediction")
st.subheader("Just enter variabel below then click Predict button :sunglasses:")

# Create form of input
with st.form(key="company_data_form"):
    # Create box for number input
    Attr21 = st.number_input(
        label="1.Enter Attr21 Value:",
        min_value=-135.15,
        max_value=7661.50,
        help="Value range from -135.15 to 7661.50"
    )

    Attr13 = st.number_input(
        label="2.Enter Attr13 Value:",
        min_value=-310.34,
        max_value=2340.20,
        help="Value range from -310.34 to 2340.20"
    )

    Attr27 = st.number_input(
        label="3.Enter Attr27 Value:",
        min_value=-158130.00,
        max_value=565940.00,
        help="Value range from -158130 to 565940"
    )

    Attr34 = st.number_input(
        label="4.Enter Attr34 Value:",
        min_value=-16.015,
        max_value=7590.50,
        help="Value range from -16.015 to 7590.50"
    )

    Attr24 = st.number_input(
        label="5.Enter Attr24 Value:",
        min_value=-463.89,
        max_value=252.34,
        help="Value range from 463.89 to 252.34"
    )

    Attr35 = st.number_input(
        label="6.Enter Attr35 Value:",
        min_value=-431.59,
        max_value=15.541,
        help="Value range from -431.59 to 15.541"
    )

    Attr51 = st.number_input(
        label="7.Enter Attr51 Value:",
        min_value=-431.59,
        max_value=15.541,
        help="Value range from -431.59 to 15.541"
    )

    Attr6 = st.number_input(
        label="8.Enter Attr6 Value:",
        min_value=-463.89,
        max_value=543.25,
        help="Value range from -463.89 to 543.25"
    )

    Attr56 = st.number_input(
        label="9.Enter Attr56 Value:",
        min_value=-46.788,
        max_value=1.651,
        help="Value range from -46.788 to 1.651"
    )

    Attr49 = st.number_input(
        label="10.Enter Attr49 Value:",
        min_value=-144.800,
        max_value=16.866,
        help="Value range from -144.800 to 16.866"
    )

    Attr40 = st.number_input(
        label="11.Enter Attr40 Value:",
        min_value=-9.0686,
        max_value=4303.20,
        help="Value range from -9.0686 to 4303.20"
    )

    Attr55 = st.number_input(
        label="12.Enter Attr55 Value:",
        min_value=-1.118500e+06,
        max_value=4.212200e+06,
        help="Value range from -1.118500e+06 to 4.212200e+06"
    )

    Attr50 = st.number_input(
        label="13.Enter Attr50 Value:",
        min_value=-0.012175,
        max_value=6845.800,
        help="Value range from -0.012175 to 6845.800"
    )

    Attr58 = st.number_input(
        label="14.Enter Attr58 Value:",
        min_value=-0.164390,
        max_value=47.788000,
        help="Value range from -0.1643906 to 47.788000"
    )

    Attr43 = st.number_input(
        label="15.Enter Attr43 Value:",
        min_value=-3975.600,
        max_value=40515.00,
        help="Value range from -3975.600 to 40515.00"
    )

    # Create button to submit the form
    submitted = st.form_submit_button("Predict")

    # Condition when form submitted
    if submitted:
        # Create dict of all data in the form
        raw_data = {
            "Attr21": Attr21,
            "Attr13": Attr13,
            "Attr27": Attr27,
            "Attr34": Attr34,
            "Attr24": Attr24,
            "Attr35": Attr35,
            "Attr51": Attr51,
            "Attr6": Attr6,
            "Attr56": Attr56,
            "Attr49": Attr49,
            "Attr40": Attr40,
            "Attr55": Attr55,
            "Attr50": Attr50,
            "Attr58": Attr58,
            "Attr43": Attr43,
        }

        # Create loading animation while predicting
        with st.spinner("Sending data to prediction server ..."):
            res = requests.post("http://api:8080/predict", json=raw_data).json()

        # Parse the prediction result
        if res["error_msg"] != "":
            st.error("Error Occurs While Predicting: {}".format(res["error_msg"]))
        else:
            if res["prediction"] == 1:
                st.warning("Predicted Company will Bankrupt.!!")
            else:
                st.success("Predicted Company will Not Bankrupt")