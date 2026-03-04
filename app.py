
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import random

model = tf.keras.models.load_model("skin_cnn_model.h5")

class_names = ["Dry", "Normal", "Oily"]

product_recommendations = {
    "Oily": {
        "Cleanser": [
            "Neutrogena Oil-Free Acne Wash",
            "La Roche-Posay Effaclar Gel Cleanser",
            "Minimalist Salicylic Acid Cleanser",
            "Plum Green Tea Face Wash",
            "Bioderma Sebium Gel",
            "Mamaearth Tea Tree Face Wash",
            "Clean & Clear Foaming Face Wash",
            "Dot & Key Cica Face Wash",
            "Re'equil Oil Control Face Wash",
            "The Body Shop Tea Tree Wash"
        ],
        "Moisturizer": [
            "Neutrogena Hydro Boost Water Gel",
            "Cetaphil Oil-Free Moisturizer",
            "Plum Green Tea Oil-Free Moisturizer",
            "Dot & Key Cica Gel Moisturizer",
            "Pond’s Super Light Gel",
            "Re'equil Oil Free Moisturizer",
            "Foxtale Oil Control Gel",
            "Mamaearth Oil-Free Moisturizer",
            "Bioderma Sebium Hydra",
            "Minimalist Sepicalm Moisturizer"
        ],
        "Sunscreen": [
            "Re'equil Ultra Matte SPF 50",
            "Neutrogena Ultra Sheer SPF 50",
            "La Shield SPF 40 Gel",
            "The Derma Co Oil Free SPF 50",
            "Acne UV Gel SPF 50",
            "UV Doux SPF 50",
            "Minimalist SPF 50 Sunscreen",
            "Dot & Key Matte Sunscreen",
            "Foxtale Matte Sunscreen",
            "Aqualogica Detan+ SPF 50"
        ],
        "Serum": [
            "Minimalist Niacinamide 10%",
            "The Derma Co Niacinamide Serum",
            "Plum 10% Niacinamide Serum",
            "La Roche-Posay Effaclar Serum",
            "The Ordinary Niacinamide",
            "Paula’s Choice BHA",
            "Re'equil Acne Control Serum",
            "Dot & Key Cica Serum",
            "Mamaearth Tea Tree Serum",
            "Pilgrim Niacinamide Serum"
        ]
    },

    "Dry": {
        "Cleanser": [
            "Cetaphil Gentle Skin Cleanser",
            "CeraVe Hydrating Cleanser",
            "Simple Moisturizing Face Wash",
            "La Roche-Posay Hydrating Cleanser",
            "Bioderma Atoderm Cleanser",
            "Minimalist Oat Cleanser",
            "Re'equil Gentle Cleanser",
            "Dot & Key Barrier Cleanser",
            "Foxtale Hydrating Cleanser",
            "The Body Shop Aloe Cleanser"
        ],
        "Moisturizer": [
            "CeraVe Moisturizing Cream",
            "Cetaphil Moisturizing Cream",
            "Minimalist Marula Oil Cream",
            "Dot & Key Ceramide Cream",
            "Bioderma Atoderm Cream",
            "Foxtale Barrier Repair Cream",
            "Re'equil Ceramide Cream",
            "The Body Shop Vitamin E Cream",
            "Nivea Soft Cream",
            "Simple Rich Moisturizer"
        ],
        "Sunscreen": [
            "CeraVe Hydrating Sunscreen",
            "La Roche-Posay Hydrating SPF 50",
            "Bioderma Cream SPF 50",
            "Minimalist SPF 50 Cream",
            "Dr Sheth Ceramide SPF 50",
            "Dot & Key Vitamin C SPF 50",
            "Foxtale Dewy Sunscreen",
            "Nivea Moisturizing SPF 50",
            "Cetaphil SPF 50 Lotion",
            "Earth Rhythm Glow SPF 50"
        ],
        "Serum": [
            "Minimalist Hyaluronic Acid",
            "The Ordinary Hyaluronic Acid",
            "Dot & Key Hyaluronic Serum",
            "Plum Hyaluronic Serum",
            "La Roche-Posay Hyalu B5",
            "Bioderma Hydrabio Serum",
            "CeraVe Hydrating Serum",
            "Re'equil Hyaluronic Serum",
            "Foxtale Hydrating Serum",
            "The Body Shop Vitamin E Serum"
        ]
    },

    "Normal": {
        "Cleanser": [
            "Cetaphil Daily Cleanser",
            "Simple Kind to Skin Cleanser",
            "Minimalist Gentle Cleanser",
            "Plum Gentle Face Wash",
            "Mamaearth Ubtan Face Wash",
            "Dot & Key Gentle Cleanser",
            "CeraVe Foaming Cleanser",
            "Foxtale Daily Cleanser",
            "Aqualogica Gentle Cleanser",
            "The Body Shop Vitamin C Wash"
        ],
        "Moisturizer": [
            "Pond’s Super Light Gel",
            "Cetaphil Daily Moisturizer",
            "Simple Hydrating Light Moisturizer",
            "Plum Green Tea Moisturizer",
            "Dot & Key Vitamin C Cream",
            "CeraVe Moisturizing Lotion",
            "Foxtale Daily Moisturizer",
            "Re'equil Daily Cream",
            "Nivea Soft Cream",
            "Minimalist Sepicalm Cream"
        ],
        "Sunscreen": [
            "Neutrogena SPF 50",
            "Minimalist SPF 50",
            "Re'equil SPF 50",
            "Dot & Key SPF 50",
            "Plum SPF 50",
            "Mamaearth SPF 50",
            "CeraVe SPF 50",
            "La Roche-Posay SPF 50",
            "Foxtale SPF 50",
            "Bioderma SPF 50"
        ],
        "Serum": [
            "Minimalist Vitamin C",
            "The Ordinary Vitamin C",
            "Dot & Key Vitamin C Serum",
            "Plum Vitamin C Serum",
            "Mamaearth Vitamin C Serum",
            "CeraVe Vitamin C Serum",
            "Foxtale Glow Serum",
            "Re'equil Vitamin C Serum",
            "Pilgrim Vitamin C Serum",
            "The Body Shop Vitamin C Serum"
        ]
    }
}

st.title("AI Skin Type Detector & Skincare Recommender")

uploaded_file = st.file_uploader("Upload Your Face Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_resized = cv2.resize(image, (128, 128))

    st.image(image, channels="BGR")

    img_array = np.expand_dims(image_resized / 255.0, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Detected Skin Type: {predicted_class}")

    if st.button("Generate Skincare Routine"):

        cleanser = random.choice(product_recommendations[predicted_class]["Cleanser"])
        moisturizer = random.choice(product_recommendations[predicted_class]["Moisturizer"])
        sunscreen = random.choice(product_recommendations[predicted_class]["Sunscreen"])
        serum = random.choice(product_recommendations[predicted_class]["Serum"])

        st.subheader("Your Personalized Routine")

        st.write("Cleanser:", cleanser)
        st.write("Moisturizer:", moisturizer)
        st.write("Sunscreen:", sunscreen)
        st.write("Serum:", serum)
