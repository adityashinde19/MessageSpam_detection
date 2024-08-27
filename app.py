import numpy as np
import pickle
import streamlit as st

# Load the model, vectorizer, and label encoder
with open("SMS_ham.pkl", "rb") as model_file:
    classifier = pickle.load(model_file)

# with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
#     vectorizer = pickle.load(vectorizer_file)

with open("SMS_label.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Define the mapping for the dependent variable
def map_prediction(prediction):
    return label_encoder.inverse_transform(prediction)[0]

# Prediction function
def sms_prediction(sms_text):
    # Transform the input text using the vectorizer
    # sms_vector = vectorizer.transform([sms_text])

    # Predict using the classifier
    prediction = classifier.predict([sms_text])

    # Map prediction to 'SPAM' or 'HAM'
    return map_prediction(prediction)

# Main function for Streamlit app
def main():
    # Custom CSS for styling
    st.markdown("""
        <style>
            .main {background-color: #f7f7f7; padding: 10px;}
            h2 {color: white;}
            .title-div {background-color: #FF6347; padding: 20px; border-radius: 10px;}
            .input-div {background-color: #FFFFFF; padding: 20px; border-radius: 10px; box-shadow: 2px 2px 10px #aaa;}
            .button-div {text-align: center;}
            .footer {background-color: #FF6347; padding: 10px; text-align: center; color: white;}
        </style>
    """, unsafe_allow_html=True)

    # Title and description
    st.markdown("""
    <div class="title-div">
        <h2 style="text-align:center;">SMS SPAM DETECTION SYSTEM </h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='text-align:center; color:#888;'>Enter an SMS message below to check if it's Spam or Ham.</p>", unsafe_allow_html=True)

    # Input box for SMS description
    sms_input = st.text_input("Enter the SMS Message:", placeholder="Type your message here...")

    # Button for prediction
    if st.button("Check Message"):
        if sms_input.strip():  # Ensure input is not empty
            with st.spinner('Analyzing the message...'):
                result = sms_prediction(sms_input)
                st.success(f"The given message is: **{result}**")
        else:
            st.error("Please enter a valid SMS message.")

    # Footer with some custom message or branding
    st.markdown("""
    <div class="footer">
        <p>Powered by <strong>Aditya Shinde </strong> | SMS Spam Detection</p>
    </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == '__main__':
    main()
