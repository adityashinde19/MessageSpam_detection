import numpy as np
import pickle
import streamlit as st


with open("SMS_ham.pkl", "rb") as model_file:
    classifier = pickle.load(model_file)



with open("SMS_label.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Define the mapping for the dependent variable
def map_prediction(prediction):
    return label_encoder.inverse_transform(prediction)[0]

# Prediction function
def sms_prediction(sms_text):
  

   
    prediction = classifier.predict([sms_text])

    
    return map_prediction(prediction)


def main():
    
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

  
    st.markdown("""
    <div class="title-div">
        <h2 style="text-align:center;">SMS SPAM DETECTION SYSTEM </h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='text-align:center; color:#888;'>Enter an SMS message below to check if it's Spam or Ham.</p>", unsafe_allow_html=True)

    
    sms_input = st.text_input("Enter the SMS Message:", placeholder="Type your message here...")

  
    if st.button("Check Message"):
        if sms_input.strip():  # Ensure input is not empty
            with st.spinner('Analyzing the message...'):
                result = sms_prediction(sms_input)
                st.success(f"The given message is: **{result}**")
        else:
            st.error("Please enter a valid SMS message.")

   
    st.markdown("""
    <div class="footer">
        <p>Powered by <strong>Aditya Shinde </strong> | SMS Spam Detection</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
