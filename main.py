import streamlit as st
import pandas as pd
import joblib
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# Load model
hmodel = joblib.load("heart.pkl")

# Prediction
def predict_heart(data: dict):
    df = pd.DataFrame([data])
    pred = hmodel.predict(df)[0]
    if pred == 1:
        return "Yes, you have a heart disease"
    else:
        return "No, you are ok"

# LangChain tool
@tool
def preddes(data: dict):
    """
    Predict heart disease using user-provided data.
    Required keys: Age, Cholesterol, MaxHR, RestingBP, Oldpeak
    """
    return predict_heart(data)

# LLM Setup
template = '''
You are a smart medical assistant (not a doctor).
Your task is to check if the user has heart disease based on their inputs.

If ANY required input is missing, ask the user ONLY for that missing value.

Required inputs (exact names):
- Age
- Cholesterol
- MaxHR
- RestingBP
- Oldpeak

Once all 5 values are available, call the tool with:
preddes(data)

If the user is NOT talking about disease prediction, chat normally.

User message:
{question}
'''
prompt = ChatPromptTemplate.from_template(template)
model = OllamaLLM(model="llama3.2", max_token=100, temperature=0.1)
chain = prompt | model

# UI
st.title("💓 Heart Disease Predictor Chat")
st.write("This AI driven tool helps with predicting heart disease.")

#ST form for prediction func
with st.form("heart_form"):
    age = st.number_input("Age", min_value=0, max_value=120)
    cholesterol = st.number_input("Cholesterol", min_value=0)
    max_hr = st.number_input("MaxHR", min_value=0)
    resting_bp = st.number_input("RestingBP", min_value=0)
    oldpeak = st.number_input("Oldpeak", min_value=0.0, step=0.1)
    submit = st.form_submit_button("Predict Heart Disease")

if submit:
    user_data = {
        "Age": age,
        "Cholesterol": cholesterol,
        "MaxHR": max_hr,
        "RestingBP": resting_bp,
        "Oldpeak": oldpeak
    }
    prediction = predict_heart(user_data)
    st.success(prediction)


# Chat container

if "messages" not in st.session_state:
    st.session_state.messages = []

# Displaying Messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'''
        <div style="
            text-align: right;
            background-color: #4CAF50;
            color: white;
            padding: 10px;
            border-radius: 15px;
            margin: 5px 50px 5px 5px;
            max-width: 70%;
            font-family: Arial, sans-serif;">
            {msg["content"]}
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div style="
            text-align: left;
            background-color: #E0E0E0;
            color: black;
            padding: 10px;
            border-radius: 15px;
            margin: 5px 5px 5px 50px;
            max-width: 70%;
            font-family: Arial, sans-serif;">
            {msg["content"]}
        </div>
        ''', unsafe_allow_html=True)

# ----------------------------
# Input  session_state
# ----------------------------
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message here...", key="input_box")
    send = st.form_submit_button("Send")

    if send and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        response = chain.invoke({"question": user_input})
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()   
