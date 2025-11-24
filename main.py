from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
import pandas as pd
import streamlit as st
import joblib
hmodel = joblib.load("heart.pkl")
@tool
def  preddes(data:dict):
    """
    Predict heart disease based on user inputs.
    Required keys:
    Age, Cholesterol, MaxHR, RestingBP, Oldpeak
    """    
    df=pd.DataFrame([data])
    pred=hmodel.predict(df)[0]
    if pred == 1:
        return "yes you have a heart disease"
    else:
        return "no you are ok"
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
prompt=ChatPromptTemplate.from_template(template)
model=OllamaLLM(model="llama3.2",max_token= 100,temperature=0.1)
chain=prompt|model
required_fields = ["Age", "Cholesterol", "MaxHR", "RestingBP", "Oldpeak"]
user_data = {field: None for field in required_fields}
while True:
    question=input("user:\n")
    if question.lower() in ["quit","exit", "end"]:
        break
    filled_field = False
    for field in required_fields:
        if user_data[field] is None:
            try:
                user_data[field] = float(question)
                filled_field = True
                break
            except ValueError:
                pass

    if all(user_data[field] is not None for field in required_fields):
        response = preddes(user_data)
        print(f"agent:\n{response}")
        user_data = {field: None for field in required_fields}
    elif filled_field:
       
        next_field = next(f for f in required_fields if user_data[f] is None)
        print(f"agent:\n Please insert the next field{next_field} ")
    else:
       
        response = chain.invoke({"question": question})
        print(f"agent:\n{response}")