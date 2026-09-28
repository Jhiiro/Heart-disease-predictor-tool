# Heart Disease Prediction with LLM Chat Interface

Binary classifier (Logistic Regression) + local LLM chat interface for heart disease risk assessment.

> *Academic prototype only.*

---



**Dataset:** [Kaggle Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) (918 records, 11 features)  
**Performance:** 71.7% test accuracy

---

Setup

1. Install [Ollama](https://ollama.com) and pull Mistral:
```bash
   ollama pull mistral
   
   git clone https://github.com/your-username/heart-disease-prediction.git
   cd heart-disease-prediction
   pip install -r requirements.txt
   
   streamlit run app.py
   
