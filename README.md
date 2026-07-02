# Veritasium Comment Analyzer
### RAG, LLM Agents, and MLflow Monitoring

A YouTube comments analysis system built with NLP and RAG. Scrapes, preprocesses, 
and analyzes comments using sentiment analysis, NER, topic modeling, and an 
LLM-powered agent for Q&A and summarization.

This can work on both python versions 3.10 and 3.11, but 3.11 is recommended.

## Setup Instructions: 

* Download the folder and open the folder in **Visual Studio Code**.

**Very Important**: When you download the this GitHub zip folder and you have extracted the files, make sure to choose the correct folder and move it out into your main C:\ drive area. 

* Open terminal and create the venv folder:
`py -m venv venv`

or if you want a specific version of python:
`py (python version) -m venv venv`

Or this easy command:
`py -3.11 -m venv venv`

* Activate the venv environment through terminal:
`.\venv\Scripts\activate`
you will have activated it as indicated by a green '(venv)' at the begining of your command lines.

## Install all dependencies

* When in the virtual environment, install all the dependencies required by executing **requirements_v.txt** file in the terminal: `pip install -r requirements_v.txt`

* Download Ollama desktop app and install it on your machine from:
https://ollama.com

* After installing the app, go back to vs code terminal and pull all the required models: 
- ollama pull qwen2.5:7b
- ollama pull llama3.2:3b
- ollama pull mxbai-embed-large
- ollama pull nomic-embed-text
- ollama pull embeddinggemma

  And then type `ollama list`, if you can a list of your models there, then that means they've have been installed.

To make sure no errors occur or no processes get stuck anywhere, make sure the Ollama app is running in the background or make sure the Ollama app icon is visible in the taskbar. 

## FAISS Vector Store

The FAISS store is inlcuded as `veritasium_faiss_store_2.zip`
Unzip this before runnig the dashboard.
It is important the name should not be changed as it is used in the code. Make sure the name is the same.
No reuilding needed. 

# After all of this you are ready ot open the dashboard/ Run the application:

**Important** The two main files that are of concern are the **Processed comments/Veritasium_comments_modeling_topics_3.0.csv** and **veritasium_faiss_store_2/**, do not change their names or locations.

# Start MLflow:

* Open mlflow dashboard through vs code terminal first:
`mlflow ui --port 5000`

After excecution visit 'http://127.0.0.1:5000' to visit the mlflow dashboard. 

This will be used for evaluating the performance of the AI agent within the app dashboard, look for the name **Veritasium video comments analyzer**.

# Start main dashboard:

* Once thats done, open the app dashboard by executing this in the vs code terminal: `streamlit run dashboard.py`

This will open on 'http://localhost:8501'

It will automatically open in a browser tab and wait until the site loads.
Once open all of the graphs and tables will be visible and you can look through and test around with all the features.




