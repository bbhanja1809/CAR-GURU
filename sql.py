from dotenv import load_dotenv
load_dotenv() ## load all the environemnt variables

import streamlit as st
import os
import sqlite3

import google.generativeai as genai
## Configure Genai Key

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## Function To Load Google Gemini Model and provide queries as response

def get_gemini_response(question,prompt):
    model=genai.GenerativeModel('gemini-pro')
    response=model.generate_content([prompt[0],question])
    return response.text

## Fucntion To retrieve query from the database

def read_sql_query(query,db):
    conn=sqlite3.connect(db)
    cur=conn.cursor()
    cur.execute(query)
    rows=cur.fetchall()
    conn.commit()
    conn.close()
    for row in rows:
        print(row)
    return rows

query = """
    DROP TABLE IF EXISTS cars
"""

read_sql_query(query,"student.db")

query = """
    CREATE TABLE cars (
    id int primary key,
    name varchar(100),
    selling_price int(255),
    km_driven int(255),
    fuel varchar(50),
    transmission varchar(50),
    mileage float,
    seats int(255)
    )
"""


######   RAG   ##############################################
from llama_index.core import Document, download_loader
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document
from langchain.utilities import SQLDatabase
from langchain.llms import OpenAI
# from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate

huggingface="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"
os.environ["HUGGINGFACEHUB_API_TOKEN"]=huggingface
# llm = HuggingFaceHub(repo_id="cssupport/t5-small-awesome-text-to-sql")
# llm = HuggingFaceHub(repo_id="MatrixIA/gemma-2b-FT-text-to-Sql")
prompt = PromptTemplate(
    input_variables=["question"],
    template="Translate English to SQL: {question}"
)
##########################################

import pandas as pd

read_sql_query(query,"student.db")

data = pd.read_csv(r".\car_final.csv")
df = pd.DataFrame(data)


conn=sqlite3.connect("student.db")
cur=conn.cursor()


for row in df.itertuples():
    query = """
        INSERT INTO cars (id,name,selling_price,km_driven,fuel,transmission,mileage,seats)
        VALUES(?,?,?,?,?,?,?,?)
    """
    cur.execute(query,row)
    conn.commit()
cur.execute("""SELECT * FROM cars""")
conn.close()    


# ## Define Your Prompt
prompt=[
    """
    You are an expert in converting English questions to SQL query!
    The SQL database has the name cars and has the following columns - id, name, 
    selling_price,km_driven,fuel,transmission,mileage,seats \n\nFor example,\nExample 1 - How many entries of records are present?, 
    the SQL command will be something like this SELECT COUNT(*) FROM car ;
    \nExample 2 - Tell me all the names of cars with mileage greater than 20.0, 
    the SQL command will be something like this SELECT * FROM cars 
    where mileage>20.0; 
    also the sql code should not have ``` in beginning or end and sql word in output

    """
]

# ## Streamlit App

st.set_page_config(page_title="CAR GURU")
st.header("Used CAR :racing_car: Database")

question=st.text_input("Input: ",key="input")

submit=st.button("Your Used Car Guru: Ask away!")

# if submit is clicked
if submit:
    response=get_gemini_response(question,prompt)
    print(response)
    response=read_sql_query(response,"student.db")
    if len(response) == 0:
        st.header("Looks like this car is on a secret mission! We can't find it in the database yet.")
    else:
        for row in response:
            sep = ", "
            formt_str = sep.join(str(x) for x in row)
            print(row)
            st.header(formt_str)