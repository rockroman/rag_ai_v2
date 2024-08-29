import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
import requests
import json

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit setup
st.title("Chat with OpenAI RAG AI solution using PDF")

if "history" not in st.session_state:
    st.session_state.history = []

# Step 1: Load the PDF data

# loader = PyPDFLoader('./conditions-of-sale-2023-revised-fillable-pdf.pdf')
# loader = PyPDFLoader('https://www.lawsociety.ie/globalassets/documents/enewsletters/ezine-files/conditionsofsale2019-draft.pdf')
loader = PyPDFLoader('./merged_files.pdf')
docs = loader.load()

# Step 2: Split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)

# # Step 3: Create embeddings and vector store using FAISS
# if "vector" not in st.session_state:
#     st.session_state.embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
#     st.session_state.vector = FAISS.from_documents(documents, st.session_state.embeddings)

 # Step 3: Create embeddings and vector store using Chroma DB
if "vector" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
    st.session_state.vector = Chroma.from_documents(documents, st.session_state.embeddings)

# Function to interact with OpenAI API
def openai_llm(question, context):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    data = {
        "model": "gpt-3.5-turbo",  # or "gpt-4" if you have access
        "messages": [{"role": "user", "content": formatted_prompt}]
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code}, {response.text}"

# Function to check question relevance
def is_question_relevant_to_context(context, question):
    context_keywords = set(context.split())
    question_keywords = set(question.split())
    intersection = context_keywords.intersection(question_keywords)
    return len(intersection) > 0

# Function to create the prompt template
def get_prompt(context, input):
    return ChatPromptTemplate.from_template(
        """
        If the context provides relevant information, use it to answer the following question.
        If not, rely on your general knowledge to answer.

        Context: {context}

        Question: {input}
        """
    )

# Function to reset input
def reset_input():
    st.session_state["main_input"] = ""

# Reset input before rendering the text input
if st.session_state.get("reset", False):
    reset_input()
    st.session_state["reset"] = False

# Input field for prompt
prompt = st.text_input("Input your prompt here", key="main_input")

if prompt:
    # Create the prompt template
    prompt_template = get_prompt(st.session_state.get("context", ""), prompt)

    # Create the document chain and retrieval setup
    document_chain = create_stuff_documents_chain(openai_llm, prompt_template)
    retriever = st.session_state.vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Process the prompt and display the response
    # retrieved_docs = retriever.get_relevant_documents(prompt)
    retrieved_docs = retriever.invoke(prompt)
    formatted_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    response = openai_llm(prompt, formatted_context)

    # st.write(response)


    # Save the current prompt and response to the history
    st.session_state.history.append({"prompt": prompt, "response": response})
    # Clear the input field after submission
    # st.session_state["main_input"] = ""
    

# Display the history of prompts and responses
for entry in reversed(st.session_state.history):
    st.write(f"**Prompt:** {entry['prompt']}")
    st.write(f"**Response:** {entry['response']}")
    st.write("---")



































# import streamlit as st
# from dotenv import load_dotenv
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores import FAISS
# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# import os
# import requests
# import json

# # Load environment variables
# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")

# # Streamlit setup
# st.title("Chat with OpenAI RAG AI solution using PDF")

# # Step 1: Load the PDF data

# # loader = PyPDFLoader('./conditions-of-sale-2023-revised-fillable-pdf.pdf')
# # loader = PyPDFLoader('https://www.lawsociety.ie/globalassets/documents/enewsletters/ezine-files/conditionsofsale2019-draft.pdf')
# loader = PyPDFLoader('./merged_files.pdf')
# docs = loader.load()

# # Step 2: Split the text into smaller chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# documents = text_splitter.split_documents(docs)

# # # Step 3: Create embeddings and vector store using FAISS
# # if "vector" not in st.session_state:
# #     st.session_state.embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
# #     st.session_state.vector = FAISS.from_documents(documents, st.session_state.embeddings)

#  # Step 3: Create embeddings and vector store using Chroma DB
# if "vector" not in st.session_state:
#     st.session_state.embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
#     st.session_state.vector = Chroma.from_documents(documents, st.session_state.embeddings)

# # Function to interact with OpenAI API
# def openai_llm(question, context):
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {openai_api_key}"
#     }
#     formatted_prompt = f"Question: {question}\n\nContext: {context}"
#     data = {
#         "model": "gpt-3.5-turbo",  # or "gpt-4" if you have access
#         "messages": [{"role": "user", "content": formatted_prompt}]
#     }
#     response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
#     if response.status_code == 200:
#         return response.json()['choices'][0]['message']['content']
#     else:
#         return f"Error: {response.status_code}, {response.text}"

# # Function to check question relevance
# def is_question_relevant_to_context(context, question):
#     context_keywords = set(context.split())
#     question_keywords = set(question.split())
#     intersection = context_keywords.intersection(question_keywords)
#     return len(intersection) > 0

# # Function to create the prompt template
# def get_prompt(context, input):
#     return ChatPromptTemplate.from_template(
#         """
#         If the context provides relevant information, use it to answer the following question.
#         If not, rely on your general knowledge to answer.

#         Context: {context}

#         Question: {input}
#         """
#     )

# # Function to reset input
# def reset_input():
#     st.session_state["main_input"] = ""

# # Reset input before rendering the text input
# if st.session_state.get("reset", False):
#     reset_input()
#     st.session_state["reset"] = False

# # Input field for prompt
# prompt = st.text_input("Input your prompt here")

# if prompt:
#     # Create the prompt template
#     prompt_template = get_prompt(st.session_state.get("context", ""), prompt)

#     # Create the document chain and retrieval setup
#     document_chain = create_stuff_documents_chain(openai_llm, prompt_template)
#     retriever = st.session_state.vector.as_retriever()
#     retrieval_chain = create_retrieval_chain(retriever, document_chain)

#     # Process the prompt and display the response
#     # retrieved_docs = retriever.get_relevant_documents(prompt)
#     retrieved_docs = retriever.invoke(prompt)
#     formatted_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
#     response = openai_llm(prompt, formatted_context)

#     # st.write(response)

# # Save the current prompt and response to the history
# st.session_state.history.append({"prompt": prompt, "response": response})
    

# if "history" not in st.session_state:
#     st.session_state.history = []

# # Display the history of prompts and responses
# for entry in reversed(st.session_state.history):
#     st.write(f"**Prompt:** {entry['prompt']}")
#     st.write(f"**Response:** {entry['response']}")
#     st.write("---")
