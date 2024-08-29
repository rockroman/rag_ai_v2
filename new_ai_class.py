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

# Import the OpenAIChat class from the helper file
from openai_helper import OpenAIChat

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit setup
st.title("Chat with OpenAI RAG AI solution using PDF")

if "history" not in st.session_state:
    st.session_state.history = []

# Step 1: Load the PDF data
loader = PyPDFLoader('./merged_files.pdf')
docs = loader.load()

# Step 2: Split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)

# Step 3: Create embeddings and vector store using Chroma DB
if "vector" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
    st.session_state.vector = Chroma.from_documents(documents, st.session_state.embeddings)

# Instantiate the OpenAIChat class
chat = OpenAIChat(openai_api_key=openai_api_key)

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
    document_chain = create_stuff_documents_chain(chat.send_message, prompt_template)
    retriever = st.session_state.vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Process the prompt and display the response
    retrieved_docs = retriever.invoke(prompt)
    formatted_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    response = chat.send_message(formatted_context)

    # Save the current prompt and response to the history
    st.session_state.history.append({"prompt": prompt, "response": response})

# Display the history of prompts and responses
for entry in reversed(st.session_state.history):
    st.write(f"**Prompt:** {entry['prompt']}")
    st.write(f"**Response:** {entry['response']}")
    st.write("---")
