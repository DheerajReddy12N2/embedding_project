# app.py
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

st.title("📄 AI PDF Query App (Enhanced QA with Context Checking)")

# Step 1: Upload PDFs
uploaded_files = st.file_uploader(
    "Upload your PDF files", type=["pdf"], accept_multiple_files=True
)

# Persistent vectorstore
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

# Step 2: Create embeddings and store in FAISS only once
if uploaded_files and st.session_state.vectorstore is None:
    all_text = ""
    for uploaded_file in uploaded_files:
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            all_text += page.extract_text()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(all_text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    st.session_state.vectorstore = FAISS.from_texts(chunks, embeddings)
    st.success("PDF embeddings created and stored successfully!")

# Step 3: Initialize LLM pipeline
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_length=512,
    temperature=0
)
llm = HuggingFacePipeline(pipeline=pipe)

# Step 4: Setup memory & retrieval chain if vectorstore exists
if st.session_state.vectorstore:
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Step 4a: Prompt template to avoid hallucinations
    prompt_template = """
     You are an AI assistant. Use ONLY the following context to answer the question.
     If the answer is not present in the context, reply with:
        "I don't know the answer based on the provided PDFs."

     Context: {context}

     Question: {question}

     Provide a clear, concise, and complete answer in proper sentences.
     """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    # Step 5: Chat interface
    st.subheader("Ask questions about your PDFs")
    user_question = st.text_input("Your question:")

    if user_question:
        with st.spinner("Generating answer..."):
            response = qa({"question": user_question})
        st.write(response["answer"])
