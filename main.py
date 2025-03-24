###########################################################################
## Chatbot that retrieves a list of small CVs using RAG + LLM
## This project is intended to showcase an MVP for a smart Chatbot.
## Developed under the Caleidos interview process for GenAI.
## All rights reserved to the Author: Errol.mamani@ucsp.edu.pe,
## https://wild10.github.io/
###########################################################################

# Import necessary libraries
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import requests
import fitz  # PyMuPDF
import boto3
import re
from aws_boto3 import load_pdfs, generate_presigned_urls_for_bucket
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama.chat_models import ChatOllama
import gradio as gr

# Generate pre-signed URLs from S3 and load PDFs
bucket_name = 'cvs-caleidos-wild22032025'
urls = generate_presigned_urls_for_bucket(bucket_name)
pdfs_text = load_pdfs(urls)

# Split documents into smaller chunks for efficient processing
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
splitted_documents = text_splitter.create_documents(pdfs_text)

# Convert Document objects to a list of strings
list_texts = [doc.page_content for doc in splitted_documents]    

# Generate embeddings using a Spanish model
spanish_embed = OllamaEmbeddings(model='jina/jina-embeddings-v2-base-es')
vectors = spanish_embed.embed_documents(list_texts)

# Initialize Pinecone client
pc = Pinecone(api_key="put your API key ")
index_name = "spanish-test-index"

# Create index if it doesn't exist
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=768,  # Match Spanish embedding dimensions
        metric="cosine",
        spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
    )

# Initialize vector store and add documents
index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=spanish_embed)
vector_store.add_documents(documents=splitted_documents)

# Set up Retrieval-Augmented Generation (RAG) with LLM
retriever = vector_store.as_retriever(search_type="similarity")
llm = ChatOllama(model="llama3")
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

# Function to handle chatbot interaction
def chat_with_llm(message, history):
    prompt = f"You are a helpful assistant that responds in Spanish considering all uploaded PDFs. {message}"
    result = qa.invoke({"query": prompt})
    clean_response = re.sub(r'<think>.*?</think>', '', result['result'], flags=re.DOTALL).strip()
    print(clean_response)
    return clean_response

# Launch chatbot UI
gr.ChatInterface(
    fn=chat_with_llm, 
    type="messages"
).launch()
