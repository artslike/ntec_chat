import os
import re
import streamlit as st
from typing import List
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# Cache to store embeddings
embedding_cache = {}

def load_documents_from_folder(folder_path: str) -> List[Document]:
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    file_documents = [Document(page_content=sentence) for sentence in re.split(r'\n\n', text)]
                    documents.extend(file_documents)
                    st.write(f"Loaded {len(file_documents)} documents from {file_path}")
            except Exception as e:
                st.write(f"Error reading {file_path}: {e}")
    return documents

def main():
    st.title("Document Chatbot")
    st.write("Chat with the documents loaded from the specified folder.")

    folder_path = st.text_input("Enter the folder path containing the documents:", 'docs_mni/raw')
    if st.button("Load Documents"):
        if folder_path in embedding_cache:
            st.write("Embeddings already loaded from cache.")
            vector_store = embedding_cache[folder_path]
        else:
            documents = load_documents_from_folder(folder_path)
            st.write(f"Total documents loaded: {len(documents)}")

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0, add_start_index=True)
            all_splits = text_splitter.split_documents(documents)
            st.write(f"Total splits created: {len(all_splits)}")

            model = ChatOllama(model="llama3.2", temperature=0)
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vector_store = Chroma(embedding_function=embeddings, persist_directory="chroma_db")  # Ensure persistence
            ids = vector_store.add_documents(documents=all_splits)
            st.write(f"Total documents added to vector store: {len(ids)}")

            # Cache the embeddings
            embedding_cache[folder_path] = vector_store

        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})

        query = st.text_input("Enter your query:")
        if st.button("Get Response"):
            results = retriever.batch([query])
            st.write(f"Retriever results: {results}")

if __name__ == "__main__":
    main()