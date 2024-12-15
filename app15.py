import os
import re
import streamlit as st
from typing import List
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.llms import Ollama

# Initialize the LLM
llm = Ollama(model="llama3.2")

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
    st.title("Llama 3.2 Chatbot with Document Retrieval")

    # Initialize session state for conversation history if not already done
    if "history" not in st.session_state:
        st.session_state.history = []

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

            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vector_store = Chroma(embedding_function=embeddings, persist_directory="chroma_db")
            ids = vector_store.add_documents(documents=all_splits)
            st.write(f"Total documents added to vector store: {len(ids)}")

            # Cache the embeddings
            embedding_cache[folder_path] = vector_store

    if 'vector_store' in locals():
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})

        user_input = st.text_input("You:", "")

        if user_input:
            try:
                # Retrieve relevant document content
                results = retriever.batch([user_input])
                retrieved_content = results[0][0].page_content if results and results[0] else "No relevant documents found."

                # Get response from LLM
                response = llm.invoke(f"{user_input}\n\nContext: {retrieved_content}")
            except Exception as e:
                response = f"Error: {e}"
            
            # Store conversation history
            st.session_state.history.append({"user": user_input, "llm": response})

        # Display conversation history
        for chat in st.session_state.history:
            st.write(f"**You**: {chat['user']}")
            st.write(f"**Llama 3.2**: {chat['llm']}")

if __name__ == "__main__":
    main()