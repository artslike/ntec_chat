import os
import re
from typing import List

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

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
                    print(f"Loaded {len(file_documents)} documents from {file_path}")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return documents

def main():
    model = ChatOllama(model="llama3.2", temperature=0)
    folder_path = 'docs_mni/raw'
    documents = load_documents_from_folder(folder_path)
    print(f"Total documents loaded: {len(documents)}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0, add_start_index=True)
    all_splits = text_splitter.split_documents(documents)
    print(f"Total splits created: {len(all_splits)}")

    responses = [model.invoke(split.page_content) for split in all_splits]
    print(f"Total responses received: {len(responses)}")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectors = [embeddings.embed_query(split.page_content) for split in all_splits]
    print(f"Total vectors created: {len(vectors)}")

    vector_store = Chroma(embedding_function=embeddings)
    ids = vector_store.add_documents(documents=all_splits)
    print(f"Total documents added to vector store: {len(ids)}")

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    queries = ["How many hospitals in NTEC?", "how to apply for data port request?"]
    results = retriever.batch(queries)
    print(f"Retriever results: {results}")

if __name__ == "__main__":
    main()
