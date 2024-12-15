"""
%pip install streamlit
%pip install langchain-community
%pip install langchain-core
%pip install langchain
%pip install asyncio
%pip install python-dotenv
"""




import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

async def get_conversational_answer(retriever, input, chat_history):
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    llm = ChatOllama(model="llama3.2")
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = """You are an assistant for question-answering related to medical network interface (MNI) projects and the project life cycle. \
    Use the retrieved context below to answer the question. \
    If specific information is not available, advise consulting the local hospital IT department for guidance; only direct to local IT department, do not refer to 'relevant authorities' or 'someone'. \
    Avoid vague references; provide solid and confident answers. \
    Avoid mentioning explicitly in the provided context. \
    Avoid mentioning you could not find any direct information. \
    Keep responses concise, with a maximum of seven sentences. \
{context}"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    ai_msg = rag_chain.invoke({"input": input, "chat_history": chat_history})
    return ai_msg

def main():
    st.header('NTEC MNI Helpdesk')

    # Initialize session state variables if not already set
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "activate_chat" not in st.session_state:
        st.session_state.activate_chat = False

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.session_state.chat_history = []

    # Display previous messages in the chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message['avatar']):
            st.markdown(message["content"])

    embed_model = OllamaEmbeddings(model='mxbai-embed-large')

    with st.sidebar:
        st.subheader('Upload Your PDF File')
        docs = st.file_uploader('⬆️ Upload your PDF & Click to process', accept_multiple_files=True, type=['pdf'])

        if st.button('Process'):
            if docs:
                os.makedirs('./data', exist_ok=True)

                for doc in docs:
                    save_path = os.path.join('./data', doc.name)
                    with open(save_path, 'wb') as f:
                        f.write(doc.getbuffer())
                    st.write(f'Processed file: {save_path}')

                with st.spinner('Processing'):
                    loader = PyPDFDirectoryLoader("./data")
                    documents = loader.load()
                    vector_store = FAISS.from_documents(documents, embed_model)
                    retriever = vector_store.as_retriever()

                    if "retriever" not in st.session_state:
                        st.session_state.retriever = retriever

                    st.session_state.activate_chat = True

                # Delete uploaded PDF files after loading
                for doc in os.listdir('./data'):
                    os.remove(os.path.join('./data', doc))

                # Provide feedback on processing completion
                st.success('PDF files processed successfully! You can now ask questions.')

    # Chat functionality
    if st.session_state.activate_chat:
        prompt = st.chat_input("Ask your question concerning MNI?")

        if prompt:
            # Display user message
            with st.chat_message("user", avatar='👨🏻'):
                st.markdown(prompt)

            # Append user message to session state
            st.session_state.messages.append({"role": "user", "avatar": '👨🏻', "content": prompt})

            retriever = st.session_state.retriever
            
            try:
                # Get AI response asynchronously
                ai_msg = asyncio.run(get_conversational_answer(retriever, prompt, st.session_state.chat_history))
                
                # Update chat history with user and AI messages
                cleaned_response = ai_msg["answer"]
                st.session_state.chat_history.extend([HumanMessage(content=prompt), cleaned_response])

                # Display AI response
                with st.chat_message("assistant", avatar='🤖'):
                    st.markdown(cleaned_response)

                # Append AI message to session state
                st.session_state.messages.append({"role": "assistant", "avatar": '🤖', "content": cleaned_response})
            
            except Exception as e:
                # Handle any errors that occur during the AI response generation
                error_message = f"An error occurred while processing your request: {str(e)}"
                st.error(error_message)
                
                # Optionally log error details to a file (for debugging purposes)
                with open('error_log.txt', 'a') as log_file:
                    log_file.write(error_message + '\n')
        
        else:
            # Prompt user to upload PDFs if no input is provided
            st.markdown('Please upload your PDFs to chat.')

if __name__ == '__main__':
    main()