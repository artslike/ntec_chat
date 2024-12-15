import streamlit as st
from langchain_community.llms import Ollama

# Initialize the LLM
llm = Ollama(model="llama3.2")

def main():
    # Streamlit UI setup
    st.title("Llama 3.2 Chatbot")

    # Initialize session state for conversation history if not already done
    if "history" not in st.session_state:
        st.session_state.history = []

    user_input = st.text_input("You:", "")

    if user_input:
        try:
            # Get response from LLM
            response = llm.invoke(user_input)
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
