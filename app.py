import streamlit as st
import os
import shutil
import stat
from dotenv import load_dotenv

from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

# --- NEW: Load environment variables from .env file at the start ---
load_dotenv()

# --- NEW: Get API key from environment variables ---
# This works for both local .env file and Hugging Face secrets
groq_api_key = os.getenv("GROQ_API_KEY")

# --- Helper Function for Windows Permission Errors ---
def remove_readonly(func, path, _):
    """Error handler for shutil.rmtree to handle read-only files."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

# --- Page Configuration ---
st.set_page_config(
    page_title="Code Documentation RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Main App UI ---
st.title("Code Documentation RAG with Syntax Understanding 🤖")
st.write("This application allows you to chat with a GitHub repository. Enter the repository URL, and the system will process the code and let you ask questions about its implementation.")

# --- NEW: Check for API Key and stop if not found ---
if not groq_api_key:
    st.error("GROQ_API_KEY is not set! Please add it to your .env file or Hugging Face secrets.")
    st.stop()

# --- Session State Initialization ---
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processing_done" not in st.session_state:
    st.session_state.processing_done = False

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Configuration")
    repo_url = st.text_input("GitHub Repository URL:", placeholder="https://github.com/user/repo")
    process_button = st.button("Process Repository")
    st.markdown("---")
    st.markdown(
        "**Note:** Processing can take a few minutes for large repositories."
    )

# --- Main Processing Logic ---
if process_button:
    if not repo_url:
        st.error("Please enter a GitHub repository URL.")
    else:
        with st.spinner("Processing repository... This may take a moment."):
            repo_path = "./temp_repo"
            try:
                if os.path.exists(repo_path):
                    shutil.rmtree(repo_path, onerror=remove_readonly)

                loader = GitLoader(
                    clone_url=repo_url,
                    repo_path=repo_path,
                    branch="main",
                    file_filter=lambda file_path: file_path.endswith((".py", ".js", ".ts", ".java", ".md", ".html", ".css"))
                )
                documents = loader.load()
                
                if not documents:
                    st.error("No compatible files found in the repository or the 'main' branch does not exist.")
                else:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    texts = text_splitter.split_documents(documents)

                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                    vectorstore = Chroma.from_documents(texts, embeddings)
                    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})

                    llm = ChatGroq(api_key=groq_api_key, model_name="llama3-70b-8192")
                    st.session_state.conversation_chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=retriever,
                        return_source_documents=True
                    )
                    
                    st.session_state.processing_done = True
                    st.session_state.chat_history = []
                    st.success("Repository processed successfully! You can now ask questions.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                if os.path.exists(repo_path):
                    shutil.rmtree(repo_path, onerror=remove_readonly)

# --- Chat Interface ---
if st.session_state.processing_done:
    st.header("Chat with the Codebase")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question := st.chat_input("Ask a question about the repository's implementation..."):
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.spinner("Thinking..."):
            result = st.session_state.conversation_chain({
                "question": user_question,
                "chat_history": [(msg["role"], msg["content"]) for msg in st.session_state.chat_history]
            })
            answer = result["answer"]

            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
else:
    st.info("Please enter a repository URL in the sidebar and click 'Process Repository' to begin.")