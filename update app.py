import streamlit as st
import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.schema.output_parser import StrOutputParser

# --- Page Configuration ---
st.set_page_config(
    page_title="Project Ascent AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Default Prompt Template ---
DEFAULT_PROMPT_TEMPLATE = """
You are a world-class AI research assistant for PwC. Your task is to synthesize information from the provided context to answer the user's question.
Provide a comprehensive, well-structured answer. If the context contains conflicting information, point it out.
Always cite your sources clearly.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

# --- Caching Functions for Performance ---
@st.cache_resource
def load_embedding_model():
    """Loads a lightweight and efficient sentence-transformer model."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_llm(temperature):
    """Loads the Gemini Pro model from Google, configured with the given temperature."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY environment variable not set. Please set it in your Streamlit secrets.")
        return None
    # CORRECTED MODEL NAME: Using the latest recommended model to avoid 404 errors.
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key, temperature=temperature)

# --- Core Functions ---
def get_document_text(uploaded_files):
    """Extracts text from uploaded files (PDFs and TXT)."""
    text = ""
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.name.endswith('.pdf'):
                with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                    text += "".join(page.get_text() for page in doc)
            elif uploaded_file.name.endswith('.txt'):
                text += uploaded_file.read().decode("utf-8")
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {e}")
    return text

def get_text_chunks(raw_text):
    """Splits raw text into manageable chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(raw_text)

def create_vector_store(text_chunks, embedding_model):
    """Creates a Chroma vector store from text chunks."""
    if not text_chunks: return None
    try:
        vector_store = Chroma.from_texts(texts=text_chunks, embedding=embedding_model)
        return vector_store
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None

def perform_web_search(query):
    """Performs a web search using DuckDuckGo."""
    try:
        search = DuckDuckGoSearchRun()
        return search.run(query)
    except Exception as e:
        return f"Web search failed: {e}"

# --- Main Application UI and Logic ---
def main():
    st.title("🚀 Project Ascent: Custom AI Research Agent")

    # Initialize session state variables
    if "llm_temperature" not in st.session_state:
        st.session_state.llm_temperature = 0.7
    if "prompt_template" not in st.session_state:
        st.session_state.prompt_template = DEFAULT_PROMPT_TEMPLATE
    if "enable_web_search" not in st.session_state:
        st.session_state.enable_web_search = True
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! Configure your knowledge base and agent settings, then ask me anything."}]

    # --- UI Tabs ---
    tab1, tab2, tab3 = st.tabs(["🧠 Knowledge Base", "⚙️ Agent Configuration", "💬 Chat"])

    with tab1:
        st.header("Build Your Knowledge Base")
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files. These are processed in-memory and not stored.", 
            type=["pdf", "txt"], 
            accept_multiple_files=True
        )
        if st.button("Build Knowledge Base from Files"):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    raw_text = get_document_text(uploaded_files)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        embedding_model = load_embedding_model()
                        vector_store = create_vector_store(text_chunks, embedding_model)
                        if vector_store:
                            st.session_state.vector_store = vector_store
                            st.success("Knowledge Base is ready!")
            else:
                st.warning("Please upload at least one document.")
    
    with tab2:
        st.header("Customize Your Agent")
        st.session_state.llm_temperature = st.slider(
            "LLM Temperature (Creativity)", 0.0, 1.0, st.session_state.llm_temperature, 0.05
        )
        st.session_state.enable_web_search = st.toggle(
            "Enable Live Web Search", value=st.session_state.enable_web_search
        )
        st.session_state.prompt_template = st.text_area(
            "Edit the Core System Prompt", st.session_state.prompt_template, height=300
        )
        if st.button("Reset Prompt to Default"):
            st.session_state.prompt_template = DEFAULT_PROMPT_TEMPLATE
            st.rerun()

    with tab3:
        st.header("Chat with your Custom Agent")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    context = ""
                    # Step 1: Retrieve from Knowledge Base
                    if 'vector_store' in st.session_state and st.session_state.vector_store:
                        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                        docs = retriever.get_relevant_documents(prompt)
                        if docs:
                            context += "**From your documents:**\n\n" + "\n\n".join([doc.page_content for doc in docs]) + "\n\n---\n\n"

                    # Step 2: Retrieve from Web Search
                    if st.session_state.enable_web_search:
                        web_results = perform_web_search(prompt)
                        if web_results:
                            context += f"**From live web search:**\n\n{web_results}\n\n---\n\n"

                    # Step 3: Generate Response
                    if context:
                        prompt_template = PromptTemplate(template=st.session_state.prompt_template, input_variables=["context", "question"])
                        llm = load_llm(st.session_state.llm_temperature)
                        
                        if llm:
                            chain = prompt_template | llm | StrOutputParser()
                            final_answer = chain.invoke({"context": context, "question": prompt})
                        else:
                            final_answer = "The Language Model is not available. Please check your API key."
                    else:
                        final_answer = "I don't have enough information to answer. Please build a knowledge base or enable web search."

                    st.markdown(final_answer)
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})

if __name__ == '__main__':
    main()
