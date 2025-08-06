import streamlit as st
import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma  # Switched from FAISS to ChromaDB
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI  # Modern library for Gemini
from langchain.prompts import PromptTemplate
from duckduckgo_search import DDGS

# --- Page Configuration ---
st.set_page_config(
    page_title="Project Ascent AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching Functions for Performance ---
@st.cache_resource
def load_embedding_model():
    """Loads a powerful sentence-transformer model for creating embeddings."""
    return HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

@st.cache_resource
def load_llm():
    """Loads the Gemini Pro model from Google."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY environment variable not set. Please set it in your Streamlit secrets.")
        return None
    # Use the modern ChatGoogleGenerativeAI for Gemini
    return ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key, temperature=0.7)

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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(raw_text)

def create_vector_store(text_chunks, embedding_model):
    """Creates a Chroma vector store from text chunks."""
    if not text_chunks:
        return None
    try:
        # Chroma is a reliable, file-based vector store.
        vector_store = Chroma.from_texts(texts=text_chunks, embedding=embedding_model)
        return vector_store
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None

def perform_web_search(query):
    """Performs a web search using DuckDuckGo and returns formatted results."""
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=5)]
            return "\n".join([f"Snippet: {res.get('body', 'N/A')}\nURL: {res.get('href', 'N/A')}\n---" for res in results])
    except Exception as e:
        return f"Web search failed: {e}"

# --- Main Application UI and Logic ---
def main():
    st.title("🚀 Project Ascent: Custom AI Research Agent")
    st.markdown("### A secure, high-performance AI agent for Tax, Legal, and Audit Solutions.")

    with st.sidebar:
        st.header("1. Knowledge Base Setup")
        st.write("Upload your secure documents here. They are processed in-memory.")
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
        )

        if st.button("Build Knowledge Base"):
            if uploaded_files:
                with st.spinner("Processing documents... This may take a moment."):
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

        st.header("2. Agentic Features")
        if 'enable_web_search' not in st.session_state:
            st.session_state.enable_web_search = True
        
        st.session_state.enable_web_search = st.toggle(
            "Enable Live Web Search", value=st.session_state.enable_web_search
        )

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_context = ""

                if 'vector_store' in st.session_state and st.session_state.vector_store:
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                    docs = retriever.get_relevant_documents(prompt)
                    if docs:
                        context_from_docs = "\n\n".join([doc.page_content for doc in docs])
                        response_context += f"**From your documents:**\n\n{context_from_docs}\n\n---\n\n"

                if st.session_state.enable_web_search:
                    web_results = perform_web_search(prompt)
                    if web_results:
                        response_context += f"**From live web search:**\n\n{web_results}\n\n---\n\n"

                if response_context:
                    template = """
                    You are a world-class AI research assistant for PwC. Your task is to synthesize information from the provided context to answer the user's question.
                    Provide a comprehensive, well-structured answer. If the context contains conflicting information, point it out.
                    Always cite your sources clearly.

                    CONTEXT:
                    {context}

                    QUESTION:
                    {question}

                    ANSWER:
                    """
                    prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])
                    llm = load_llm()
                    
                    if llm:
                        chain = prompt_template | llm
                        final_answer = chain.invoke({"context": response_context, "question": prompt}).content
                    else:
                        final_answer = "The Language Model is not available. Please check your API key."
                else:
                    final_answer = "I don't have enough information to answer. Please build a knowledge base or enable web search."

                st.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})

if __name__ == '__main__':
    main()
