import streamlit as st
import os
import fitz  # PyMuPDF for PDF extraction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings # Example embedding model
from langchain.llms import GooglePalm # Example LLM, can be swapped with Gemini
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from duckduckgo_search import DDGS # For the web search agentic feature

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
    # This is a high-quality, open-source embedding model.
    # In a real-world scenario, this could be hosted on a secure internal server.
    return HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

@st.cache_resource
def load_llm():
    """Loads the Large Language Model. Replace with Gemini or other preferred model."""
    # This uses Google PaLM as an example. You would substitute this with the API
    # for Gemini or another model hosted securely within PwC's environment.
    # The API key would be managed via secure environment variables, not hardcoded.
    return GooglePalm(google_api_key=os.environ.get("GOOGLE_API_KEY"))

# --- Core Functions ---
def get_document_text(uploaded_files):
    """Extracts text from uploaded files (PDFs and TXT)."""
    text = ""
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.pdf'):
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text()
        elif uploaded_file.name.endswith('.txt'):
            text += uploaded_file.read().decode("utf-8")
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
    """Creates a FAISS vector store from text chunks."""
    if not text_chunks:
        return None
    embeddings = embedding_model.embed_documents(text_chunks)
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embedding_model)
    return vector_store

def perform_web_search(query):
    """Performs a web search using DuckDuckGo and returns top results."""
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=5)]
            return "\n".join(}\nSnippet: {res['body']}\nURL: {res['href']}\n---" for res in results])
    except Exception as e:
        return f"Web search failed: {e}"

# --- Main Application UI and Logic ---
def main():
    st.title("🚀 Project Ascent: Custom AI Research Agent")
    st.markdown("### A secure, high-performance AI agent for Tax, Legal, and Audit Solutions.")

    # --- Sidebar for Controls and Knowledge Base ---
    with st.sidebar:
        st.header("1. Knowledge Base Setup")
        st.write("Upload your secure documents here. They are processed in-memory and are not stored permanently.")
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )

        if st.button("Build Knowledge Base"):
            if uploaded_files:
                with st.spinner("Processing documents... This may take a moment."):
                    raw_text = get_document_text(uploaded_files)
                    text_chunks = get_text_chunks(raw_text)
                    embedding_model = load_embedding_model()
                    vector_store = create_vector_store(text_chunks, embedding_model)
                    st.session_state.vector_store = vector_store
                    st.success("Knowledge Base is ready!")
            else:
                st.warning("Please upload at least one document.")

        st.header("2. Agentic Features")
        st.session_state.enable_web_search = st.toggle(
            "Enable Live Web Search",
            value=True,
            help="Allows the agent to search the web for real-time information."
        )

    # --- Main Chat Interface ---
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today? Please build the knowledge base first if you want to ask questions about your documents."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_text = ""
                context_sources =

                # --- Agentic Logic ---
                # Step 1: Search the secure knowledge base (if it exists)
                if 'vector_store' in st.session_state and st.session_state.vector_store:
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                    docs = retriever.get_relevant_documents(prompt)
                    if docs:
                        context_from_docs = "\n\n".join([doc.page_content for doc in docs])
                        response_text += f"**From your documents:**\n\n{context_from_docs}\n\n---\n\n"
                        context_sources.append("Internal Documents")

                # Step 2: Perform a web search (if enabled)
                if st.session_state.enable_web_search:
                    web_results = perform_web_search(prompt)
                    if web_results:
                        response_text += f"**From live web search:**\n\n{web_results}\n\n---\n\n"
                        context_sources.append("Web Search")

                # Step 3: Generate the final answer using the LLM with all context
                if response_text:
                    template = """
                    You are a world-class AI research assistant for PwC. Your task is to synthesize information from the provided context to answer the user's question.
                    Provide a comprehensive, well-structured answer. If the context contains conflicting information, point it out.
                    Always cite your sources clearly using the provided source names (e.g., 'Internal Documents', 'Web Search').

                    CONTEXT:
                    {context}

                    QUESTION:
                    {question}

                    ANSWER:
                    """
                    prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])
                    llm = load_llm()
                    chain = RetrievalQA.from_llm(llm=llm, prompt=prompt_template, retriever=st.session_state.get('vector_store', None).as_retriever() if 'vector_store' in st.session_state else None)
                    
                    # This is a simplified way to pass the combined context. A more advanced agent would be more sophisticated.
                    final_prompt = prompt_template.format(context=response_text, question=prompt)
                    final_answer = llm(final_prompt) # Direct call for synthesis
                else:
                    final_answer = "I don't have enough information to answer. Please build a knowledge base or enable web search."

                st.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})

if __name__ == '__main__':
    # To run this app:
    # 1. Make sure you have Python installed.
    # 2. Install necessary libraries: pip install streamlit PyMuPDF langchain faiss-cpu sentence-transformers google-generativeai duckduckgo-search
    # 3. Set your Google API key as an environment variable: export GOOGLE_API_KEY="YOUR_API_KEY"
    # 4. Save the code as a Python file (e.g., app.py) and run from your terminal: streamlit run app.py
    main()
