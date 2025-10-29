# Import all stuffs 
from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List, Iterable

import streamlit as st

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="RAG Chat Box", layout="wide")

# -----------------------------------------------------------------------------
# Imports sanity check
# -----------------------------------------------------------------------------
try:
    from openai import OpenAI
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        CharacterTextSplitter,
        MarkdownHeaderTextSplitter,
    )
    from langchain_chroma import Chroma
    from langchain_core.prompts import PromptTemplate
    from langchain_core.documents import Document

    st.success("Pre-environment setup is ready!")
except Exception as e:
    st.error(f"❌ Import error: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
API_KEY: str = os.environ.get("API_KEY", "")
BASE_URL: str = "https://api.ai.it.cornell.edu"
CHAT_MODEL: str = "openai.gpt-4o"
EMBED_MODEL: str = "openai.text-embedding-3-large"
COLLECTION_NAME: str = "document_collection"

# Increase the chunk size to avoid content fragmentation
CHUNK_SIZE_DEFAULT: int = 1200
CHUNK_OVERLAP_DEFAULT: int = 100
K_DOCUMENTS_DEFAULT: int = 10

# -----------------------------------------------------------------------------
# API key check
# -----------------------------------------------------------------------------
if not API_KEY:
    st.error("❌ API_KEY not found! Set it in terminal: export API_KEY='your_key'")
    st.stop()
else:
    st.success("API_KEY is valid!")

# -----------------------------------------------------------------------------
# Session state init
# -----------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("vectorstore", None)
ss.setdefault("uploaded_files_names", [])
ss.setdefault("temp_dir", None)
ss.setdefault("model_name", CHAT_MODEL)
ss.setdefault("temperature", 0.2)
ss.setdefault("chunk_strategy", "recursive")

# -----------------------------------------------------------------------------
# Initialize clients
# -----------------------------------------------------------------------------
try:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    if "llm" not in ss:
        ss.llm = ChatOpenAI(
            model=ss.model_name,
            temperature=ss.temperature,
            api_key=API_KEY,
            base_url=BASE_URL,
        )
    st.success("OpenAI clients initialized!")
except Exception as e:
    st.error(f"❌ Client initialization error: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _safe_filename(name: str, default_ext: str, fallback_stem: str) -> str:
    sanitized = "".join(c for c in name if c.isalnum() or c in ("_", "-", "."))
    if not sanitized:
        sanitized = f"{fallback_stem}.{default_ext}"
    p = Path(sanitized)
    if not p.suffix and default_ext:
        sanitized = f"{p.name}.{default_ext}"
    return sanitized

def _show_traceback(e: Exception) -> None:
    import traceback
    st.code(traceback.format_exc())
    st.error(f"   ❌ Error detail: {e}")

# -----------------------------------------------------------------------------
# Core functions
# -----------------------------------------------------------------------------
def load_document(file_path: str | Path, file_type: str) -> List[Document]:
    file_path = Path(file_path)
    try:
        if file_type == "txt":
            encodings: Iterable[str] = ("utf-8", "latin-1", "cp1252", "iso-8859-1")
            for enc in encodings:
                try:
                    content = file_path.read_text(encoding=enc)
                    return [Document(page_content=content, metadata={"source": file_path.name})]
                except UnicodeDecodeError:
                    continue
                except Exception:
                    continue
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            return [Document(page_content=content, metadata={"source": file_path.name})]

        elif file_type == "pdf":
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
            return docs
        else:
            return []
    except Exception as e:
        _show_traceback(e)
        return []

def _chunk_documents(all_documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    if ss.chunk_strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = splitter.split_documents(all_documents)

    elif ss.chunk_strategy == "character":
        splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = splitter.split_documents(all_documents)

    elif ss.chunk_strategy == "markdown":
        md_headers = [
            ("#", "h1"), ("##", "h2"), ("###", "h3"),
            ("####", "h4"), ("#####", "h5"), ("######", "h6"),
        ]
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=md_headers)
        tmp_docs: List[Document] = []
        for d in all_documents:
            sections = md_splitter.split_text(d.page_content)
            for s in sections:
                s.metadata.update(d.metadata)
            tmp_docs.extend(sections)

        recursive = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = recursive.split_documents(tmp_docs)

    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = splitter.split_documents(all_documents)

    return chunks

def process_documents(uploaded_files, chunk_size: int, chunk_overlap: int) -> None:
    if not uploaded_files:
        st.warning("No files uploaded")
        return
    try:
        if ss.temp_dir is None or not Path(ss.temp_dir).exists():
            ss.temp_dir = tempfile.mkdtemp()

        temp_dir = Path(ss.temp_dir)
        all_documents: List[Document] = []
        file_names: List[str] = []

        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            try:
                ext = uploaded_file.name.split(".")[-1].lower()
                safe_name = _safe_filename(uploaded_file.name, default_ext=ext, fallback_stem=f"file_{idx}")
                temp_path = temp_dir / safe_name
                temp_path.write_bytes(uploaded_file.getbuffer())

                docs = load_document(temp_path, ext)
                if docs:
                    for d in docs:
                        d.metadata["source"] = uploaded_file.name
                    all_documents.extend(docs)
                    file_names.append(uploaded_file.name)
            except Exception as e:
                _show_traceback(e)
                continue

        if not all_documents:
            st.error("❌ No documents were successfully loaded")
            return

        if ss.get("vectorstore") is not None:
            try:
                ss.vectorstore.delete_collection()
            except Exception:
                try:
                    ss.vectorstore._client.delete_collection(ss.vectorstore._collection.name)
                except Exception:
                    pass
            ss.vectorstore = None

        unique_collection = f"{COLLECTION_NAME}_{uuid.uuid4().hex[:8]}"
        chunks = _chunk_documents(all_documents, chunk_size, chunk_overlap)

        embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=API_KEY, base_url=BASE_URL)
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, collection_name=unique_collection)

        ss.vectorstore = vectorstore
        ss.uploaded_files_names = file_names

        st.success(f"✅ Documents processed successfully! {len(file_names)} file(s), {len(chunks)} chunks created.")
    except Exception as e:
        st.error(f"❌ Processing error: {e}")
        _show_traceback(e)

def retrieve_and_generate(question: str, k: int = K_DOCUMENTS_DEFAULT) -> dict:
    if ss.vectorstore is None:
        return {"answer": "Please upload and process documents first.", "sources": []}

    try:
        # If the question is "summarize/generalize", go directly to the full text summary
        if any(word in question for word in ["总结", "概括", "overview", "summarize"]):
            docs = ss.vectorstore.get()["documents"]
            sources = ss.vectorstore.get()["metadatas"]
            context = "\n\n".join(d for d in docs)

            template = (
                "You are an assistant for summarizing documents.\n"
                "Please provide a clear and concise summary of each uploaded document separately.\n\n"
                "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            )
            prompt = PromptTemplate.from_template(template)
            messages = prompt.invoke({"question": question, "context": context})
            response = ss.llm.invoke(messages)

            return {"answer": response.content, "sources": list({m.get('source','Unknown') for meta in sources for m in [meta]}), "retrieved_chunks": len(docs)}

        # General Q&A → Normal search
        retriever = ss.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        retrieved_docs: List[Document] = retriever.invoke(question)

        context = "\n\n".join(f"[From: {doc.metadata.get('source','Unknown')}]\n{doc.page_content}" for doc in retrieved_docs)

        template = (
            "You are a helpful assistant for QA.\n"
            "Based on the retrieved context, answer the question as best as possible.\n"
            "If context is limited, provide the best partial answer you can.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        prompt = PromptTemplate.from_template(template)
        messages = prompt.invoke({"question": question, "context": context})
        response = ss.llm.invoke(messages)
        sources = list({doc.metadata.get("source", "Unknown") for doc in retrieved_docs})

        return {"answer": response.content, "sources": sources, "retrieved_chunks": len(retrieved_docs)}

    except Exception as e:
        st.error(f"Error: {e}")
        _show_traceback(e)
        return {"answer": f"An error occurred: {e}", "sources": []}

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("RAG Chat Box 🤖")

with st.sidebar:
    st.header("Model Settings")
    model_options = [
        "openai.gpt-4o",
        "openai.gpt-4o-mini",
        "openai.o4-mini",
        "openai.gpt-4.1-mini",
    ]
    selected_model = st.selectbox("ChatGPT Model", options=model_options, index=model_options.index(ss.model_name) if ss.get("model_name") in model_options else 0)
    chunk_strategy = st.selectbox("Chunking Strategy", options=["recursive", "character", "markdown"], index=["recursive", "character", "markdown"].index(ss.chunk_strategy) if ss.get("chunk_strategy") else 0)
    if chunk_strategy != ss.chunk_strategy:
        ss.chunk_strategy = chunk_strategy
        st.success(f"✅ Chunking strategy switched to: {ss.chunk_strategy}")
    temperature = st.slider("Temperature", 0.0, 1.0, float(ss.temperature), 0.05)
    if (selected_model != ss.model_name) or (temperature != ss.temperature):
        ss.model_name = selected_model
        ss.temperature = temperature
        ss.llm = ChatOpenAI(model=selected_model, temperature=temperature, api_key=API_KEY, base_url=BASE_URL)
        st.success(f"✅ Switched to {selected_model} (temperature={temperature})")

st.subheader("Upload Documents")
uploaded_files = st.file_uploader("Upload .txt or .pdf files", type=["txt", "pdf"], accept_multiple_files=True)

cols = st.columns([1, 1, 6])
with cols[0]:
    if st.button("Retrieve Documents", disabled=not uploaded_files):
        process_documents(uploaded_files, CHUNK_SIZE_DEFAULT, CHUNK_OVERLAP_DEFAULT)
with cols[1]:
    if st.button("Clear Conversation"):
        ss.messages = []
        ss.vectorstore = None
        ss.uploaded_files_names = []
        # Delete the temporary directory and make sure to upload again
    if ss.temp_dir and Path(ss.temp_dir).exists():
        shutil.rmtree(ss.temp_dir)
        ss.temp_dir = None
        st.rerun()

if ss.uploaded_files_names:
    st.caption("Loaded Documents:")
    st.write("\n".join(f"✓ {name}" for name in ss.uploaded_files_names))

if not ss.vectorstore:
    st.info("Upload materials and click **Retrieve Documents** to build the index.")
else:
    st.success(f"{len(ss.uploaded_files_names)} document(s) uploaded successfully! Try now!")
    st.caption(f"Model: {ss.model_name} | Temp: {ss.temperature} | Chunking: {ss.chunk_strategy}")

for msg in ss.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("Sources"):
                st.write(f"Retrieved {msg.get('retrieved_chunks', 0)} chunks from:")
                for source in msg["sources"]:
                    st.write(f"• {source}")

if question := st.chat_input("Ask a question about the uploaded docs...", disabled=not ss.vectorstore):
    ss.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = retrieve_and_generate(question)
            st.write(result["answer"])
            if result.get("sources"):
                with st.expander("Sources"):
                    st.write(f"Retrieved {result.get('retrieved_chunks', 0)} chunks from:")
                    for source in result["sources"]:
                        st.write(f"• {source}")
    ss.messages.append({"role": "assistant", "content": result["answer"], "sources": result.get("sources", []), "retrieved_chunks": result.get("retrieved_chunks", 0)})
