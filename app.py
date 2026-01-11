import streamlit as st
import os
from src.config import init_settings, DATA_DIR
from src.loader import DocumentLoader
from src.indexer import DocumentIndexer
from src.storage import IndexStorage
from src.query_engine import RAGQueryEngine
from src.evaluator import RAGEvaluator
from src.history_manager import HistoryManager
import uuid

# Page Configuration
st.set_page_config(page_title="Modular RAG Pro", page_icon="🏗️", layout="wide")

# Lucide Icons Integration
st.markdown('<script src="https://unpkg.com/lucide@latest"></script>', unsafe_allow_html=True)
st.markdown('<script>lucide.createIcons();</script>', unsafe_allow_html=True)

# Custom Styling (Professional Light Mode)
st.markdown("""
<style>
    /* Clean Light Palette */
    :root {
        --primary: #0ea5e9; /* Professional Light Blue */
        --primary-soft: rgba(14, 165, 233, 0.1);
        --secondary: #0284c7;
        --bg-main: #ffffff;
        --bg-sidebar: #f8fafc;
        --text-base: #1e293b;
        --text-muted: #64748b;
        --border-color: #e2e8f0;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    /* Base Styling */
    .stApp {
        background-color: var(--bg-main);
        color: var(--text-base);
    }

    /* Clean Chat Containers */
    .stChatMessage {
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        background: var(--bg-main);
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-sm);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stChatMessage:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
        border-color: var(--primary);
    }

    /* Modern Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: 6px;
        margin-right: 8px;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        border: 1px solid transparent;
    }
    .badge-pass {
        background: #f0fdf4;
        color: #16a34a;
        border-color: #bbf7d0;
    }
    .badge-fail {
        background: #f8fafc;
        color: var(--text-muted);
        border-color: var(--border-color);
    }

    /* Icons */
    .li-icon {
        width: 16px;
        height: 16px;
        margin-right: 8px;
        vertical-align: middle;
        stroke: var(--primary);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-color);
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: var(--text-base) !important;
        font-weight: 700;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--primary); }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
@st.cache_resource(show_spinner="Warming up logic engine...")
def load_rag_assets():
    """Cached initialization of index and query engine."""
    index = IndexStorage.load_index()
    engine = RAGQueryEngine(index) if index else None
    return index, engine

if "initialized" not in st.session_state:
    if init_settings():
        st.session_state.initialized = True
        st.session_state.index, st.session_state.query_engine = load_rag_assets()
        st.session_state.current_session_id = str(uuid.uuid4())
    else:
        st.error("GROQ_API_KEY not found. Please check your .env file.")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: Parsing, Loading, Storing, and ARCHIVE
with st.sidebar:
    st.markdown('### <i data-lucide="folder-open" class="li-icon"></i> Workspace', unsafe_allow_html=True)
    
    # 1. ARCHIVE OF CHATS
    st.markdown('#### <i data-lucide="history" class="li-icon"></i> Chat Archive', unsafe_allow_html=True)
    sessions = HistoryManager.list_sessions()
    
    # "New Chat" button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state.current_session_id = str(uuid.uuid4())
        st.session_state.messages = []
        if st.session_state.query_engine:
            st.session_state.query_engine.memory.reset()
        st.rerun()

    for s in sessions:
        col1, col2 = st.columns([0.8, 0.2])
        is_active = s["id"] == st.session_state.current_session_id
        
        with col1:
            if st.button(
                f"{s['preview']}", 
                key=f"load_{s['id']}", 
                use_container_width=True,
                type="secondary" if not is_active else "primary"
            ):
                st.session_state.current_session_id = s["id"]
                st.session_state.messages = HistoryManager.load_session(s["id"])
                # Also reset query engine memory to this history if needed (simplified here)
                if st.session_state.query_engine:
                    st.session_state.query_engine.memory.reset()
                st.rerun()
        
        with col2:
            if st.button("🗑️", key=f"del_{s['id']}", use_container_width=True):
                HistoryManager.delete_session(s["id"])
                if is_active:
                    st.session_state.current_session_id = str(uuid.uuid4())
                    st.session_state.messages = []
                st.rerun()

    st.divider()

    with st.expander("🛠️ Data Pipeline", expanded=False):
        st.markdown('#### <i data-lucide="upload-cloud" class="li-icon"></i> Source Management', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload sources", 
            type=["pdf", "txt", "md", "docx", "csv"], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(DATA_DIR, uploaded_file.name)
                if os.path.exists(file_path):
                    st.warning(f"⏩ Skipped: `{uploaded_file.name}` already exists.")
                else:
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"✅ Loaded: `{uploaded_file.name}`")

        allowed_exts = st.multiselect(
            "Allowed Extensions",
            [".pdf", ".txt", ".md", ".docx", ".csv"],
            default=[".pdf", ".txt", ".md"],
            help="Specify which file extensions to parse from the data directory."
        )

        if st.button("🏗️ Build/Update Index", use_container_width=True):
            with st.spinner("Indexing documents..."):
                docs = DocumentLoader.load_from_data_dir(required_exts=allowed_exts)
                if docs:
                    st.session_state.index = DocumentIndexer.create_index(docs)
                    IndexStorage.persist_index(st.session_state.index)
                    st.session_state.query_engine = RAGQueryEngine(st.session_state.index)
                    st.success("Index stored!")
                else:
                    st.warning("No documents foundation.")

# Main Chat: Querying & Evaluation
st.markdown('# <i data-lucide="bot" class="li-icon" style="width: 40px; height: 40px;"></i> RAG Archive Analyst', unsafe_allow_html=True)
st.caption(f"🛡️ Secure Session: {st.session_state.current_session_id}")

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "eval" in msg:
            e = msg["eval"]
            f_class = "badge-pass" if e["faithfulness"] else "badge-fail"
            r_class = "badge-pass" if e["relevancy"] else "badge-fail"
            st.markdown(f"""
                <div style="display: flex; margin-top: 10px; gap: 8px;">
                    <span class="badge {f_class}"><i data-lucide="shield-check" class="li-icon"></i>Faithful: {e['faithfulness']}</span>
                    <span class="badge {r_class}"><i data-lucide="target" class="li-icon"></i>Relevant: {e['relevancy']}</span>
                </div>
            """, unsafe_allow_html=True)

# Input
if prompt := st.chat_input("Query your documents..."):
    if not st.session_state.query_engine:
        st.warning("Please build the index first using the sidebar!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Retrieving & Analyzing..."):
                # Use the new high-precision method
                response = st.session_state.query_engine.query_with_precision(prompt)
                
                # Perform Evaluation
                eval_results = RAGEvaluator.evaluate(prompt, response)
                
                content = str(response)
                st.markdown(content)
                
                # Show evaluation markers
                f_class = "badge-pass" if eval_results["faithfulness"] else "badge-fail"
                r_class = "badge-pass" if eval_results["relevancy"] else "badge-fail"
                st.markdown(f"""
                    <div style="display: flex; margin-top: 10px; gap: 8px;">
                        <span class="badge {f_class}"><i data-lucide="shield-check" class="li-icon"></i>Faithful: {eval_results['faithfulness']}</span>
                        <span class="badge {r_class}"><i data-lucide="target" class="li-icon"></i>Relevant: {eval_results['relevancy']}</span>
                    </div>
                """, unsafe_allow_html=True)
                
                # Update history session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": content,
                    "eval": eval_results
                })
                
                # PERSIST TO ARCHIVE
                HistoryManager.save_session(
                    st.session_state.current_session_id, 
                    st.session_state.messages
                )
