"""
RVwise RAG Chatbot using LlamaCloud and Streamlit
"""
import streamlit as st
from llama_cloud_services import LlamaCloudIndex
from config import Config
import time
from typing import List, Dict

# Page configuration
st.set_page_config(
    page_title=Config.APP_TITLE,
    page_icon=Config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stChat {
        max-width: 100%;
    }
    .user-message {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    .source-card {
        background-color: #fff3e0;
        border-radius: 8px;
        padding: 8px;
        margin: 5px 0;
        border-left: 3px solid #ff9800;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_index():
    """Initialize and cache the LlamaCloud index"""
    try:
        Config.validate()
        index = LlamaCloudIndex(
            name=Config.LLAMA_CLOUD_INDEX_NAME,
            project_name=Config.LLAMA_CLOUD_PROJECT_NAME,
            organization_id=Config.LLAMA_CLOUD_ORG_ID,
            api_key=Config.LLAMA_CLOUD_API_KEY,
        )
        return index
    except Exception as e:
        st.error(f"Failed to initialize LlamaCloud index: {str(e)}")
        return None


def display_sources(nodes):
    """Display source documents using Streamlit components"""
    if not nodes:
        return
    
    st.markdown("#### 📚 Sources:")
    
    for i, node_with_score in enumerate(nodes, 1):
        try:
            # Extract score
            score = getattr(node_with_score, 'score', 0.0)
            score_formatted = f"{score:.6f}" if isinstance(score, float) else str(score)
            
            # Extract text content from the nested node structure
            text_content = ""
            metadata = {}
            
            if hasattr(node_with_score, 'node'):
                # It's a NodeWithScore object with nested node
                inner_node = node_with_score.node
                
                # Get text
                if hasattr(inner_node, 'text'):
                    text_content = inner_node.text
                elif hasattr(inner_node, 'get_content'):
                    text_content = inner_node.get_content()
                else:
                    text_content = str(inner_node)
                
                # Get metadata
                if hasattr(inner_node, 'metadata'):
                    metadata = inner_node.metadata
            else:
                # Try direct access
                if hasattr(node_with_score, 'text'):
                    text_content = node_with_score.text
                elif hasattr(node_with_score, 'get_content'):
                    text_content = node_with_score.get_content()
                else:
                    text_content = str(node_with_score)
                
                if hasattr(node_with_score, 'metadata'):
                    metadata = node_with_score.metadata
            
            # Clean up HTML entities in text
            text_content = text_content.replace('&#x26;', '&')
            
            # Create source display
            with st.expander(f"📄 Source {i} (Relevance: {score_formatted})", expanded=False):
                # Display metadata if available
                if metadata:
                    # Extract key metadata fields
                    if 'file_path' in metadata:
                        st.caption(f"📁 {metadata['file_path']}")
                    if 'file_name' in metadata:
                        st.caption(f"📄 File: {metadata['file_name']}")
                    if 'page_label' in metadata:
                        st.caption(f"📖 Page: {metadata['page_label']}")
                
                # Display text content
                st.markdown("**Content Preview:**")
                # Show first 500 characters
                preview_length = 500
                if len(text_content) > preview_length:
                    st.text(text_content[:preview_length] + "...")
                else:
                    st.text(text_content)
                
        except Exception as e:
            st.error(f"Error displaying source {i}: {str(e)}")


def process_query(query: str, index: LlamaCloudIndex) -> Dict:
    """Process a query using the RAG system"""
    try:
        # Retrieve relevant documents
        retriever = index.as_retriever(similarity_top_k=Config.TOP_K_RESULTS)
        nodes = retriever.retrieve(query)
        
        # Generate response using query engine
        query_engine = index.as_query_engine()
        response = query_engine.query(query)
        
        return {
            "response": str(response),
            "sources": nodes,
            "success": True
        }
    except Exception as e:
        return {
            "response": f"Error processing query: {str(e)}",
            "sources": [],
            "success": False
        }


def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def display_chat_history():
    """Display the chat history"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                display_sources(message["sources"])


def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title(f"{Config.APP_ICON} {Config.APP_TITLE}")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Display configuration status
        st.subheader("📊 System Status")
        index = initialize_index()
        if index:
            st.success("✅ LlamaCloud Index Connected")
            st.info(f"Index: {Config.LLAMA_CLOUD_INDEX_NAME}")
            st.info(f"Project: {Config.LLAMA_CLOUD_PROJECT_NAME}")
        else:
            st.error("❌ Index Connection Failed")
            st.warning("Please check your API credentials in the config.py file")
        
        st.markdown("---")
        
        # Retrieval settings
        st.subheader("🔍 Retrieval Settings")
        Config.TOP_K_RESULTS = st.slider(
            "Number of sources to retrieve",
            min_value=1,
            max_value=10,
            value=Config.TOP_K_RESULTS
        )
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
        
        # About section
        st.markdown("---")
        st.subheader("ℹ️ About")
        st.markdown("""
        This RAG chatbot uses LlamaCloud to provide intelligent responses 
        based on your indexed knowledge base.
        
        **Features:**
        - 🔍 Semantic search through documents
        - 💬 Context-aware responses
        - 📚 Source attribution
        - 🚀 Fast retrieval using LlamaCloud
        """)
    
    # Main chat interface
    if not index:
        st.error("⚠️ Please configure your LlamaCloud API credentials to use the chatbot.")
        st.stop()
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your knowledge base..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                result = process_query(prompt, index)
                
                if result["success"]:
                    st.markdown(result["response"])
                    
                    # Display sources if available
                    if result["sources"]:
                        display_sources(result["sources"])
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["response"],
                        "sources": result["sources"]
                    })
                else:
                    st.error(result["response"])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["response"],
                        "sources": []
                    })
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<center><small>Powered by LlamaCloud 🦙 and Streamlit 🚀</small></center>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
