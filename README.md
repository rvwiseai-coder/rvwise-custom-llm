# RVwise RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with LlamaCloud and Streamlit.

## Features

- 🔍 **Semantic Search**: Intelligent document retrieval using LlamaCloud's managed index
- 💬 **Interactive Chat Interface**: Clean and modern Streamlit UI
- 📚 **Source Attribution**: View relevant sources for each response
- 🚀 **Fast Performance**: Leverages LlamaCloud's optimized infrastructure
- 🎨 **Beautiful UI**: Modern, responsive design with custom styling

## Installation

1. **Clone the repository** (or navigate to the project directory):
   ```bash
   cd /home/ubuntu/workspace/rvwise-rag
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. **Create a `.env` file** in the project root with your API credentials:
   ```env
   # LlamaCloud Configuration
   LLAMA_CLOUD_API_KEY=llx-your-actual-api-key-here
   LLAMA_CLOUD_ORG_ID=14239941-881b-4efc-8ab8-9d223de0741e
   LLAMA_CLOUD_INDEX_NAME=RVwise RAG Knowledge Base
   LLAMA_CLOUD_PROJECT_NAME=Default
   
   # OpenAI Configuration (optional)
   OPENAI_API_KEY=sk-your-openai-api-key-here
   ```

2. **Alternative**: You can also modify the default values directly in `config.py`

## Usage

1. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to the URL shown (typically `http://localhost:8501`)

3. **Start chatting** with your knowledge base!

## Project Structure

```
rvwise-rag/
├── app.py              # Main Streamlit application
├── config.py           # Configuration settings
├── debug_nodes.py      # check response from test query
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (create this)
└── README.md           # This file
```

## How It Works

1. **Index Connection**: The app connects to your LlamaCloud index using the provided credentials
2. **Query Processing**: User queries are processed through the RAG pipeline:
   - Retrieval: Relevant documents are retrieved from the index
   - Generation: A response is generated based on the retrieved context
3. **Response Display**: The response and source documents are displayed in the chat interface

## Customization

### Modify Retrieval Settings
- Adjust `TOP_K_RESULTS` in the sidebar to change the number of sources retrieved
- Modify default settings in `config.py`

### Update Index Configuration
- Change index name, project name, or organization ID in the `.env` file or `config.py`

### Customize UI
- Modify the CSS styles in `app.py` to change the appearance
- Update the app title and icon in `config.py`

## Troubleshooting

### Index Connection Failed
- Verify your `LLAMA_CLOUD_API_KEY` is correct
- Ensure your index name and project name match your LlamaCloud setup
- Check your internet connection

### No Results Returned
- Verify your index contains data
- Try broader queries
- Check the LlamaCloud dashboard for index status

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Verify you're using Python 3.8 or higher

## Requirements

- Python 3.8+
- Active LlamaCloud account with configured index
- Internet connection for API calls

## Support

For issues or questions:
1. Check the LlamaCloud documentation
2. Review Streamlit documentation for UI-related questions
3. Check the error messages in the terminal for debugging

## License

This project is provided as-is for educational and development purposes.
