"""
FastAPI server providing OpenAI-compatible chat completions endpoint for Vapi.ai
Integrates with LlamaCloud RAG for knowledge-based responses
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, AsyncGenerator
import json
import time
import uuid
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

# Import LlamaCloud components
from llama_cloud_services import LlamaCloudIndex
from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult, AgentStream
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.memory import Memory
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from config import Config
import hashlib

# Initialize LlamaCloud index globally
llama_index = None
rag_agent = None

# Memory cache to avoid recreating memory for each request
memory_cache = {}
memory_last_access = {}
MEMORY_CACHE_TTL = 1800  # 30 minutes in seconds


async def cleanup_old_memories():
    """Background task to clean up old memory cache entries"""
    while True:
        try:
            current_time = time.time()
            sessions_to_remove = []
            
            for session_id, last_access in memory_last_access.items():
                if current_time - last_access > MEMORY_CACHE_TTL:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                if session_id in memory_cache:
                    del memory_cache[session_id]
                    del memory_last_access[session_id]
                    print(f"🗑️ Cleaned up expired memory cache for session: {session_id}")
            
            await asyncio.sleep(300)  # Check every 5 minutes
            
        except Exception as e:
            print(f"Error in memory cleanup: {e}")
            await asyncio.sleep(300)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - initialize resources on startup"""
    global llama_index, rag_agent
    
    # Start background cleanup task
    cleanup_task = asyncio.create_task(cleanup_old_memories())
    
    try:
        Config.validate()
        llama_index = LlamaCloudIndex(
            name=Config.LLAMA_CLOUD_INDEX_NAME,
            project_name=Config.LLAMA_CLOUD_PROJECT_NAME,
            organization_id=Config.LLAMA_CLOUD_ORG_ID,
            api_key=Config.LLAMA_CLOUD_API_KEY,
        )
        
        # Create query engine tool
        query_engine = llama_index.as_query_engine(streaming=True)
        query_engine_tool = QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name="rvwise_knowledge_base", 
                description="Search the RVwise knowledge base for information about RV systems, troubleshooting, and maintenance"
            )
        )
        
        # Initialize the agent with the query engine tool
        rag_agent = FunctionAgent(
            llm=OpenAI(model=Config.OPENAI_MODEL, api_key=Config.OPENAI_API_KEY, streaming=True),
            tools=[query_engine_tool],
            system_prompt="""You are an assistant dedicated exclusively to answering questions about RVs (Recreational Vehicles), RV maintenance, RV travel, or RV accessories. For any question not directly related to these topics, politely respond: "I'm here to help with RV-related questions. Please ask about RVs, RV travel, or RV maintenance." For inappropriate or offensive questions, answer: "I'm sorry, I can't assist with that request." Do not answer unrelated or inappropriate questions.
Give me a short, simple answer like a real person would, making sure to include all the necessary information."""
        )
        
        print(f"✅ LlamaCloud Index initialized: {Config.LLAMA_CLOUD_INDEX_NAME}")
        print(f"✅ RAG Agent initialized with OpenAI model: {Config.OPENAI_MODEL}")
        print(f"✅ Memory cache cleanup task started")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        llama_index = None
        rag_agent = None
    
    yield
    
    # Cleanup on shutdown
    cleanup_task.cancel()
    print("🔄 Shutting down API server...")

# Create FastAPI app with lifespan management
app = FastAPI(
    title="RVwise RAG API for Vapi.ai",
    description="OpenAI-compatible chat completions endpoint with LlamaCloud RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for Vapi.ai access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for OpenAI-compatible API
class Message(BaseModel):
    role: str = Field(..., description="The role of the message author (system, user, assistant)")
    content: str = Field(..., description="The content of the message")
    name: Optional[str] = Field(None, description="Optional name of the message author")

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="rvwise-rag", description="Model to use for completion")
    messages: List[Message] = Field(..., description="List of messages in the conversation")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    n: Optional[int] = Field(1, ge=1, description="Number of completions to generate")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    user: Optional[str] = Field(None, description="Unique identifier for the user")

class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


def extract_user_query(messages: List[Message]) -> str:
    """Extract the latest user query from messages"""
    # Get the last user message
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    
    # If no user message, try to get the last message
    if messages:
        return messages[-1].content
    
    return ""


def get_session_id_from_messages(messages: List[Message]) -> str:
    """Generate a unique session ID based on the conversation context"""
    # Create a hash from the first few messages to identify the session
    session_data = ""
    for msg in messages[:3]:  # Use first 3 messages for session identification
        session_data += f"{msg.role}:{msg.content[:100]}"
    
    # Generate a short hash for the session ID
    session_hash = hashlib.sha256(session_data.encode()).hexdigest()[:16]
    return f"session_{session_hash}"


def get_or_update_memory(messages: List[Message]) -> Memory:
    """Get cached memory or create/update if needed"""
    global memory_cache, memory_last_access
    
    # Generate session ID from messages
    session_id = get_session_id_from_messages(messages)
    
    # Update last access time
    memory_last_access[session_id] = time.time()
    
    # Check if we have a cached memory for this session
    if session_id in memory_cache:
        cached_memory, cached_message_count = memory_cache[session_id]
        
        # Check if the message count matches (no new messages)
        if cached_message_count == len(messages):
            print(f"♻️ Using cached memory for session: {session_id}")
            return cached_memory
        
        # We have new messages, update the existing memory
        print(f"📝 Updating memory for session: {session_id} ({cached_message_count} -> {len(messages)} messages)")
        
        # Add only the new messages to the existing memory
        new_messages = []
        for msg in messages[cached_message_count:]:
            new_messages.append(ChatMessage(role=msg.role, content=msg.content))
        
        if new_messages:
            cached_memory.put_messages(new_messages)
        
        # Update cache with new message count
        memory_cache[session_id] = (cached_memory, len(messages))
        return cached_memory
    
    # No cached memory, create new one
    print(f"🆕 Creating new memory for session: {session_id}")
    memory = Memory.from_defaults(session_id=session_id, token_limit=40000)
    
    # Convert all messages to ChatMessages
    chat_messages = []
    for msg in messages:
        chat_messages.append(ChatMessage(role=msg.role, content=msg.content))
    
    # Put all messages into memory
    if chat_messages:
        memory.put_messages(chat_messages)
    
    # Cache the memory with message count
    memory_cache[session_id] = (memory, len(messages))
    
    return memory


def format_context_from_sources(nodes) -> str:
    """Format retrieved sources into context for the LLM"""
    if not nodes:
        return ""
    
    context_parts = []
    for i, node_with_score in enumerate(nodes, 1):
        try:
            # Extract text content from node
            text_content = ""
            if hasattr(node_with_score, 'node'):
                inner_node = node_with_score.node
                if hasattr(inner_node, 'text'):
                    text_content = inner_node.text
                elif hasattr(inner_node, 'get_content'):
                    text_content = inner_node.get_content()
            elif hasattr(node_with_score, 'text'):
                text_content = node_with_score.text
            
            # Clean up HTML entities
            text_content = text_content.replace('&#x26;', '&')
            
            # Add to context with source number
            if text_content:
                context_parts.append(f"[Source {i}]:\n{text_content}\n")
        except Exception:
            continue
    
    return "\n".join(context_parts)


async def generate_rag_response(query: str, messages: List[Message]) -> tuple[str, List[Any]]:
    """Generate response using RAG agent (non-streaming)"""
    print(f"Generating response using RAG Agent for query: {query}")
    if not rag_agent or not llama_index:
        return "RAG system is not available. Please check the configuration.", []
    
    try:
        chat_history = None
        
        if len(messages) > 1:
            # Get or update cached memory (excluding current query)
            memory = get_or_update_memory(messages[:-1])
            chat_history = memory.get()
            print(f"📚 Using chat history with {len(chat_history)} messages")
        
        # Run the agent with the query and chat history
        if chat_history:
            # Pass chat history to override any existing memory
            handler = await rag_agent.run(query, chat_history=chat_history)
        else:
            # No history, just run with the query
            handler = await rag_agent.run(query)
        
        # Collect the full response from the stream
        full_response = str(handler)
        
        # For now, we don't have direct access to nodes from the agent
        # This would require modifying the agent to expose the retrieved nodes
        nodes = []
        
        return full_response, nodes
        
    except Exception as e:
        print(f"❌ Error generating response: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error generating response: {str(e)}", []


def estimate_tokens(text: str) -> int:
    """Rough estimation of token count (approximately 4 characters per token)"""
    return len(text) // 4


async def stream_rag_response(query: str, messages: List[Message], request_id: str, model: str) -> AsyncGenerator[str, None]:
    """Stream response using RAG agent with LlamaCloud"""
    if not rag_agent or not llama_index:
        error_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": "RAG system is not available. Please check the configuration."},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        return
    
    try:
        chat_history = None
        
        if len(messages) > 1:
            # Get or update cached memory (excluding current query)
            memory = get_or_update_memory(messages[:-1])
            chat_history = memory.get()
            print(f"📚 Streaming with chat history containing {len(chat_history)} messages")
        
        # Run the agent with the query and chat history
        if chat_history:
            # Pass chat history to override any existing memory
            handler = rag_agent.run(query, chat_history=chat_history)
        else:
            # No history, just run with the query
            handler = rag_agent.run(query)
        
        # Stream the response chunks
        async for ev in handler.stream_events():
            if isinstance(ev, ToolCallResult):
                # Log tool calls for debugging
                print(f"\n🔧 Tool Call: {ev.tool_name}")
                print(f"   Args: {ev.tool_kwargs}")
                # Tool results are not sent to client, only logged
            elif isinstance(ev, AgentStream):
                # Stream the actual response text
                if ev.delta:  # Only send if there's content
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": ev.delta},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    await asyncio.sleep(0.001)  # Small delay to prevent overwhelming the client
        
        # Add source attribution if we have sources
        # if nodes and len(nodes) > 0:
        #     attribution_chunk = {
        #         "id": request_id,
        #         "object": "chat.completion.chunk",
        #         "created": int(time.time()),
        #         "model": model,
        #         "choices": [{
        #             "index": 0,
        #             "delta": {"content": ""},
        #             "finish_reason": None
        #         }]
        #     }
        #     yield f"data: {json.dumps(attribution_chunk)}\n\n"
        
        # Send final chunk
        final_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        print(f"❌ Streaming error: {str(e)}")
        error_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": f"\n\nError: {str(e)}"},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "RVwise RAG API",
        "description": "OpenAI-compatible chat completions endpoint with LlamaCloud RAG",
        "endpoints": {
            "/chat/completions": "POST - Chat completions endpoint (OpenAI-compatible)",
            "/health": "GET - Health check endpoint",
            "/models": "GET - List available models"
        },
        "documentation": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if llama_index else "degraded",
        "llama_index_initialized": llama_index is not None,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/models")
async def list_models():
    """List available models (OpenAI-compatible endpoint)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "rvwise-rag",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "rvwise",
                "permission": [],
                "root": "rvwise-rag",
                "parent": None
            }
        ]
    }


@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint for Vapi.ai
    Processes chat messages and returns RAG-enhanced responses
    """
    try:
        # Extract user query from messages
        user_query = extract_user_query(request.messages)
        
        if not user_query:
            raise HTTPException(status_code=400, detail="No user message found in request")
        
        # Generate unique ID for this completion
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        
        # Handle streaming response
        if request.stream:
            print(f"Streaming response for query: {user_query}")
            return StreamingResponse(
                stream_rag_response(user_query, request.messages, completion_id, request.model),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"  # Disable nginx buffering
                }
            )
        
        # Generate RAG response (async)
        response_text, sources = await generate_rag_response(user_query, request.messages)
        
        # Non-streaming response
        prompt_tokens = sum(estimate_tokens(msg.content) for msg in request.messages)
        completion_tokens = estimate_tokens(response_text)
        
        response = ChatCompletionResponse(
            id=completion_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
        
        print(f"✅ Response: {response}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/v1/chat/completions")
async def chat_completions_v1(request: ChatCompletionRequest):
    """Alias for /chat/completions with v1 prefix (OpenAI compatibility)"""
    return await chat_completions(request)


if __name__ == "__main__":
    import uvicorn
    
    # Run the FastAPI server
    uvicorn.run(
        app,
        host="0.0.0.0",  # Listen on all interfaces
        port=Config.PORT,        # Default port, can be changed
        log_level="info"
    )
