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
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import Context
from config import Config

# Initialize LlamaCloud index globally
llama_index = None
rag_agent = None
ctx = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - initialize resources on startup"""
    global llama_index, rag_agent, ctx
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
            system_prompt="""You are an RV expert assistant with access to a comprehensive knowledge base about RV systems, troubleshooting, and maintenance.
Always search the knowledge base to provide accurate, helpful responses based on the available information.
If the knowledge base doesn't contain relevant information, indicate that clearly."""
        )
        
        # Create a context for the agent
        ctx = Context(rag_agent)
        
        print(f"✅ LlamaCloud Index initialized: {Config.LLAMA_CLOUD_INDEX_NAME}")
        print(f"✅ RAG Agent initialized with OpenAI model: {Config.OPENAI_MODEL}")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        llama_index = None
        rag_agent = None
    
    yield
    
    # Cleanup on shutdown if needed
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


async def generate_rag_response(query: str, messages: List[Message]) -> tuple[str, list]:
    """
    Force RAG retrieval every time using LlamaCloud
    """
    if not llama_index:
        return "I’m unable to access the knowledge base right now.", []

    try:
        from llama_index.llms.openai import OpenAI

        # 🔑 Force retrieval with explicit LLM
        query_engine = llama_index.as_query_engine(
            llm=OpenAI(
                model=Config.OPENAI_MODEL,
                api_key=Config.OPENAI_API_KEY,
            )
        )

        response = query_engine.query(query)
        answer = str(response).strip()

        if not answer or len(answer) < 30:
            return (
                "I’m having trouble pulling detailed references right now, "
                "but I can still help based on RV best practices.",
                []
            )

        return answer, []

    except Exception as e:
        print(f"❌ RAG query failed: {e}")
        return (
            "I’m having trouble accessing the knowledge base at the moment.",
            []
        )



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
        # Build conversation context from messages
        conversation_context = ""
#         if len(messages) > 1:
#             for msg in messages[:-1]:  # Exclude current query
#                 conversation_context += f"{msg.role}: {msg.content}\n"
            
#             # Enhance query with conversation context
#             enhanced_query = f"""Previous conversation:
# {conversation_context}

# Current question: {query}

# Please provide a response that takes into account the conversation history above."""
#         else:
#             enhanced_query = query
        
        # Run the agent with the query
        handler = rag_agent.run(query, ctx=ctx)
        
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
