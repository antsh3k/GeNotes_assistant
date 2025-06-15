import os
import json
import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing_extensions import Annotated

# LangChain imports
from langchain.agents import AgentExecutor
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain.schema.document import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="GeNotes API",
    description="API for GeNotes Genomic Guidelines Assistant",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OAuth2 setup for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Load environment variables
load_dotenv()

# Constants
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATASET_DIR = DATA_DIR / "datasets"
VECTOR_STORE_DIR = DATA_DIR / "chromadb"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "genomic_guidelines")

# Create necessary directories
for directory in [DATA_DIR, DATASET_DIR, VECTOR_STORE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Initialize embeddings model
embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
)

# Initialize Chroma vector store
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(VECTOR_STORE_DIR),
)

# Initialize chat model
llm = init_chat_model(
    os.getenv("CHAT_MODEL"),
    model_provider=os.getenv("MODEL_PROVIDER"),
    temperature=0
)

# Define the prompt template with all required variables
prompt_template = """
You are a helpful assistant. You will be provided with a query and a chat history.
Your task is to retrieve relevant information from the vector store and provide response.
For this you use the tool 'retrieve' to get the relevant information.

Query: {input}

Chat History:
{chat_history}

You can use the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
For every piece of information you provide, also provide the source.

Return text as follows:

<Answer to the question>

Source: source_url
"""

prompt = PromptTemplate.from_template(prompt_template)

# Define Pydantic models for request/response
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []

    class Config:
        json_schema_extra = {
            "example": {
                "message": "What are the guidelines for BRCA testing?",
                "history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi! How can I help you with genomic guidelines today?"}
                ]
            }
        }

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, str]] = []
    metadata: Dict[str, Any] = {}

class ScrapeRequest(BaseModel):
    url: str
    source_type: str = "website"

class CollectionInfo(BaseModel):
    name: str
    document_count: int
    embeddings: str
    is_active: bool
    created_at: str
    updated_at: str

# Define the retrieve tool
@tool
def retrieve(query: str) -> str:
    """Retrieve information related to a query from the vector store."""
    try:
        docs = vector_store.similarity_search(query, k=3)
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        return "Error retrieving information. Please try again later."

# Initialize tools and agent
tools = [retrieve]

# Create the agent with proper configuration
from langchain.agents import AgentExecutor, create_openai_tools_agent

# Create the agent with the chat model and tools
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Initialize the agent executor with consistent settings
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,  # Prevent infinite loops
    early_stopping_method="generate"  # Stop when the agent generates a final answer
)

# Initialize the agent executor with consistent settings
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
    early_stopping_method="generate",
    return_intermediate_steps=True  # To get access to tool usage details
)

# Helper functions
def format_chat_history(messages: List[Message]) -> List[BaseMessage]:
    """Convert message history to LangChain message format."""
    formatted_messages = []
    for msg in messages:
        if msg.role == "user":
            formatted_messages.append(HumanMessage(content=msg.content))
        else:
            formatted_messages.append(AIMessage(content=msg.content))
    return formatted_messages

async def generate_response(query: str, history: List[Message] = None) -> Dict[str, Any]:
    """
    Generate a response using the agent.
    
    Args:
        query: The user's query
        history: List of previous messages in the conversation
        
    Returns:
        Dictionary containing response, sources, and metadata
    """
    if history is None:
        history = []
    
    try:
        formatted_history = format_chat_history(history)
        
        # Use consistent input format for the agent
        agent_input = {
            "input": query,
            "chat_history": formatted_history
        }
        
        # Get response from agent
        response = await agent_executor.ainvoke(agent_input)
        
        # Extract sources from tool usage if available
        sources = []
        if "intermediate_steps" in response:
            for step in response["intermediate_steps"]:
                if hasattr(step[0], 'tool'):
                    sources.append({
                        "tool": step[0].tool,
                        "input": step[0].tool_input,
                        "output": str(step[1])  # Convert tool output to string
                    })
        
        # Add default source if no tools were used
        if not sources:
            sources = [{"title": "Model Knowledge Base", "description": "General knowledge"}]
        
        return {
            "response": response["output"],
            "sources": sources,
            "metadata": {
                "model": os.getenv("CHAT_MODEL", "llama3"),
                "timestamp": datetime.utcnow().isoformat(),
                "tool_calls": len(sources) > 0
            }
        }
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating response"
        )

# API Endpoints
@app.get("/")
async def read_root():
    """Root endpoint that returns a welcome message."""
    return {"message": "Welcome to GeNotes API"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    """Handle chat messages and return responses."""
    try:
        response = await generate_response(
            query=chat_request.message,
            history=chat_request.history
        )
        return response
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing your request"
        )

@app.post("/api/scrape")
async def scrape_website(request: ScrapeRequest):
    """Scrape and process a website URL."""
    try:
        # In a real implementation, you would:
        # 1. Scrape the website
        # 2. Process the content
        # 3. Add to vector store
        # 4. Return success/failure
        
        # For now, just return a success message
        return {
            "status": "success",
            "message": f"Successfully scraped {request.url}",
            "documents_processed": 1
        }
    except Exception as e:
        logger.error(f"Error scraping website: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error scraping website"
        )

@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and process files."""
    try:
        saved_files = []
        
        for file in files:
            # Save file to dataset directory
            file_path = DATASET_DIR / file.filename
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # In a real implementation, you would process the file here
            # and add its content to the vector store
            
            saved_files.append({
                "filename": file.filename,
                "size": len(content),
                "saved_path": str(file_path)
            })
        
        return {
            "status": "success",
            "message": f"Successfully uploaded {len(files)} file(s)",
            "files": saved_files
        }
    except Exception as e:
        logger.error(f"Error uploading files: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error uploading files"
        )

@app.get("/api/collections")
async def list_collections():
    """List all collections in the vector store."""
    try:
        # In a real implementation, you would get this from your vector store
        collections = [
            {
                "id": "1",
                "name": "Genomic Guidelines",
                "description": "NHS Genomic Medicine Service guidelines",
                "document_count": 42,
                "embeddings": "all-mpnet-base-v2",
                "is_active": True,
                "created_at": "2023-06-01T10:30:00Z",
                "updated_at": "2023-06-15T14:45:00Z",
            }
        ]
        return {"collections": collections}
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error listing collections"
        )

@app.get("/api/status")
async def get_status():
    """Get system status and statistics."""
    try:
        return {
            "status": "operational",
            "version": "0.1.0",
            "models": {
                "chat": os.getenv("CHAT_MODEL", "llama3"),
                "embeddings": os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
            },
            "storage": {
                "documents": len(list(DATASET_DIR.glob("*"))),
                "vector_store_size_mb": sum(f.stat().st_size for f in VECTOR_STORE_DIR.rglob('*') if f.is_file()) / (1024 * 1024),
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error getting system status"
        )

# Initialize chat history for the API
chat_history = [
    AIMessage("Hello, I'm the GeNotes Assistant. I can help you find and understand genomic clinical guidelines. How can I assist you today?")
]

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatSessionRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    stream: bool = False

class ChatSessionResponse(BaseModel):
    message: str
    session_id: str
    timestamp: str
    sources: List[Dict[str, str]] = []

class ChatSession(BaseModel):
    session_id: str
    messages: List[Dict[str, str]]
    created_at: str
    updated_at: str

# In-memory storage for chat sessions (in production, use a database)
sessions: Dict[str, List[BaseMessage]] = {}

@app.post("/api/chat", response_model=ChatSessionResponse)
async def chat_endpoint(chat_request: ChatSessionRequest):
    """
    Handle chat messages and return responses.
    
    Args:
        chat_request: The chat request containing the message and optional session_id
        
    Returns:
        ChatSessionResponse containing the assistant's response
    """
    try:
        # Get or create session
        session_id = chat_request.session_id or str(uuid.uuid4())
        if session_id not in sessions:
            sessions[session_id] = chat_history.copy()
        
        # Add user message to session
        user_message = HumanMessage(content=chat_request.message)
        sessions[session_id].append(user_message)
        
        # Generate response with consistent input format
        agent_input = {
            "input": chat_request.message,
            "chat_history": sessions[session_id][:-1]  # All messages except the current one
        }
        result = agent_executor.invoke(agent_input)
        
        ai_message = result["output"]
        assistant_message = AIMessage(content=ai_message)
        sessions[session_id].append(assistant_message)
        
        # Extract sources if available
        sources = []
        if hasattr(result, 'sources'):
            sources = result.sources
        
        return ChatSessionResponse(
            message=ai_message,
            session_id=session_id,
            timestamp=datetime.utcnow().isoformat(),
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing your request: {str(e)}"
        )

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat.
    """
    await websocket.accept()
    session_id = None
    
    try:
        # Initialize session
        session_id = str(uuid.uuid4())
        sessions[session_id] = chat_history.copy()
        
        # Send welcome message
        welcome_msg = {
            "type": "welcome",
            "session_id": session_id,
            "message": sessions[session_id][0].content
        }
        await websocket.send_json(welcome_msg)
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            if not data:
                continue
                
            try:
                # Parse message
                message = json.loads(data)
                if "message" not in message:
                    await websocket.send_json({"error": "Message is required"})
                    continue
                
                # Add user message to session
                user_message = HumanMessage(content=message["message"])
                sessions[session_id].append(user_message)
                
                # Generate response with consistent input format
                agent_input = {
                    "input": message["message"],
                    "chat_history": sessions[session_id][:-1]  # All messages except the current one
                }
                result = agent_executor.invoke(agent_input)
                
                ai_message = result["output"]
                assistant_message = AIMessage(content=ai_message)
                sessions[session_id].append(assistant_message)
                
                # Send response
                response = {
                    "type": "message",
                    "message": ai_message,
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                if hasattr(result, 'sources'):
                    response["sources"] = result.sources
                
                await websocket.send_json(response)
                
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON format"})
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
                await websocket.send_json({"error": f"Error processing message: {str(e)}"})
                
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket connection error: {str(e)}")
    finally:
        # Clean up session on disconnect (optional: persist sessions)
        if session_id and session_id in sessions:
            del sessions[session_id]

@app.get("/api/sessions/{session_id}", response_model=ChatSession)
async def get_chat_session(session_id: str):
    """
    Get chat session by ID.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "messages": [
            {"role": "assistant" if isinstance(msg, AIMessage) else "user", "content": msg.content}
            for msg in sessions[session_id]
        ],
        "created_at": "",  # Add timestamp tracking for production
        "updated_at": datetime.utcnow().isoformat()
    }

