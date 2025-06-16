"""
Chat and LLM functionality.
"""
import logging
import os
from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor, Tool, AgentType, initialize_agent
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    MessagesPlaceholder, 
    HumanMessagePromptTemplate, 
    ChatPromptTemplate,
    SystemMessagePromptTemplate
)
from langchain.schema import Document
from langchain.vectorstores import Chroma
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ChatManager:
    """Manages chat interactions with the LLM."""
    
    def __init__(self, model_name: str, model_provider: str, tools: Optional[list] = None, vector_store: Any = None):
        """
        Initialize the chat manager.
        
        Args:
            model_name: Name of the LLM to use
            model_provider: Provider of the LLM (e.g., 'ollama', 'openai')
            tools: List of tools the agent can use
            vector_store: Vector store for document retrieval
        """
        self.model_name = model_name
        self.model_provider = model_provider
        self.vector_store = vector_store
        self.tools = tools or []
        self.llm = self._init_llm()
        self.agent = self._init_agent()
        
        # Initialize chat history with a welcome message
        self.chat_history = [
            AIMessage(content="Hello, I'm the GeNotes Assistant. I can help you find and understand genomic clinical guidelines. How can I assist you today?")
        ]
    
    def _init_llm(self):
        """Initialize the language model."""
        try:
            if self.model_provider.lower() == "ollama":
                return ChatOllama(
                    model=self.model_name,
                    base_url=os.getenv("OLLAMA_API_BASE_URL", "http://ollama:11434"),
                    temperature=0.1
                )
            else:
                raise ValueError(f"Unsupported model provider: {self.model_provider}")
        except Exception as e:
            logger.error(f"Error initializing language model: {str(e)}")
            raise
    
    def _init_agent(self) -> AgentExecutor:
        """Initialize the agent with tools."""
        try:
            # Define the system message
            system_message = """You are a helpful assistant that helps answer questions about genomic guidelines.
            You have access to a knowledge base of genomic guidelines and resources.
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            """

            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_message),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            
            # Initialize memory
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="output"
            )
            
            # Initialize agent
            agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                verbose=True,
                memory=memory,
                return_intermediate_steps=True,
                handle_parsing_errors=True,
                max_iterations=3,
                agent_kwargs={
                    "system_message": system_message,
                    "output_parser": None
                }
            )
            
            return agent
        except Exception as e:
            logger.error(f"Error initializing agent: {str(e)}")
            raise
    
    def format_chat_history(self, messages: List[Dict[str, str]]) -> List[BaseMessage]:
        """
        Convert message history to LangChain message format.
        
        Args:
            messages: List of messages in format [{"role": "user", "content": "..."}, ...]
            
        Returns:
            List of LangChain message objects
        """
        formatted_messages = []
        for msg in messages:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            
            if role == "user":
                formatted_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                formatted_messages.append(AIMessage(content=content))
        
        return formatted_messages
    
    def retrieve_documents(self, query: str, k: int = 3) -> List[Document]:
        """
        Retrieve relevant documents from the vector store.
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if not self.vector_store:
            return []
            
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def generate_response(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Generate a response to a user query.
        
        Args:
            query: The user's query
            chat_history: List of previous messages in the conversation
            
        Returns:
            Dictionary containing the response, sources, and metadata
        """
        try:
            # Format chat history
            formatted_history = self.format_chat_history(chat_history or [])
            
            # Retrieve relevant documents
            docs = self.retrieve_documents(query)
            
            # Add context from documents if available
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Prepare the input with context
            if context:
                query_with_context = f"Context: {context}\n\nQuestion: {query}"
            else:
                query_with_context = query
            
            # Add current query to history
            current_message = HumanMessage(content=query)
            
            # Run the agent
            result = self.agent({
                "input": query_with_context,
                "chat_history": formatted_history
            })
            
            # Extract sources from documents
            sources = []
            for i, doc in enumerate(docs, 1):
                if hasattr(doc, 'metadata'):
                    source = doc.metadata.get('source', f'Document {i}')
                    sources.append({
                        'source': source,
                        'content': doc.page_content[:200] + '...'  # First 200 chars as preview
                    })
            
            # Prepare response
            response = {
                "response": result.get("output", "I'm sorry, I couldn't generate a response."),
                "sources": sources,
                "metadata": {
                    "model": self.model_name,
                    "provider": self.model_provider,
                    "context_used": bool(context)
                }
            }
            
            # Update chat history
            self.chat_history.extend([current_message, AIMessage(content=response["response"])])
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "response": "I'm sorry, I encountered an error processing your request.",
                "sources": [],
                "metadata": {
                    "error": str(e),
                    "model": self.model_name,
                    "provider": self.model_provider
                }
            }
