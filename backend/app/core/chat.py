"""
Chat and LLM functionality using RAG with Chroma and Ollama.
"""
import logging
import os
from typing import List, Dict, Any, Optional, Union
from langchain.agents import AgentExecutor, Tool, AgentType, initialize_agent
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory

logger = logging.getLogger(__name__)

class ChatManager:
    """Manages chat interactions with the LLM using RAG."""
    
    def __init__(self, model_name: str, embedding_model: str, vector_store: Any = None):
        """
        Initialize the chat manager with RAG capabilities.
        
        Args:
            model_name: Name of the LLM to use
            embedding_model: Name of the embedding model to use
            vector_store: Vector store for document retrieval
        """
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm = self._init_llm()
        self.embeddings = self._init_embeddings()
        self.agent = self._init_agent()
    
    def _init_llm(self):
        """Initialize the language model."""
        try:
            return ChatOllama(
                model=self.model_name,
                base_url=os.getenv("OLLAMA_API_BASE_URL", "http://ollama:11434"),
                temperature=0
            )
        except Exception as e:
            logger.error(f"Error initializing language model: {str(e)}")
            raise
    
    def _init_embeddings(self):
        """Initialize the embedding model."""
        try:
            return OllamaEmbeddings(
                model=self.embedding_model,
                base_url=os.getenv("OLLAMA_API_BASE_URL", "http://ollama:11434")
            )
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            raise
    
    def _init_agent(self) -> AgentExecutor:
        """Initialize the agent with RAG capabilities."""
        try:
            # Define the system message
            system_message = """You are a helpful assistant that helps answer questions about genomic guidelines.
            You have access to a knowledge base of genomic guidelines and resources.
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            """

            # Create the retrieval tool
            def retrieve_docs(query: str) -> str:
                """
                Retrieve information related to a query from the vector store.
                
                Args:
                    query: The search query
                    
                Returns:
                    A string containing the retrieved information with sources
                """
                try:
                    # Get documents from the vector store
                    retrieved_docs = self.vector_store.similarity_search(query, k=3)
                    
                    if not retrieved_docs:
                        return "No relevant information found in the knowledge base."
                    
                    # Format the results
                    results = []
                    seen_sources = set()
                    
                    for doc in retrieved_docs:
                        try:
                            # Handle both Document objects and dictionaries
                            if hasattr(doc, 'metadata'):
                                # Document object
                                metadata = getattr(doc, 'metadata', {})
                                source = metadata.get('source', 'Unknown source')
                                title = metadata.get('title', 'No title')
                                content = getattr(doc, 'page_content', '')
                            else:
                                # Dictionary format
                                metadata = doc.get('metadata', {})
                                source = metadata.get('source', 'Unknown source')
                                title = metadata.get('title', 'No title')
                                content = doc.get('page_content', doc.get('content', ''))
                            
                            # Skip if we've already seen this source
                            if source in seen_sources:
                                continue
                                
                            seen_sources.add(source)
                            
                            # Clean and add to results
                            if content and str(content).strip():
                                result = [
                                    f"URL: {source}",
                                    f"Title: {title}",
                                    f"Content: {content.strip()}",
                                    ""  # Empty line between documents
                                ]
                                results.append("\n".join(result).strip())
                                
                        except Exception as e:
                            logger.warning(f"Error processing document: {str(e)}")
                            continue
                    
                    if not results:
                        return "No relevant information could be extracted from the documents."
                        
                    # Join all results with double newlines for better readability
                    return "\n\n".join(results)
                    
                except Exception as e:
                    error_msg = f"Error retrieving information: {str(e)}"
                    logger.error(error_msg)
                    return error_msg
            
            # Create the tool
            tools = [
                Tool(
                    name="retrieve",
                    func=retrieve_docs,
                    description="Retrieve information from the knowledge base. Input should be a search query."
                )
            ]
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            # Initialize memory
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="output"
            )
            
            # Initialize agent
            agent = initialize_agent(
                tools=tools,
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
    
    def generate_response(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Generate a response to a user query using RAG.
        
        Args:
            query: The user's query
            chat_history: List of previous messages in the conversation
            
        Returns:
            Dictionary containing the response and sources
        """
        try:
            # Format chat history for the agent
            formatted_history = []
            if chat_history:
                for msg in chat_history:
                    try:
                        if hasattr(msg, 'role') and hasattr(msg, 'content'):
                            # Handle Message object
                            role = getattr(msg, 'role', '').lower()
                            content = getattr(msg, 'content', '')
                        else:
                            # Handle dictionary
                            role = msg.get('role', '').lower() if hasattr(msg, 'get') else ''
                            content = msg.get('content', '') if hasattr(msg, 'get') else str(msg)
                        
                        if role == "user":
                            formatted_history.append(HumanMessage(content=content))
                        elif role == "assistant":
                            formatted_history.append(AIMessage(content=content))
                    except Exception as e:
                        logger.warning(f"Error formatting message: {str(e)}")
                        continue
            
            # Run the agent
            result = self.agent({"input": query, "chat_history": formatted_history})
            
            # Extract sources from the intermediate steps
            sources = set()
            response_text = result.get("output", "I'm sorry, I couldn't generate a response.")
            
            # Extract sources from the response text
            if "Source:" in response_text:
                # Extract all source lines
                source_lines = [line for line in response_text.split('\n') if line.startswith('Source:')]
                for line in source_lines:
                    # Extract the source URL or identifier
                    source = line.replace('Source:', '').strip()
                    if source and source != 'Unknown source':
                        sources.add(source)
                
                # Clean up the response text by removing source lines
                cleaned_response = '\n'.join([
                    line for line in response_text.split('\n') 
                    if not line.startswith('Source:')
                ]).strip()
            else:
                cleaned_response = response_text
            
            # Also check intermediate steps for additional sources
            if "intermediate_steps" in result:
                for step in result["intermediate_steps"]:
                    if len(step) > 1 and hasattr(step[1], 'content'):
                        content = step[1].content
                        # Extract URLs from the content
                        urls = re.findall(r'https?://[^\s\n]+', content)
                        for url in urls:
                            sources.add(url)
            
            return {
                "response": cleaned_response,
                "sources": list(sources),
                "metadata": {
                    "model": self.model_name,
                    "embedding_model": self.embedding_model
                }
            }
            
        except Exception as e:
            logger.error(f"Error in generate_response: {str(e)}")
            return {
                "response": "I apologize, but I encountered an error processing your request. Please try again later.",
                "sources": [],
                "metadata": {
                    "error": str(e)
                }
            }
