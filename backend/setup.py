from setuptools import setup, find_packages

setup(
    name="genotes_backend",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Core dependencies
        "fastapi>=0.115.9",
        "uvicorn[standard]",
        "python-dotenv",
        "pydantic>=2.0.0",
        "python-multipart",
        
        # LangChain and AI/ML
        "langchain",
        "langchain-chroma",
        "langchain-core",
        "langchain-ollama",
        "chromadb",
        
        # Web scraping
        "beautifulsoup4",
        "requests",
        "httpx",
        
        # Utilities
        "python-jose[cryptography]",
        "passlib[bcrypt]",
        "python-dateutil",
        "PyYAML",
    ],
    python_requires=">=3.10",
    author="Your Name",
    author_email="your.email@example.com",
    description="Backend service for GeNotes Genomic Guidelines Assistant",
    url="https://github.com/yourusername/genotes",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
