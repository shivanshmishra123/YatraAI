from setuptools import setup, find_packages

setup(
    name="yatra-ai",
    version="0.1.0",
    description="Intelligent Travel Planning Assistant for the Indian Travel Ecosystem",
    author="Yatra AI Team",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "langchain>=0.3.0",
        "langgraph>=0.2.0",
        "streamlit>=1.38.0",
        "chromadb>=0.5.0",
        "openai>=1.40.0",
        "python-dotenv>=1.0.0",
        "httpx>=0.27.0",
        "pydantic>=2.0.0",
        "python-dateutil>=2.9.0",
    ],
)
