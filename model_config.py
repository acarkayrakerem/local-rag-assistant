import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

class ModelConfig:
    def __init__(self, provider, api_key=None, model_name=None):
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name

def get_llm(config: ModelConfig):
    if(config.provider == "openai"):
        return create_openai_llm(config)
    elif(config.provider == "google"):
        return create_google_llm(config)
    elif(config.provider == "anthropic"):
        return create_anthropic_llm(config)
    elif(config.provider == "ollama(free)"):
        return create_ollama_llm(config)
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")

def create_openai_llm(config: ModelConfig):
    return ChatOpenAI(
        api_key=config.api_key,
        model=config.model_name or "gpt-4.1-mini",
        temperature=0.0,
    )
    
def create_google_llm(config: ModelConfig):
    os.environ["GOOGLE_API_KEY"] = config.api_key
    return ChatGoogleGenerativeAI(
        model=config.model_name or "gemini-2.5-flash",
        temperature=0.0,  
    )


def create_anthropic_llm(config: ModelConfig):
    os.environ["ANTHROPIC_API_KEY"] = config.api_key
    return ChatAnthropic(
        model=config.model_name or "claude-haiku-4-5-20251001",
        temperature=0.0,
    )

def create_ollama_llm(config: ModelConfig):
    return ChatOllama(
        model=config.model_name or "llama3.2:latest",
        temperature=0.0,
    )