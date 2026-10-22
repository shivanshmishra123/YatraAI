"""Unified LLM Provider Factory for Yatra AI.

Supports high-speed free tier providers (Groq Cloud with Llama 3.3 70B),
OpenRouter (free models), and OpenAI as a fallback.
"""

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("yatra_ai.llm")


def get_primary_llm(
    temperature: float = 0.0,
    model_name: Optional[str] = None,
    json_mode: bool = False,
) -> Optional[Any]:
    """Instantiate the primary fast LLM for NLU extraction, agent planning, and routing.
    
    Priority order:
    1. Groq Cloud (Recommended: 100% Free Tier, Llama 3.3 70B / 3.1 8B, 30 RPM, zero cost)
    2. OpenRouter (Free tier models like meta-llama/llama-3.3-70b-instruct:free)
    3. OpenAI (Legacy support if OPENAI_API_KEY is provided)
    """
    # 1. Groq Cloud (Free Tier Fast Inference)
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key and not groq_key.startswith("gsk_your") and len(groq_key) > 15:
        try:
            from langchain_groq import ChatGroq
            target_model = model_name or os.getenv("PRIMARY_MODEL", "openai/gpt-oss-120b")
            model_kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
            llm = ChatGroq(
                model=target_model,
                temperature=temperature,
                api_key=groq_key,
                model_kwargs=model_kwargs,
            )
            logger.info(f"Initialized Groq Cloud LLM ({target_model})")
            return llm
        except Exception as e:
            logger.warning(f"Could not initialize Groq LLM: {e}")

    # 2. Google Gemini (Fast & High Intelligence)
    google_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if google_key and not google_key.startswith("AIza_your") and len(google_key) > 20:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            target_model = model_name or os.getenv("GEMINI_PRIMARY_MODEL", "gemini-2.5-flash")
            llm = ChatGoogleGenerativeAI(
                model=target_model,
                temperature=temperature,
                google_api_key=google_key,
            )
            logger.info(f"Initialized Google Gemini LLM ({target_model})")
            return llm
        except Exception as e:
            logger.warning(f"Could not initialize Google Gemini LLM: {e}")

    # 3. OpenRouter (Alternative provider)
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if openrouter_key and not openrouter_key.startswith("sk-or-your") and len(openrouter_key) > 15:
        try:
            from langchain_openai import ChatOpenAI
            target_model = model_name or os.getenv("PRIMARY_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
            model_kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
            llm = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                model=target_model,
                temperature=temperature,
                api_key=openrouter_key,
                model_kwargs=model_kwargs,
            )
            logger.info(f"Initialized OpenRouter LLM ({target_model})")
            return llm
        except Exception as e:
            logger.warning(f"Could not initialize OpenRouter LLM: {e}")

    # 4. OpenAI
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key and not openai_key.startswith("sk-your") and len(openai_key) > 20:
        try:
            from langchain_openai import ChatOpenAI
            target_model = model_name or "gpt-4o-mini"
            model_kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
            llm = ChatOpenAI(
                model=target_model,
                temperature=temperature,
                api_key=openai_key,
                model_kwargs=model_kwargs,
            )
            logger.info(f"Initialized OpenAI LLM ({target_model})")
            return llm
        except Exception as e:
            logger.warning(f"Could not initialize OpenAI LLM: {e}")

    return None


def get_active_llm_info() -> Dict[str, str]:
    """Return name and details of currently active primary LLM provider."""
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key and not groq_key.startswith("gsk_your") and len(groq_key) > 15:
        return {
            "provider": "Groq Cloud",
            "model": os.getenv("PRIMARY_MODEL", "openai/gpt-oss-120b"),
            "status": "ONLINE (Free Tier)",
            "badge": "Groq Cloud",
        }

    google_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if google_key and not google_key.startswith("AIza_your") and len(google_key) > 20:
        return {
            "provider": "Google Gemini",
            "model": "gemini-2.5-flash",
            "status": "ONLINE",
            "badge": "Gemini 2.5 Flash",
        }

    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if openrouter_key and not openrouter_key.startswith("sk-or-your") and len(openrouter_key) > 15:
        return {
            "provider": "OpenRouter",
            "model": os.getenv("PRIMARY_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
            "status": "ONLINE (Free Model)",
            "badge": "OpenRouter Free",
        }

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key and not openai_key.startswith("sk-your") and len(openai_key) > 20:
        return {
            "provider": "OpenAI",
            "model": "gpt-4o-mini",
            "status": "ONLINE",
            "badge": "OpenAI GPT-4o-mini",
        }

    return {
        "provider": "None",
        "model": "None",
        "status": "OFFLINE_NO_API_KEY",
        "badge": "No API Key",
    }
