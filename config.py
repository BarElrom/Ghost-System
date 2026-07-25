"""
GHOST System — Configuration
Centralized settings for service connections and system parameters.

Usage:
    from config import CHROMA_SETTINGS, OLLAMA_SETTINGS
    print(CHROMA_SETTINGS.host)  # "localhost"
    print(OLLAMA_SETTINGS.model) # "llama3.2:3b"

    # Override via environment variables:
    #   GHOST_CHROMA_HOST=192.168.1.10
    #   GHOST_CHROMA_PORT=9000
    #   GHOST_CHROMA_COLLECTION=my_fingerprints
    #   GHOST_OLLAMA_HOST=192.168.1.10
    #   GHOST_OLLAMA_PORT=11434
    #   GHOST_OLLAMA_MODEL=llama3:8b
    #   GHOST_OLLAMA_TIMEOUT=60.0
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ChromaSettings:
    """Connection settings for the ChromaDB vector database.

    Args:
        host: ChromaDB server hostname.
        port: ChromaDB server HTTP port.
        collection_name: Name of the fingerprint collection.
    """

    host: str = os.getenv("GHOST_CHROMA_HOST", "localhost")
    port: int = int(os.getenv("GHOST_CHROMA_PORT", "8000"))
    collection_name: str = os.getenv(
        "GHOST_CHROMA_COLLECTION", "spatial_fingerprints"
    )


# Default instance for easy import.
CHROMA_SETTINGS = ChromaSettings()


@dataclass(frozen=True)
class OllamaSettings:
    """Connection settings for the Ollama LLM server.

    Args:
        host: Ollama server hostname.
        port: Ollama server HTTP port.
        model: Model name/tag to use for inference.
        timeout: Request timeout in seconds.
    """

    host: str = os.getenv("GHOST_OLLAMA_HOST", "localhost")
    port: int = int(os.getenv("GHOST_OLLAMA_PORT", "11434"))
    model: str = os.getenv("GHOST_OLLAMA_MODEL", "llama3.2:3b")
    timeout: float = float(os.getenv("GHOST_OLLAMA_TIMEOUT", "30.0"))

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


OLLAMA_SETTINGS = OllamaSettings()