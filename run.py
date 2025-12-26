#!/usr/bin/env python3
"""
Production Entry Point - Agentic RAG System

Requirements:
- OpenAI API Key (OPENAI_API_KEY)
- Milvus Cloud URI (MILVUS_URI)
- Milvus Cloud Token (MILVUS_TOKEN)
"""

import os
import sys
import logging
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_environment() -> None:
    """Load environment variables from .env file."""
    from dotenv import load_dotenv
    
    # Check for .env files in multiple locations
    env_locations = [
        PROJECT_ROOT / "config" / ".env",
        PROJECT_ROOT / ".env",
    ]
    
    for env_path in env_locations:
        if env_path.exists():
            load_dotenv(env_path)
            logging.info(f"Loaded environment from: {env_path}")
            break


def setup_logging() -> None:
    """Configure logging for the application."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    if debug:
        log_level = "DEBUG"
    
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def validate_environment() -> bool:
    """Validate required environment variables."""
    required_vars = ["OPENAI_API_KEY"]
    milvus_vars = ["MILVUS_URI", "ZILLIZ_URI"]  # Either one is required
    
    missing = []
    
    # Check required vars
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    # Check Milvus vars (one of them is required)
    has_milvus = any(os.getenv(var) for var in milvus_vars)
    if not has_milvus:
        missing.append("MILVUS_URI or ZILLIZ_URI")
    
    if missing:
        for var in missing:
            logging.error(f"Missing required environment variable: {var}")
        return False
    
    logging.info("✓ Environment validated successfully")
    return True


def ensure_crewai_settings() -> None:
    """Ensure CrewAI settings file exists."""
    if not os.environ.get("CREWAI_SETTINGS_PATH"):
        settings_path = PROJECT_ROOT / "config" / "crew_settings.json"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        if not settings_path.exists():
            settings_path.write_text("{}\n", encoding="utf-8")
        os.environ["CREWAI_SETTINGS_PATH"] = str(settings_path)


def main() -> None:
    """Main entry point."""
    # Load environment first
    load_environment()
    
    # Setup logging
    setup_logging()
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Starting Agentic RAG System (Production)")
    logger.info("=" * 60)
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed. Please check your .env file.")
        sys.exit(1)
    
    # Ensure CrewAI settings
    ensure_crewai_settings()
    
    # Get server configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    reload = os.getenv("DEBUG", "false").lower() == "true"
    
    logger.info(f"Configuration:")
    logger.info(f"  Host: {host}")
    logger.info(f"  Port: {port}")
    logger.info(f"  Workers: {workers}")
    logger.info(f"  Reload: {reload}")
    logger.info(f"  LLM: OpenAI GPT-4o-mini")
    logger.info(f"  Vector Store: Milvus Cloud (HNSW)")
    logger.info("=" * 60)
    
    # Run the server
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
