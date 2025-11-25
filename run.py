import os
import sys
from pathlib import Path

# Add the DAY10 directory to Python path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """Main entry point."""
    import uvicorn
    from dotenv import load_dotenv
    
    # Load environment variables
    env_file = Path(__file__).parent / "config" / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        # Try loading from current directory
        load_dotenv()
    
    # Validate required environment variables
    if not os.getenv("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY not set. Set it in .env or as environment variable.")
    
    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("APP_ENV", "development") == "development"
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    print(f"Starting Multi-Agent RAG API on {host}:{port}")
    print(f"Environment: {os.getenv('APP_ENV', 'development')}")
    print(f"Debug: {os.getenv('DEBUG', 'false')}")
    print(f"Docs available at: http://{host}:{port}/docs")
    
    # Run the server
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        access_log=True
    )


if __name__ == "__main__":
    main()

