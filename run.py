"""
============================================================
MedAI — Server Runner
============================================================
Starts the FastAPI backend with uvicorn.

Usage:
    python run.py
    python run.py --port 8080
    python run.py --reload   (development mode)
============================================================
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def check_setup():
    """Verify setup has been run before starting server."""
    required = [
        "backend/ml/saved_models/best_model.pkl",
        "backend/rag/faiss_index/medical.index",
        "backend/data/disease_info.json",
    ]
    missing = [f for f in required if not Path(f).exists()]
    if missing:
        print("\n❌ Setup not complete. Missing files:")
        for f in missing:
            print(f"   • {f}")
        print("\nPlease run setup first:")
        print("   python setup.py\n")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="MedAI Server")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 8000)))
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    parser.add_argument("--skip-check", action="store_true", help="Skip setup verification")
    args = parser.parse_args()

    if not args.skip_check:
        check_setup()

    import uvicorn

    print(f"""
╔══════════════════════════════════════════════════════╗
║   MedAI — Clinical Decision Support System           ║
║   Starting server...                                 ║
╠══════════════════════════════════════════════════════╣
║   URL:      http://{args.host}:{args.port:<35}║
║   API Docs: http://{args.host}:{args.port}/api/docs        ║
║   Mode:     {'Development (reload ON)' if args.reload else 'Production':<35}║
╚══════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "backend.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()