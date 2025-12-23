# Enhanced RAG System v7.0 - Setup Script
# =========================================
# Run this script to set up your environment

import os
import shutil

def setup_environment():
    """Setup the RAG system environment"""
    
    print("=" * 70)
    print("Enhanced RAG System v7.0 - Setup")
    print("=" * 70)
    
    # Step 1: Create .env file from template
    if not os.path.exists('.env'):
        if os.path.exists('.env.example'):
            print("\n📝 Creating .env file from template...")
            shutil.copy('.env.example', '.env')
            print("✅ Created .env file")
            print("⚠️  Please edit .env and add your API keys!")
        else:
            print("❌ .env.example not found!")
    else:
        print("\n✅ .env file already exists")
    
    # Step 2: Check for required dependencies
    print("\n📦 Checking dependencies...")
    
    try:
        import PyMuPDF
        print("✅ PyMuPDF installed")
    except ImportError:
        print("❌ PyMuPDF not installed - run: pip install PyMuPDF")
    
    try:
        import faiss
        print("✅ FAISS installed")
    except ImportError:
        print("❌ FAISS not installed - run: pip install faiss-cpu")
    
    try:
        import sentence_transformers
        print("✅ Sentence Transformers installed")
    except ImportError:
        print("❌ Sentence Transformers not installed - run: pip install sentence-transformers")
    
    try:
        import pydantic
        print("✅ Pydantic installed")
    except ImportError:
        print("❌ Pydantic not installed - run: pip install pydantic")
    
    try:
        import pyperclip
        print("✅ Pyperclip installed (clipboard support enabled)")
    except ImportError:
        print("⚠️  Pyperclip not installed - clipboard features disabled")
        print("   Install with: pip install pyperclip")
    
    try:
        from dotenv import load_dotenv
        print("✅ python-dotenv installed")
    except ImportError:
        print("⚠️  python-dotenv not installed - install for easier .env loading")
        print("   Install with: pip install python-dotenv")
    
    # Step 3: Check for Ollama
    print("\n🤖 Checking Ollama (local AI)...")
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("✅ Ollama is running")
            models = response.json().get("models", [])
            if models:
                print(f"   Available models: {[m['name'] for m in models[:3]]}")
        else:
            print("⚠️  Ollama not responding correctly")
    except:
        print("❌ Ollama not running")
        print("   Install: https://ollama.ai/")
        print("   Then run: ollama pull llama3.2:3b")
    
    # Step 4: Create directories
    print("\n📁 Creating data directories...")
    os.makedirs(".enhanced_rag_projects", exist_ok=True)
    os.makedirs(".rag_cache", exist_ok=True)
    print("✅ Directories created")
    
    # Step 5: Final instructions
    print("\n" + "=" * 70)
    print("Setup Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Edit .env and add your API keys (optional but recommended)")
    print("2. Run: python v7_complete.py")
    print("3. Create a project and add PDFs")
    print("4. Start asking questions!")
    print("\nFor help, see: README_v7.md or V7_FINAL_SUMMARY.md")
    print("=" * 70)

if __name__ == "__main__":
    setup_environment()
