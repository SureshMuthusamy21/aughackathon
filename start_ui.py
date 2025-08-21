#!/usr/bin/env python3
"""
Streamlit App Launcher for Medical Data Processing System
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import streamlit
        print("✅ Streamlit is installed")
        return True
    except ImportError:
        print("❌ Streamlit is not installed")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def check_environment():
    """Check if environment is properly configured"""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  Warning: .env file not found")
        print("Please create a .env file with your OPENAI_API_KEY")
        return False
    
    # Load .env file
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        print("Please set your OpenAI API key in the .env file")
        return False
    
    print("✅ Environment configured properly")
    return True

def main():
    """Main launcher function"""
    print("🏥 Medical Data Processing System - Streamlit Launcher")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Launch Streamlit app
    print("🚀 Launching Streamlit application...")
    print("The app will open in your default web browser")
    print("Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Streamlit server stopped")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
