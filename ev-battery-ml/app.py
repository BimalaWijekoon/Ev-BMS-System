"""
Root entry point for HuggingFace Spaces
This file runs the Streamlit app from the app/ directory
"""
import subprocess
import sys

if __name__ == "__main__":
    # Run the actual Streamlit app
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "app/app.py",
        "--server.port=7860",
        "--server.address=0.0.0.0"
    ])
