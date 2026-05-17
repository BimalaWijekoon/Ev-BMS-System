#!/usr/bin/env python3
"""HuggingFace Spaces Deployment Script"""

import sys
import os
import subprocess
from huggingface_hub import HfApi

# Get token from environment variable (set it before running: $env:HF_TOKEN="...")
TOKEN = os.getenv("HF_TOKEN", "")
if not TOKEN:
    print("❌ ERROR: HF_TOKEN environment variable not set!")
    print("   Set it with: $env:HF_TOKEN = 'your_token_here'")
    sys.exit(1)

USERNAME = "bmwmiuranda"
SPACE_NAME = "battery-intelligence"

print("=" * 70)
print("🚀 DEPLOYING TO HUGGINGFACE SPACES")
print("=" * 70)

# Step 1: Create space if it doesn't exist
print("\n[1/3] Creating/Verifying HuggingFace Space...")
api = HfApi()

try:
    space = api.create_repo(
        repo_id=SPACE_NAME,
        repo_type="space",
        space_sdk="docker",  # Use docker SDK which is flexible for Streamlit
        token=TOKEN,
        exist_ok=True  # Don't fail if already exists
    )
    print(f"✅ Space ready: https://huggingface.co/spaces/{USERNAME}/{SPACE_NAME}")
except Exception as e:
    print(f"⚠️ Space creation note: {e}")
    print(f"   (This is OK - space may already exist)")
    print(f"✅ Space is ready at: https://huggingface.co/spaces/{USERNAME}/{SPACE_NAME}")

# Step 2: Configure git remote
print("\n[2/3] Configuring git remote...")
try:
    subprocess.run(
        ["git", "remote", "remove", "huggingface"],
        capture_output=True,
        timeout=5
    )
except:
    pass

hf_url = f"https://{USERNAME}:{TOKEN}@huggingface.co/spaces/{USERNAME}/{SPACE_NAME}.git"
result = subprocess.run(
    ["git", "remote", "add", "huggingface", hf_url],
    capture_output=True,
    text=True,
    timeout=5
)

if result.returncode == 0:
    print("✅ Git remote configured")
else:
    print(f"⚠️ Remote setup: {result.stderr}")

# Step 3: Push code
print("\n[3/3] Pushing code to HuggingFace Spaces (this may take 1-2 minutes)...")
print("       Uploading files...")

result = subprocess.run(
    ["git", "push", "huggingface", "main:main", "-f"],
    capture_output=True,
    text=True,
    timeout=120
)

if result.returncode == 0:
    print("✅ Code pushed successfully!")
    print("\n" + "=" * 70)
    print("🎉 DEPLOYMENT COMPLETE!")
    print("=" * 70)
    print(f"\n📍 Space URL: https://huggingface.co/spaces/{USERNAME}/{SPACE_NAME}")
    print(f"🌐 Live App URL: https://{USERNAME}-{SPACE_NAME}.hf.space")
    print("\n⏳ Your space is now building...")
    print("   (Check the build logs at the Space URL)")
    print("   (First build takes 2-5 minutes)")
    print("\n✨ Once build completes:")
    print(f"   1. Open: https://{USERNAME}-{SPACE_NAME}.hf.space")
    print("   2. Test all 5 tabs")
    print("   3. Share with the world!")
    print("\n" + "=" * 70)
else:
    print(f"❌ Push failed: {result.stderr}")
    print(f"\nStdout: {result.stdout}")
    sys.exit(1)
