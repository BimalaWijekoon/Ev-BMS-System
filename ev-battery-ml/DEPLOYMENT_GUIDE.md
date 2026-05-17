# 🚀 Deployment Guide — HuggingFace Spaces

## Your Deployment Details

| Item | Value |
|------|-------|
| **Platform** | Hugging Face Spaces |
| **Project** | battery-intelligence |
| **HF Username** | bmwmiuranda |
| **Space URL** | https://huggingface.co/spaces/bmwmiuranda/battery-intelligence |
| **Live Demo URL** | https://bmwmiuranda-battery-intelligence.hf.space |
| **GitHub Repo** | https://github.com/BimalaWijekoon/Ev-BMS-System |

---

## ✅ Pre-Deployment Checklist

- ✅ GitHub repo updated with latest code
- ✅ requirements.txt includes Streamlit & Plotly
- ✅ README.md has HF Spaces metadata header
- ✅ app/app.py is the entry point
- ✅ All model files in `/models/` directory
- ✅ CSV data in `/data/` directory
- ✅ No environment secrets needed (.env.example created)

---

## 🎯 Step 1: Create HuggingFace Space

### 1.1 Go to HuggingFace Spaces
- Open: https://huggingface.co/spaces
- Sign in with your account (bmwmiuranda)

### 1.2 Create New Space
- Click **"Create new Space"** button
- Fill in:
  - **Space name**: `battery-intelligence`
  - **License**: Apache 2.0 (or your choice)
  - **Visibility**: Public
  - **SDK**: **Streamlit** (very important!)
  - **SDK version**: 1.32.0 (optional, auto-selected)

### 1.3 Create Space
- Click **"Create Space"**
- HF creates empty space with a README template

### 1.4 You'll see:
```
✅ Space created!
Repository: https://huggingface.co/spaces/bmwmiuranda/battery-intelligence
Git URL: https://huggingface.co/spaces/bmwmiuranda/battery-intelligence.git
```

---

## 🔗 Step 2: Push Your Code to HF Space

### 2.1 Add HF Space as Remote

```bash
cd g:\Ev-BMS-System\ev-battery-ml

git remote add space https://huggingface.co/spaces/bmwmiuranda/battery-intelligence.git
```

### 2.2 Push Main Branch to Space

```bash
git push space main:main
```

**Expected output**:
```
Enumerating objects: ...
Writing objects: 100% (...)
remote: Updating build job for bmwmiuranda/battery-intelligence
```

### 2.3 Wait for Build

- Go to https://huggingface.co/spaces/bmwmiuranda/battery-intelligence
- You'll see a "Building..." status
- **First build takes 2-5 minutes** (installing dependencies)
- Watch the build logs in the interface

### 2.4 Build Completes ✅
Once build completes:
- Status changes to "Running" (green)
- App URL shows at top: `https://bmwmiuranda-battery-intelligence.hf.space`
- Space is **LIVE and PUBLIC**!

---

## 🧪 Step 3: Test the Deployment

### 3.1 Open the Live App
- Click the app URL or visit: https://bmwmiuranda-battery-intelligence.hf.space

### 3.2 Test Core Features
- **Tab 1 (⚡ Single Prediction)**:
  - Enter battery parameters
  - Click "Run Prediction"
  - Verify IR prediction shows + health insights appear
  
- **Tab 2 (📊 Batch Analysis)**:
  - Try uploading a CSV

- **Tab 3 (📖 Model Info)**:
  - Verify model details display

- **Tab 4 (🏥 Model Health)**:
  - Verify all components show green (healthy)

- **Tab 5 (🔋 Charging Simulation)**:
  - Click "Initialize Simulation"
  - Click "Step" to advance SOC
  - Verify predictions update

### 3.3 Check Browser Console
- Press F12 (Developer Tools)
- Go to **Console** tab
- Confirm **no red errors**

### 3.4 Test on Mobile
- Open app on phone/mobile browser
- Verify layout responsive

---

## 📊 Step 4: Verify Models & Data Load

### 4.1 Check Production Logs
- Go to: https://huggingface.co/spaces/bmwmiuranda/battery-intelligence
- Click **Settings** > **Logs**
- Scroll through logs looking for:
  ```
  ✅ [OK] XGBoost models loaded
  ✅ [OK] Preprocessor pipeline loaded
  ✅ [OK] Fleet data loaded
  ```

### 4.2 If Error: "Model Not Found"
- Problem: Model files not pushed to HF Space
- Solution:
  ```bash
  # Check if models dir exists locally
  ls models/
  
  # If files exist, force push with LFS:
  git lfs install
  git add models/*.pkl
  git commit -m "fix: add model files"
  git push space main:main
  ```

---

## 🎨 Step 5: Add Demo & Links to GitHub README

### 5.1 Update GitHub README
Edit: `README.md` on GitHub

Add at the very top (after front matter):

```markdown
## 🎯 Live Demo

[![Launch in HF Spaces](https://img.shields.io/badge/🤗-Launch%20in%20Spaces-blue)](https://huggingface.co/spaces/bmwmiuranda/battery-intelligence)
[![GitHub](https://img.shields.io/badge/GitHub-Open%20Repo-black)](https://github.com/BimalaWijekoon/Ev-BMS-System)

**[👉 Open Live App](https://bmwmiuranda-battery-intelligence.hf.space)**

Try the battery intelligence system instantly — no installation needed!
```

### 5.2 Add Screenshots Section

Add near bottom of README:

```markdown
## 📸 Screenshots

### Single Prediction Tab
[Add screenshot showing Tab 1 with health insights]

### Charging Simulator Tab
[Add screenshot showing Tab 5 with SOC gauge and live predictions]

### Fleet Analytics
[Add screenshot showing fleet comparison metrics]
```

To capture screenshots:
1. Open the live app
2. Right-click > "Take a screenshot" (or use Snipping Tool / ShareX)
3. Save as PNG in `/plots/` folder
4. Reference in README

### 5.3 Commit & Push to GitHub

```bash
cd g:\Ev-BMS-System\ev-battery-ml

git add README.md
git commit -m "docs: add HF Spaces deployment link and demo badges"
git push origin main
```

---

## 🔍 Step 6: Monitor Deployment

### 6.1 Set Up Uptime Monitoring (Free)

Go to: https://uptimerobot.com

- Sign up (free tier)
- Click **Add New Monitor**
- Set:
  - **Monitoring Type**: HTTP(s)
  - **URL**: https://bmwmiuranda-battery-intelligence.hf.space
  - **Monitoring Interval**: 5 minutes
  - **Alert Contacts**: Your email
- Click **Create Monitor**

You'll get email alerts if the Space goes down (rare).

### 6.2 Check Space Status Regularly

- Visit: https://huggingface.co/spaces/bmwmiuranda/battery-intelligence
- Check status badge (green = running, red = error)
- Click **Settings** > **Logs** to see recent activity

### 6.3 Known Behavior: Cold Start Delay

**First access after inactivity**: Space may take 30-60 seconds to start
- This is normal on free tier
- Add note in README: "First load takes ~30 seconds"

---

## 📋 Step 7: Update GitHub Topics & Metadata

### 7.1 Go to GitHub Repo Main Page
- https://github.com/BimalaWijekoon/Ev-BMS-System

### 7.2 Click "⚙️ About" (gear icon on right side)

Add:
- **Description**: "AI/ML battery health prediction system with real-time insights and RUL estimation"
- **Website**: https://bmwmiuranda-battery-intelligence.hf.space
- **Topics**: `ai` `machine-learning` `battery-management` `xgboost` `streamlit` `huggingface`

### 7.3 Pin This Repo
- On your GitHub profile
- Shows as highlight to visitors

---

## 🎯 Step 8: Share Your Project

### Share Links:
1. **HF Spaces**: https://huggingface.co/spaces/bmwmiuranda/battery-intelligence
2. **GitHub**: https://github.com/BimalaWijekoon/Ev-BMS-System
3. **Live Demo**: https://bmwmiuranda-battery-intelligence.hf.space

### On Social Media:
```
🔋 Just deployed my EV Battery Intelligence System to production! 

✨ Real-time battery health scoring using XGBoost ML models
✨ 0-100% SOC interactive simulator with live predictions
✨ Fleet analytics & remaining useful life (RUL) estimation

Try it live: [HF Spaces URL]

Built with: Python, Streamlit, XGBoost, scikit-learn
#AI #MachineLearning #Battery #EV #HuggingFace
```

---

## ✅ Final Deployment Checklist

- [ ] HF Space created: https://huggingface.co/spaces/bmwmiuranda/battery-intelligence
- [ ] Code pushed to HF Space (`git push space main:main`)
- [ ] Build completed successfully
- [ ] Live app accessible and all 5 tabs work
- [ ] Model health checks pass
- [ ] No errors in browser console
- [ ] README updated with HF badge & demo links
- [ ] GitHub repo description updated
- [ ] Uptime monitoring configured
- [ ] Topics added to GitHub
- [ ] Project shared (optional)

---

## 🆘 Troubleshooting

### Problem: "Module not found" error in logs

**Fix**: Missing package in requirements.txt

```bash
pip freeze > requirements.txt
git add requirements.txt
git commit -m "fix: update requirements"
git push space main:main
```

---

### Problem: Models not loading

**Fix**: Model files not pushed properly

```bash
# Reinstall Git LFS
git lfs install

# Add model files
git lfs track "models/*.pkl"
git add .gitattributes models/
git commit -m "fix: add models with LFS"
git push space main:main
```

---

### Problem: App shows "Logs Loading..." forever

**Fix**: Build stuck or Docker issue

- Recommended: Delete Space and create new one
- Go to Space Settings > Delete Space
- Recreate with same name

---

### Problem: CORS or API errors

**Not applicable** — project is self-contained, no external APIs

---

## 🎓 What You Deployed

**System Architecture** (now running on HF Spaces):

```
User Opens App (HF Spaces)
         ↓
    Streamlit UI
    (5 interactive tabs)
         ↓
    ML Inference Engine
    (XGBoost models in memory)
         ↓
    Battery Insights
    (health scoring, RUL)
         ↓
    Results Displayed
    (gauges, charts, text)
```

**Resources Used**:
- **CPU**: HF free tier (sufficient for batch inference)
- **RAM**: ~1-2GB (models + data in memory)
- **Storage**: CSV + pickle files (~20MB)
- **Cost**: **$0** (free tier)

---

## 📞 Next Steps

1. **Monitor the app** for 1-2 weeks to confirm stability
2. **Collect feedback** from users
3. **Add features** based on usage patterns
4. **Consider upgrading** to paid GPU tier if needed for faster inference

---

## 🚀 Congratulations!

Your EV Battery Intelligence System is now **LIVE IN PRODUCTION** on HuggingFace Spaces! 🎉

**Live URL**: https://bmwmiuranda-battery-intelligence.hf.space

---

**Deployment Date**: May 17, 2026  
**Status**: ✅ Production Ready  
**Support**: Check logs at HF Spaces or review PROJECT.md for technical details
