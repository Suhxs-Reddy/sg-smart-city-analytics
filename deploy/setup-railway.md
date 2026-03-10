# Deploy to Railway.app (15 Minutes) — Zero Credit Card

**Easiest deployment option** - No credit card, deploys from GitHub

---

## What You'll Get

```
✅ Backend deployed in 5 minutes
✅ Public URL (https://your-app.up.railway.app)
✅ Auto-deploys on git push
✅ 500 hours/month free ($5 credit)
✅ No credit card required
✅ SSL/HTTPS included
```

**After this**: Update Vercel dashboard → Full system live

---

## Step 1: Create Railway Account (2 minutes)

### 1.1 Go to Railway
https://railway.app

### 1.2 Sign Up
1. Click **"Login"** (top right)
2. Click **"Login with GitHub"**
3. Authorize Railway to access your repos
4. **Done!** No email verification needed

---

## Step 2: Create New Project (3 minutes)

### 2.1 New Project
1. Click **"New Project"**
2. Select **"Deploy from GitHub repo"**
3. Find: **`Suhxs-Reddy/sg-smart-city-analytics`**
4. Click on it

### 2.2 Configure Deployment

Railway will auto-detect your setup, but we need to configure it:

1. Click **"Add variables"** (or Settings → Variables)

2. Add these environment variables:

```
PORT=8000
PYTHONUNBUFFERED=1
```

### 2.3 Set Root Directory (IMPORTANT)

Railway might try to deploy the whole repo. We only want the API.

**Option A - If Railway has "Root Directory" setting:**
- Settings → Root Directory → Set to: `.` (current directory is fine)

**Option B - Create railway.json:**
We'll do this next to configure properly.

---

## Step 3: Configure Railway Settings (5 minutes)

### 3.1 Create Railway Config

We need to tell Railway how to run our FastAPI app.

**I'll create this file for you in a moment** - it tells Railway:
- Use Python 3.11
- Install dependencies
- Run FastAPI on port 8000

### 3.2 Wait for Deployment

After config is added:
1. Railway will automatically deploy
2. Watch the build logs (click "Deployments" tab)
3. Should take 2-3 minutes
4. Status changes to "Active" ✅

### 3.3 Get Your URL

1. Click "Settings" tab
2. Scroll to **"Domains"** section
3. Click **"Generate Domain"**
4. Copy the URL: `https://sg-smart-city-analytics-production.up.railway.app`

**Save this URL!** You'll need it for Vercel.

---

## Step 4: Test Your API (2 minutes)

### 4.1 Open Health Endpoint

In your browser, open:
```
https://YOUR-APP.up.railway.app/api/health
```

Should see:
```json
{
  "status": "healthy",
  "cameras": 0,
  "detections": 0,
  "uptime": "0:00:15"
}
```

### 4.2 Check API Docs

Open:
```
https://YOUR-APP.up.railway.app/docs
```

Should see: **FastAPI Swagger UI** ✅

---

## Step 5: Connect to Vercel Dashboard (3 minutes)

### 5.1 Update Vercel Environment Variable

1. Go to https://vercel.com
2. Select project: `sg-smart-city-analytics`
3. **Settings** → **Environment Variables**
4. Add or edit:
   - Key: `NEXT_PUBLIC_API_URL`
   - Value: `https://YOUR-APP.up.railway.app` (your Railway URL)
5. Click **"Save"**

### 5.2 Redeploy Dashboard

1. **Deployments** tab
2. Latest deployment → Click "..." → **"Redeploy"**
3. Wait 2-3 minutes

### 5.3 Verify

Open: https://sg-smart-city-analytics.vercel.app

**Should see**:
- ✅ Map loads (might still show demo cameras without trained model)
- ✅ No more "localhost:8000" errors in console
- ✅ API connected (check browser console - should fetch from Railway)

---

## Troubleshooting

### Issue: Railway deployment failed

**Check build logs**:
1. Deployments tab
2. Click failed deployment
3. View logs

**Common issues**:
- Missing `Procfile` or `railway.json`
- Wrong start command
- Missing dependencies

**Fix**: I'll create the config files next

### Issue: "Application failed to respond"

**Cause**: App isn't listening on correct port

**Fix**: Ensure environment variable `PORT=8000` is set

### Issue: 404 on /api/health

**Cause**: Wrong root directory or app not starting

**Fix**: Check logs, ensure `src/api/server.py` exists and runs

---

## Cost & Limits

### Free Tier
- **500 hours/month** execution time
- **$5 starter credit**
- **100 GB network egress**
- **1 GB RAM** per service

**What this means**:
- Enough for 20 days continuous running
- Resets monthly
- More than enough for demo/portfolio

### After Free Tier
- **$5/month** for 500 more hours
- Pay only for what you use
- Can pause services when not needed

---

## Railway vs AWS

| Feature | Railway | AWS EC2 |
|---------|---------|---------|
| **Setup time** | 5 minutes | 30 minutes |
| **Credit card** | Not required | Required |
| **Auto-deploy** | Yes (git push) | No (manual) |
| **SSL/HTTPS** | Included | Manual setup |
| **Cost (free tier)** | 500h/month | 12 months |
| **Ease of use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

**Railway is better for now** - deploy fast, no hassle

---

## Next Steps

**After Railway deployment**:
1. ✅ Backend running at Railway URL
2. ✅ Dashboard connected to Railway
3. ✅ Full system live
4. ⏳ Train model on Kaggle (when data ready)
5. ⏳ Deploy trained model

**To deploy trained model later**:
```bash
# From your laptop
# (Railway doesn't support direct file upload, so use GitHub)

# Add model to repo (if small enough)
git lfs track "*.pt"
git add models/yolo11s_traffic.pt
git commit -m "feat: add trained YOLO model"
git push origin main

# Railway auto-deploys → Model available
```

Or use Railway's volume storage (more advanced).

---

## Monitoring

### Check Deployment Status
1. Railway dashboard
2. Your project
3. **Deployments** tab → See all deployments
4. **Observability** tab → See logs, metrics

### View Logs
1. Click your service
2. **"View Logs"** button
3. Live tail of application logs

### Check Usage
1. Account settings → Usage
2. See hours used / remaining
3. Network usage

---

## Summary

**Time**: 15 minutes total
**Cost**: $0 (500 hours free)
**Result**: Backend live with HTTPS

**What you'll have**:
- ✅ Live dashboard: https://sg-smart-city-analytics.vercel.app
- ✅ Live API: https://YOUR-APP.up.railway.app
- ✅ GitHub repo: Production code
- ✅ Auto-deploy: Push to GitHub → Railway redeploys
- ✅ SSL/HTTPS: Professional URLs
- ✅ Zero credit card: Completely free to start

**Ready to deploy?** → Go to https://railway.app and let's do this! 🚀
