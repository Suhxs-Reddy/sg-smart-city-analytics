# Deployment Strategy — Cloud-Agnostic Production

**Senior ML Engineer Approach**: Build once, deploy anywhere

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    User's Browser                        │
└────────────────────────┬────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────┐
│              Vercel (Next.js Dashboard)                  │
│              https://sg-smart-city.vercel.app            │
│              • FREE hosting                              │
│              • Global CDN                                │
│              • Auto HTTPS                                │
└────────────────────────┬────────────────────────────────┘
                         │
                         ↓ HTTP API calls
┌─────────────────────────────────────────────────────────┐
│              Cloud VM (FastAPI Backend)                  │
│              http://YOUR_IP:8000                         │
│              • Docker + Docker Compose                   │
│              • YOLOv11s inference                        │
│              • RESTful API                               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ↓ Fetches images
┌─────────────────────────────────────────────────────────┐
│              Singapore LTA Traffic Cameras               │
│              https://datamall.lta.gov.sg                 │
│              • 90 cameras                                │
│              • Updated every 60s                         │
└─────────────────────────────────────────────────────────┘
```

---

## Cloud Provider Comparison

| Feature | AWS EC2 | Azure VM | Google Compute | Oracle Cloud |
|---------|---------|----------|----------------|--------------|
| **Free Tier** | 12 months | ❌ (Student only) | $300 credit | Forever free |
| **Instance** | t2.micro (1GB) | B1s (1GB) | e2-micro (1GB) | VM.Standard.E2.1.Micro |
| **Monthly Cost (after free)** | $8-12 | $8 | $7 | $0 |
| **Setup Difficulty** | Easy | Easy | Easy | Medium |
| **Current Status** | ✅ Ready | ❌ Blocked by ASU | Not tried | Not tried |

**Recommendation**: AWS (already working, 12 months free)

---

## Deployment Options

### Option 1: AWS EC2 (Recommended) ⭐

**Pros**:
- ✅ 12 months free tier
- ✅ t2.micro sufficient for inference
- ✅ Simple setup
- ✅ CloudWatch monitoring included
- ✅ Works with your account

**Cons**:
- Costs $8-12/month after year 1

**Guide**: `deploy/setup-aws-ec2.md`

**Time**: 30 minutes

---

### Option 2: Azure VM (Blocked)

**Status**: ❌ ASU Azure for Students has region restrictions

**Error**: `RequestDisallowedByAzure` in ALL regions

**What we tried**:
- ❌ southeastasia
- ❌ eastus
- ❌ centralus

**To fix**: Contact Azure support (1-3 days)

**Not recommended**: Wastes momentum

---

### Option 3: Google Cloud Platform

**Free Tier**:
- $300 credit (90 days)
- Always Free: e2-micro (0.25 vCPU, 1GB RAM)

**Pros**:
- $300 credit is generous
- e2-micro always free (even after 90 days)
- Good documentation

**Cons**:
- Requires credit card verification
- More complex IAM/networking

**If AWS doesn't work**: Try this next

---

### Option 4: Oracle Cloud (Always Free)

**Free Tier**:
- Forever free (no time limit)
- 2x VM.Standard.E2.1.Micro (1GB RAM each)
- 200 GB storage

**Pros**:
- ✅ Actually free forever
- ✅ 2 VMs (can run multiple services)
- ✅ No credit card charged

**Cons**:
- Setup more complex
- Less documentation
- UI not as polished

**If you want $0 long-term**: This is the option

---

### Option 5: Local Docker (Last Resort)

**If all cloud options fail**:
```bash
cd /Users/suhasreddy/sg-smart-city-analytics
docker compose up -d
```

**Pros**:
- Works immediately
- Full control

**Cons**:
- Runs on your laptop (you said no)
- Not accessible from internet
- Dashboard would need localhost connection

**Not recommended** per your requirements

---

## Current Deployment Plan

### Phase 1: Frontend (NOW) — 10 minutes
- ✅ Deploy dashboard to Vercel
- ✅ Works in demo mode (90 simulated cameras)
- ✅ Live URL to share with recruiters
- ✅ $0 cost

**Guide**: `VERCEL_DEPLOY.md`

### Phase 2: Backend (TODAY) — 30 minutes
- ✅ Deploy FastAPI to AWS EC2
- ✅ t2.micro (1GB RAM, 1 vCPU)
- ✅ Docker + Docker Compose
- ✅ Free for 12 months

**Guide**: `deploy/setup-aws-ec2.md`

### Phase 3: Model Training (TOMORROW) — 2-3 hours
- ✅ Upload `notebooks/train_yolo.ipynb` to Kaggle
- ✅ Enable GPU T4 (free 30h/week)
- ✅ Train YOLOv11s on UA-DETRAC
- ✅ Download trained model

**Already ready**: Notebook is production-grade

### Phase 4: Connect Everything (10 minutes)
- ✅ SCP trained model to AWS EC2
- ✅ Restart API with new model
- ✅ Update Vercel env var to point to EC2
- ✅ Redeploy Vercel dashboard
- ✅ Full system live ✅

---

## Disaster Recovery

### If AWS EC2 fails:
1. Try Google Cloud (similar setup)
2. Use Oracle Cloud (always free)
3. Last resort: Local Docker

### If model training fails on Kaggle:
1. Use Google Colab (T4 GPU, free)
2. Download pre-trained YOLOv11s (works but not as good)

### If Vercel deployment fails:
1. Use Netlify (similar free tier)
2. Use Render (free for static sites)
3. GitHub Pages (static only, no API calls)

---

## Cost Optimization

### Year 1 (AWS Free Tier)
```
Vercel Dashboard:  $0
AWS EC2 t2.micro:  $0 (free tier)
EBS Storage 30GB:  $0 (free tier)
Kaggle GPU:        $0 (free 30h/week)
TOTAL:             $0/month
```

### Year 2+ (After Free Tier Expires)
```
Vercel Dashboard:  $0 (still free)
AWS EC2 t2.micro:  $8-12/month
EBS Storage 30GB:  $3/month
Kaggle GPU:        $0 (still free)
TOTAL:             $11-15/month
```

### Alternative (Always Free)
```
Vercel Dashboard:  $0
Oracle Cloud VM:   $0 (forever free)
Kaggle GPU:        $0
TOTAL:             $0/month forever
```

---

## Monitoring & Alerts

### AWS CloudWatch (Free Tier)
- 10 custom metrics
- 10 alarms
- 1 million API requests

**Set up billing alert**:
1. AWS Console → Billing
2. Preferences → "Receive Billing Alerts"
3. CloudWatch → Alarms → "Create Alarm"
4. Metric: "Estimated Charges"
5. Threshold: $5 USD
6. Email notification

### Uptime Monitoring (Free)
- **UptimeRobot**: https://uptimerobot.com
  - 50 monitors free
  - 5-minute checks
  - Email/SMS alerts

**Monitor**:
- Vercel dashboard URL (https://sg-smart-city.vercel.app)
- AWS API health (http://YOUR_IP:8000/api/health)

---

## Security Checklist

### EC2 Security Group
- ✅ Port 22 (SSH): Only from your IP
- ✅ Port 80 (HTTP): Open to 0.0.0.0/0
- ✅ Port 8000 (API): Open to 0.0.0.0/0
- ❌ Port 3306/5432 (Database): NOT OPEN (not used)

### SSH Key Management
- ✅ Use SSH keys (not passwords)
- ✅ Key has proper permissions (`chmod 400`)
- ✅ Keep private key safe (~/.ssh/)
- ❌ Never commit keys to git

### API Security
- ✅ CORS configured (allows Vercel domain)
- ✅ Rate limiting (FastAPI default)
- ⏳ Add authentication (future, when needed)
- ⏳ Move to HTTPS (future, use nginx + Let's Encrypt)

### Environment Variables
- ✅ API keys in .env file
- ✅ .env in .gitignore
- ✅ Vercel env vars in dashboard (not code)
- ❌ Never hardcode secrets

---

## Scaling Strategy (Future)

### When to scale:
- More than 100 requests/second
- More than 90 cameras (if expanding to other cities)
- Model takes >1 second per image

### Options:
1. **Vertical scaling**: Upgrade to t2.small/medium ($20-40/month)
2. **Horizontal scaling**: Add load balancer + multiple EC2 instances
3. **Serverless**: Move to AWS Lambda (pay per request)
4. **GPU inference**: Add g4dn.xlarge instance for faster detection

**Current**: t2.micro is sufficient for 90 cameras @ 1 req/min

---

## Backup Strategy

### What to backup:
1. **Trained models**: `models/yolo11s_traffic.pt` (100-200 MB)
2. **Detection results**: `data/detections/` (grows over time)
3. **Metadata**: Camera configs, thresholds, etc.

### How:
```bash
# Manual backup (weekly)
scp -r -i ~/.ssh/key.pem \
  ubuntu@YOUR_IP:~/sg-smart-city-analytics/models \
  ./backups/$(date +%Y%m%d)/

# Automated backup (cron job on EC2)
# Add to crontab: 0 3 * * 0 /home/ubuntu/backup.sh
```

### Where to store:
- **Option 1**: S3 (cheap, $0.023/GB/month)
- **Option 2**: GitHub LFS (free for small files)
- **Option 3**: Local external drive

---

## Next Steps

**RIGHT NOW** (10 min):
1. Follow `VERCEL_DEPLOY.md`
2. Deploy dashboard
3. Get live URL

**TODAY** (30 min):
1. Follow `deploy/setup-aws-ec2.md`
2. Deploy backend to AWS
3. Connect to Vercel

**TOMORROW** (2-3 hours):
1. Upload notebook to Kaggle
2. Train model
3. Deploy to AWS

**Result**: Full production system in 24 hours, $0 cost

---

**Current priority**: Deploy to AWS EC2 → Follow `deploy/setup-aws-ec2.md`
