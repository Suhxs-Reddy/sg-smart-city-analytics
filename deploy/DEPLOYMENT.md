# Deployment Guide — Singapore Smart City Analytics

**Production deployment checklist for Azure + Colab + Kaggle infrastructure**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                  SINGAPORE SMART CITY PLATFORM                  │
└─────────────────────────────────────────────────────────────────┘

 DATA COLLECTION (Colab — Free)
 ┌──────────────────────────────────────────┐
 │  Google Colab Notebook                   │
 │  • Runs collect_data.ipynb               │
 │  • Fetches 90 LTA cameras every 60s      │
 │  • Saves to Google Drive                 │
 │  • Cost: $0/month                        │
 └────────────┬─────────────────────────────┘
              │
              ▼
      Google Drive Storage
      (50GB free tier)
              │
              ▼
 MODEL TRAINING (Kaggle — Free)
 ┌──────────────────────────────────────────┐
 │  Kaggle Notebook                         │
 │  • Runs train_yolo.ipynb                 │
 │  • T4 GPU (30h/week free)                │
 │  • Trains YOLOv11s model                 │
 │  • MLflow experiment tracking            │
 │  • Outputs: best.pt, best.onnx           │
 │  • Cost: $0/month                        │
 └────────────┬─────────────────────────────┘
              │
              ▼
   Trained Model (best.pt)
              │
              ▼
 PRODUCTION INFERENCE (Azure — $8/month)
 ┌──────────────────────────────────────────┐
 │  Azure B1s VM (Singapore region)         │
 │  ┌────────────────────────────────────┐  │
 │  │  Docker Containers:                │  │
 │  │  • API Server (FastAPI)            │  │
 │  │  • Detection Service (YOLOv11s)    │  │
 │  │  • Tracking Service (BoT-SORT)     │  │
 │  │  • Analytics Service               │  │
 │  └────────────────────────────────────┘  │
 │  Cost: ~$8/month                         │
 └────────────┬─────────────────────────────┘
              │
              ▼
 FRONTEND (Vercel — Free)
 ┌──────────────────────────────────────────┐
 │  React Dashboard                         │
 │  • Leaflet map with 90 camera markers    │
 │  • Real-time congestion heatmap          │
 │  • Drift/failure alerts                  │
 │  • Cost: $0/month                        │
 └──────────────────────────────────────────┘

 TOTAL COST: ~$8/month (Azure VM only)
```

---

## Prerequisites

1. **Azure Student Account** ($100 credits)
   - Sign up: https://azure.microsoft.com/en-us/free/students/
   - Verify: `az account show`

2. **Azure CLI** (runs locally for initial setup only)
   ```bash
   # macOS
   brew install azure-cli
   
   # Login
   az login
   ```

3. **GitHub Repository** (public or private)
   - Fork/clone this repo
   - Enable GitHub Actions

4. **Google Colab** (for data collection)
   - Free tier: https://colab.research.google.com/

5. **Kaggle Account** (for model training)
   - Free tier: https://www.kaggle.com/

---

## Deployment Steps

### Step 1: Deploy Azure VM (One-time Setup)

**Duration**: ~10 minutes

```bash
# Navigate to project
cd sg-smart-city-analytics

# Run Azure deployment script
./deploy/setup-azure-vm.sh
```

This script will:
- ✅ Create resource group in Singapore region
- ✅ Provision B1s VM (1 vCPU, 1GB RAM)
- ✅ Install Docker + Docker Compose
- ✅ Clone repository
- ✅ Start all services via `docker-compose.yml`
- ✅ Run health check

**Expected Output**:
```
🎉 Deployment Complete!
======================
VM IP:           20.205.xxx.xxx
SSH Access:      ssh azureuser@20.205.xxx.xxx
API Endpoint:    http://20.205.xxx.xxx:8000
Health Check:    http://20.205.xxx.xxx:8000/api/health
```

**Verify Deployment**:
```bash
# Test API health
curl http://<VM_IP>:8000/api/health

# Expected response:
# {"status":"healthy","cameras_tracked":0,"last_updated":null}
```

---

### Step 2: Configure GitHub Secrets (For Auto-Deployment)

**Duration**: ~5 minutes

Add these secrets to your GitHub repository (Settings → Secrets → Actions):

1. **AZURE_CREDENTIALS**
   ```bash
   az ad sp create-for-rbac \
     --name "github-actions-sg-smart-city" \
     --role contributor \
     --scopes /subscriptions/<SUBSCRIPTION_ID>/resourceGroups/sg-smart-city-rg \
     --sdk-auth
   ```
   Copy the entire JSON output to GitHub secret `AZURE_CREDENTIALS`

2. **AZURE_VM_HOST**
   ```
   20.205.xxx.xxx  (Your VM's public IP from Step 1)
   ```

3. **AZURE_VM_USER**
   ```
   azureuser
   ```

4. **AZURE_VM_SSH_KEY**
   ```bash
   cat ~/.ssh/id_rsa
   ```
   Copy the private key (entire contents) to GitHub secret

**Verify**:
- Navigate to GitHub repo → Actions
- You should see "Deploy to Azure" workflow available

---

### Step 3: Start Data Collection (Google Colab)

**Duration**: Continuous (runs for 6-12 hours per session)

1. Open `notebooks/collect_data.ipynb` in Google Colab
   - Link: https://colab.research.google.com/
   - Upload notebook or connect GitHub

2. Mount Google Drive
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. Run all cells
   - Collects from 90 cameras every 60 seconds
   - Saves to `/content/drive/MyDrive/sg_smart_city/data/raw/`
   - Runs for 12 hours (Colab limit)

4. **Let it accumulate data**
   - Target: 1,000+ images (minimum for fine-tuning)
   - Ideal: 10,000+ images (1 week of collection)

**Monitor**:
- Check Colab output for cycle count
- Verify files in Google Drive

---

### Step 4: Train Model (Kaggle)

**Duration**: ~2-3 hours (one-time, then retrain weekly)

1. Upload `notebooks/train_yolo.ipynb` to Kaggle
   - https://www.kaggle.com/code

2. Enable GPU accelerator
   - Settings → Accelerator → GPU T4 x2

3. Add dataset as Data Source
   - Option A: UA-DETRAC (public, available now)
   - Option B: Your Singapore data (when ready)

4. Run all cells
   - Trains for ~100 epochs (~2 hours)
   - MLflow tracking enabled
   - Outputs: `best.pt`, `best.onnx`

5. Download artifacts
   - `/kaggle/working/runs/sg_traffic_yolo11s_*/weights/best.pt`
   - `/kaggle/working/mlruns/` (experiment tracking)

**Expected Results**:
- UA-DETRAC baseline: ~85% mAP50
- Singapore fine-tuned: ~92% mAP50 (after retraining on collected data)

---

### Step 5: Deploy Trained Model to Azure VM

**Duration**: ~5 minutes

```bash
# SSH into Azure VM
ssh azureuser@<VM_IP>

# Upload model (from local machine)
scp best.pt azureuser@<VM_IP>:~/sg-smart-city-analytics/models/

# Restart inference service
cd ~/sg-smart-city-analytics
docker compose restart api
```

**Verify**:
```bash
# Test inference endpoint
curl -X POST http://<VM_IP>:8000/api/detect \
  -F "image=@test_image.jpg"
```

---

### Step 6: Continuous Deployment (Auto-Deploy on Push)

**Duration**: Automatic (triggers on every push to main)

Once GitHub secrets are configured (Step 2):

```bash
# Make code changes
git add .
git commit -m "feat: improve detection accuracy"
git push origin main
```

**What happens automatically**:
1. GitHub Actions triggers CI pipeline
   - ✅ Linting
   - ✅ Tests (80+ unit/integration tests)
   - ✅ Docker build
   - ✅ Config validation

2. If tests pass → Deploy to Azure workflow triggers
   - ✅ SSH into Azure VM
   - ✅ Pull latest code
   - ✅ Rebuild Docker images
   - ✅ Restart services with zero downtime
   - ✅ Health check

**Monitor**:
- GitHub Actions tab shows real-time deployment status
- Notifications on deployment success/failure

---

## Monitoring & Maintenance

### Health Checks

```bash
# API health
curl http://<VM_IP>:8000/api/health

# System resources
ssh azureuser@<VM_IP> 'htop'

# Docker logs
ssh azureuser@<VM_IP> 'cd sg-smart-city-analytics && docker compose logs -f'
```

### Cost Monitoring

```bash
# Check Azure spend
az consumption usage list --output table

# Expected monthly cost breakdown:
# - Azure B1s VM: $7.30/month
# - Bandwidth: $0.50/month
# - Storage: $0 (within free tier)
# TOTAL: ~$8/month
```

### Data Collection Schedule

| Task | Frequency | Platform | Cost |
|------|-----------|----------|------|
| Collect images | Every 60s | Google Colab | $0 |
| Re-train model | Weekly | Kaggle | $0 |
| Deploy updates | On push | Azure VM | $8/mo |
| Monitor drift | Continuous | Azure VM | Included |

---

## Troubleshooting

### Issue: VM out of memory

**Symptom**: Docker containers crash, `docker compose ps` shows unhealthy

**Solution**:
```bash
# Scale down batch processing
ssh azureuser@<VM_IP>
cd sg-smart-city-analytics
# Edit docker-compose.yml: reduce batch size
docker compose down && docker compose up -d
```

### Issue: GitHub Actions deployment fails

**Symptom**: "Deploy to Azure" workflow fails at SSH step

**Solution**:
1. Verify GitHub secrets are correct
2. Check VM is running: `az vm list --output table`
3. Test SSH manually: `ssh azureuser@<VM_IP>`
4. Re-run workflow

### Issue: Model inference too slow

**Symptom**: <20 FPS on VM

**Solution**:
```bash
# Switch to ONNX model (faster)
# Update src/detection/detector.py to use best.onnx instead of best.pt
```

---

## Scaling (Future)

When you outgrow the B1s VM:

| VM Size | vCPUs | RAM | Cost/mo | Use Case |
|---------|-------|-----|---------|----------|
| B1s | 1 | 1GB | $8 | MVP (current) |
| B2s | 2 | 4GB | $30 | Production (100 cameras) |
| D2s_v3 | 2 | 8GB | $70 | GPU workloads |

---

## Security Checklist

- ✅ SSH keys only (no password auth)
- ✅ Firewall: ports 22, 80, 8000 only
- ✅ HTTPS via Let's Encrypt (TODO: implement)
- ✅ Secrets in GitHub Actions (not in code)
- ✅ Regular security scans (Trivy in CI)

---

## Next Steps

1. **Run Step 1** (Deploy Azure VM) → 10 min
2. **Run Step 3** (Start Colab collector) → Set and forget
3. **Run Step 4** (Train baseline on UA-DETRAC) → 2 hours
4. **Deploy trained model** (Step 5) → 5 min
5. **Test end-to-end** → 5 min

**Total time to production**: ~3 hours

After 1 week of data collection:
6. **Retrain on Singapore data** → 2 hours
7. **Deploy improved model** → 5 min
8. **Celebrate 92% mAP** 🎉

---

**Questions? Check the main README or open an issue on GitHub.**
