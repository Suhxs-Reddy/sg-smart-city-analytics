# Deploy to AWS EC2 (30 Minutes) — Production Grade

**Zero local CLI required** - Everything via AWS Console

---

## What You'll Get

```
✅ EC2 t2.micro instance (1GB RAM, 1 vCPU)
✅ Free for 12 months (AWS Free Tier)
✅ Ubuntu 22.04 LTS
✅ Docker + Docker Compose
✅ FastAPI backend running
✅ Public IP for Vercel connection
✅ Security group (ports 22, 80, 8000)
✅ Cost: $0 for first year
```

**After this**: Connect to Vercel dashboard → Complete system live

---

## Prerequisites

- AWS Account (free tier eligible)
- If you don't have one: https://aws.amazon.com/free → Sign up → Verify with credit card (won't be charged)

---

## Step 1: Launch EC2 Instance (10 minutes)

### 1.1 Open EC2 Console

Go to: https://console.aws.amazon.com/ec2

Or: AWS Console → Search "EC2" → Click "EC2"

### 1.2 Launch Instance

1. Click **"Launch Instance"** (orange button)

### 1.3 Configure Instance

**Name and tags**:
- Name: `sg-smart-city-backend`

**Application and OS Images (AMI)**:
- Quick Start: **Ubuntu**
- AMI: **Ubuntu Server 22.04 LTS (HVM), SSD Volume Type**
- Architecture: **64-bit (x86)**

**Instance type**:
- **t2.micro** (1 vCPU, 1 GiB RAM) ← **Free tier eligible**
- Shows green label: "Free tier eligible"

**Key pair (login)**:
- Click **"Create new key pair"**
- Key pair name: `sg-smart-city-key`
- Key pair type: **RSA**
- Private key file format: **.pem** (for Mac/Linux) or **.ppk** (for Windows)
- Click **"Create key pair"**
- **Save the file!** → You'll need it to SSH

**Network settings**:
- Click **"Edit"**
- Auto-assign public IP: **Enable**
- Firewall (security groups): **Create security group**
- Security group name: `sg-smart-city-sg`
- Description: `Traffic + API + SSH access`

**Security group rules** (add these):
1. **SSH** (already there)
   - Type: SSH
   - Port: 22
   - Source: My IP (or 0.0.0.0/0 for anywhere)

2. **HTTP** (click "Add security group rule")
   - Type: HTTP
   - Port: 80
   - Source: 0.0.0.0/0 (Anywhere)

3. **Custom TCP** (click "Add security group rule")
   - Type: Custom TCP
   - Port: 8000
   - Source: 0.0.0.0/0 (Anywhere)
   - Description: FastAPI

**Configure storage**:
- Size: **30 GiB** (free tier includes up to 30 GB)
- Volume type: **gp3** or **gp2** (both free tier eligible)

**Advanced details**: Leave defaults

### 1.4 Launch

1. Review summary on right side
2. Click **"Launch instance"** (orange button)
3. Wait ~60 seconds
4. Click **"View all instances"**

### 1.5 Get Public IP

1. Select your instance (checkbox)
2. Copy **"Public IPv4 address"** from details below
   - Example: `54.123.45.67`
3. **Save this IP** → You'll need it for Vercel

---

## Step 2: Connect to Instance (5 minutes)

### 2.1 Wait for Instance Ready

In EC2 Console:
- **Instance state**: Running ✅
- **Status check**: 2/2 checks passed ✅

Wait ~2 minutes after launch for status checks.

### 2.2 Connect via SSH

**On Mac/Linux**:
```bash
# Move key to safe location
mv ~/Downloads/sg-smart-city-key.pem ~/.ssh/
chmod 400 ~/.ssh/sg-smart-city-key.pem

# Connect (replace with YOUR IP)
ssh -i ~/.ssh/sg-smart-city-key.pem ubuntu@54.123.45.67
```

**On Windows**:
- Use PuTTY with .ppk key
- Or use AWS Console "Connect" button → "EC2 Instance Connect"

**Or use AWS Console SSH** (easiest):
1. Select instance
2. Click **"Connect"** (top right)
3. Tab: **"EC2 Instance Connect"**
4. Click **"Connect"**
5. Opens browser terminal ✅

---

## Step 3: Install Dependencies (10 minutes)

Once connected via SSH, run these commands:

### 3.1 Update System
```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### 3.2 Install Docker
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo apt-get install docker-compose-plugin -y

# Verify
docker --version
docker compose version
```

**Important**: After adding user to docker group, reconnect SSH:
```bash
exit
# Then ssh back in
```

### 3.3 Install Git
```bash
sudo apt-get install git -y
```

### 3.4 Install Monitoring Tools (optional but recommended)
```bash
sudo apt-get install htop iotop ncdu -y
```

---

## Step 4: Deploy Application (5 minutes)

### 4.1 Clone Repository
```bash
git clone https://github.com/Suhxs-Reddy/sg-smart-city-analytics.git
cd sg-smart-city-analytics
```

### 4.2 Create Environment File
```bash
cat > .env << 'EOF'
# Singapore LTA API Key (get from: https://datamall.lta.gov.sg/content/datamall/en/request-for-api.html)
LTA_API_KEY=your_api_key_here

# Model path
MODEL_PATH=models/yolo11s_traffic.pt

# API settings
API_HOST=0.0.0.0
API_PORT=8000
EOF
```

**Note**: You can run without LTA_API_KEY for now (will use demo mode)

### 4.3 Download Pre-trained Model (if you have it)

**If you trained on Kaggle**:
```bash
# From your local machine (separate terminal):
scp -i ~/.ssh/sg-smart-city-key.pem best.pt ubuntu@54.123.45.67:~/sg-smart-city-analytics/models/yolo11s_traffic.pt
```

**If you don't have trained model yet**:
```bash
# Create placeholder (API will work in demo mode)
mkdir -p models
touch models/yolo11s_traffic.pt
```

### 4.4 Start Services
```bash
# Build and start containers
docker compose up -d

# Check containers are running
docker compose ps

# Should see:
# NAME                        STATUS
# sg-smart-city-analytics-api RUNNING
```

### 4.5 Check Logs
```bash
# View logs
docker compose logs -f api

# Should see:
# INFO:     Started server process
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

Press `Ctrl+C` to exit logs

---

## Step 5: Verify Deployment (2 minutes)

### 5.1 Test API from EC2 Instance
```bash
# Health check
curl http://localhost:8000/api/health

# Should return:
# {"status":"healthy","cameras":0,"detections":0}
```

### 5.2 Test API from Your Browser

Open: `http://YOUR_EC2_IP:8000/api/health`

Example: `http://54.123.45.67:8000/api/health`

**Should see**:
```json
{
  "status": "healthy",
  "cameras": 0,
  "detections": 0,
  "uptime": "0:00:15"
}
```

### 5.3 Check API Docs

Open: `http://YOUR_EC2_IP:8000/docs`

Should see interactive FastAPI Swagger UI ✅

---

## Step 6: Connect to Vercel Dashboard (5 minutes)

Now that backend is running, connect your Vercel dashboard:

### 6.1 Update Vercel Environment Variable

1. Go to https://vercel.com
2. Select your project: `sg-smart-city-analytics`
3. Settings → Environment Variables
4. Add or edit:
   - Key: `NEXT_PUBLIC_API_URL`
   - Value: `http://YOUR_EC2_IP:8000`
   - Example: `http://54.123.45.67:8000`
5. Click **"Save"**

### 6.2 Redeploy Dashboard

1. Go to Deployments tab
2. Click "..." on latest deployment
3. Click **"Redeploy"**
4. Wait 2-3 minutes
5. Open your Vercel URL

**Dashboard should now show**:
- ✅ Real API connection (not demo mode)
- ✅ Health status from backend
- ✅ "Demo Mode" banner removed

---

## Troubleshooting

### Issue: Can't connect via SSH

**Check**:
1. Instance is "Running" with 2/2 status checks
2. Security group allows SSH (port 22) from your IP
3. Using correct key file path
4. Key has correct permissions: `chmod 400 key.pem`

**Fix**: Use AWS Console "Connect" → "EC2 Instance Connect" instead

### Issue: Can't access API on port 8000

**Check security group**:
```bash
# On EC2, check if port is listening
sudo netstat -tlnp | grep 8000

# Should show:
# tcp 0.0.0.0:8000 ... LISTEN
```

**Fix**: Add port 8000 to security group (Step 1.3)

### Issue: Docker containers not starting

**Check logs**:
```bash
docker compose logs api

# Look for errors
```

**Common causes**:
- Out of memory (t2.micro has only 1GB)
- Missing dependencies

**Fix**: Restart containers:
```bash
docker compose down
docker compose up -d
```

### Issue: API returns 500 errors

**Check**:
```bash
# View detailed logs
docker compose logs -f api

# Check disk space
df -h

# Check memory
free -h
```

**Fix**: May need to reduce batch size or use smaller model

---

## Cost Breakdown

| Component | Cost | Free Tier | Notes |
|-----------|------|-----------|-------|
| **EC2 t2.micro** | $8.50/month | **12 months free** | 750 hours/month free |
| **EBS Storage (30GB)** | $3/month | **12 months free** | 30 GB free |
| **Data Transfer Out** | $0.09/GB | First 100 GB free/month | |
| **Elastic IP** | $0/month | Free if attached to running instance | |
| **TOTAL (Year 1)** | **$0/month** | Free Tier | |
| **TOTAL (After Year 1)** | **~$12/month** | | Still cheap! |

**Optimization**: Set up billing alert at $5 to avoid surprises

---

## Production Optimizations (Optional)

### Enable Auto-Start on Reboot
```bash
# Create systemd service
sudo tee /etc/systemd/system/sg-smart-city.service > /dev/null << 'EOF'
[Unit]
Description=Singapore Smart City Analytics
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ubuntu/sg-smart-city-analytics
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

# Enable service
sudo systemctl enable sg-smart-city
sudo systemctl start sg-smart-city
```

### Set Up CloudWatch Monitoring (Free Tier)
```bash
# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i amazon-cloudwatch-agent.deb

# Configure basic metrics (CPU, memory, disk)
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config \
  -m ec2 \
  -s \
  -c default
```

### Enable Automatic Security Updates
```bash
sudo apt-get install unattended-upgrades -y
sudo dpkg-reconfigure -plow unattended-upgrades
```

---

## Maintenance Commands

### View Logs
```bash
# API logs
docker compose logs -f api

# System logs
sudo journalctl -u sg-smart-city -f
```

### Restart Services
```bash
cd ~/sg-smart-city-analytics
docker compose restart
```

### Update Code
```bash
cd ~/sg-smart-city-analytics
git pull origin main
docker compose down
docker compose up -d --build
```

### Check Resource Usage
```bash
# CPU, memory, processes
htop

# Disk usage
df -h
ncdu /

# Docker stats
docker stats
```

### Backup Important Data
```bash
# Backup models
scp -i ~/.ssh/sg-smart-city-key.pem \
  ubuntu@54.123.45.67:~/sg-smart-city-analytics/models/yolo11s_traffic.pt \
  ./backup/

# Backup detection results
scp -r -i ~/.ssh/sg-smart-city-key.pem \
  ubuntu@54.123.45.67:~/sg-smart-city-analytics/data/detections \
  ./backup/
```

---

## Summary

**Time**: 30 minutes
**Cost**: $0 (first 12 months)
**Result**: Production backend running on AWS

**What You Have**:
- ✅ FastAPI backend live at `http://YOUR_IP:8000`
- ✅ Docker containers managed
- ✅ Automatic restart on reboot (if configured)
- ✅ CloudWatch monitoring (if configured)
- ✅ Connected to Vercel dashboard

**Next Steps**:
1. Upload training notebook to Kaggle
2. Train YOLOv11s model (2-3 hours)
3. SCP trained model to EC2
4. Restart API with new model
5. Dashboard shows real detections ✅

---

## Emergency Commands

**If things go wrong**:
```bash
# Nuclear option - restart everything
sudo reboot

# Remove everything and start fresh
docker compose down -v
rm -rf ~/sg-smart-city-analytics
# Then follow deployment steps again
```

---

**Ready to deploy?** → Follow steps above, you'll have a live backend in 30 minutes! 🚀
