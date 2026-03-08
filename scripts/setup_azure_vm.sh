#!/bin/bash
# =============================================================================
# Singapore Smart City — Azure VM Setup Script
# =============================================================================
# Run this on your Azure B1s VM to set up the data collection environment.
#
# Prerequisites:
#   - Azure for Students account with $100 credits
#   - Created a B1s VM (Ubuntu 22.04 LTS, cheapest tier ~$4/mo)
#   - SSH'd into the VM
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/Suhxs-Reddy/sg-smart-city-analytics/main/scripts/setup_azure_vm.sh | bash
#   OR
#   bash scripts/setup_azure_vm.sh
# =============================================================================

set -e  # Exit on error

echo "=========================================="
echo "  🇸🇬 Singapore Smart City — VM Setup"
echo "=========================================="

# Step 1: System updates
echo ""
echo "Step 1: System updates..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv git

# Step 2: Clone repo
echo ""
echo "Step 2: Cloning repository..."
if [ ! -d "sg-smart-city-analytics" ]; then
    git clone https://github.com/Suhxs-Reddy/sg-smart-city-analytics.git
fi
cd sg-smart-city-analytics

# Step 3: Python environment
echo ""
echo "Step 3: Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Install only what's needed for data collection (no GPU deps)
pip install -q requests aiohttp Pillow imagehash pyyaml click pandas

# Step 4: Create data directories
echo ""
echo "Step 4: Creating data directories..."
mkdir -p data/raw logs

# Step 5: Test API connection
echo ""
echo "Step 5: Testing API connection..."
python3 -c "
import requests
resp = requests.get('https://api.data.gov.sg/v1/transport/traffic-images', timeout=10)
data = resp.json()
cameras = data['items'][0]['cameras']
print(f'  ✅ API connected: {len(cameras)} cameras responding')
"

# Step 6: Set up systemd service for continuous collection
echo ""
echo "Step 6: Setting up systemd service..."
sudo tee /etc/systemd/system/sg-collector.service > /dev/null <<EOF
[Unit]
Description=Singapore Smart City Data Collector
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/venv/bin/python -m src.ingestion.collector --duration 720 --interval 60
Restart=always
RestartSec=60
StandardOutput=append:$(pwd)/logs/collector.log
StandardError=append:$(pwd)/logs/collector_error.log

[Install]
WantedBy=multi-user.target
EOF

echo ""
echo "=========================================="
echo "  ✅ Setup Complete!"
echo "=========================================="
echo ""
echo "  To start collecting data:"
echo "    sudo systemctl enable sg-collector"
echo "    sudo systemctl start sg-collector"
echo ""
echo "  To check status:"
echo "    sudo systemctl status sg-collector"
echo "    tail -f logs/collector.log"
echo ""
echo "  To stop:"
echo "    sudo systemctl stop sg-collector"
echo ""
echo "  Data will be saved to: data/raw/"
echo "  Logs: logs/collector.log"
echo ""
echo "  Estimated storage: ~2 GB/day (90 cameras × 1440 cycles)"
echo "  Azure B1s has 4 GB RAM, 30 GB disk — sufficient for ~2 weeks"
echo ""
