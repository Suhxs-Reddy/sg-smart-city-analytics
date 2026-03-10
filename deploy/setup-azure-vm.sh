#!/bin/bash
#
# Singapore Smart City — Azure VM Setup Script
#
# This script provisions an Azure B1s VM and deploys the full pipeline.
# Cost: ~$8/month from $100 Azure student credits
#
# Usage:
#   ./deploy/setup-azure-vm.sh
#

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

RESOURCE_GROUP="sg-smart-city-rg"
LOCATION="centralus"  # Central US region
VM_NAME="sg-smart-city-vm"
VM_SIZE="Standard_B1s"  # 1 vCPU, 1GB RAM - $8/month
VM_IMAGE="Ubuntu2204"
ADMIN_USER="azureuser"

echo "🚀 Singapore Smart City — Azure Deployment"
echo "============================================"
echo "Resource Group: $RESOURCE_GROUP"
echo "Location:       $LOCATION"
echo "VM Size:        $VM_SIZE (~$8/month)"
echo ""

# ============================================================================
# 1. Create Resource Group
# ============================================================================

echo "📦 Creating resource group..."
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION

# ============================================================================
# 2. Create Virtual Machine
# ============================================================================

echo "🖥️  Creating Azure VM..."
az vm create \
  --resource-group $RESOURCE_GROUP \
  --name $VM_NAME \
  --image $VM_IMAGE \
  --size $VM_SIZE \
  --admin-username $ADMIN_USER \
  --generate-ssh-keys \
  --public-ip-sku Standard \
  --output json > vm-info.json

# Extract VM public IP
VM_IP=$(az vm show -d \
  --resource-group $RESOURCE_GROUP \
  --name $VM_NAME \
  --query publicIps -o tsv)

echo "✅ VM created with IP: $VM_IP"

# ============================================================================
# 3. Open Firewall Ports
# ============================================================================

echo "🔓 Opening firewall ports..."

# Port 22: SSH
# Port 80: HTTP (API)
# Port 443: HTTPS (future)
# Port 8000: FastAPI (temporary, will move to 80)

az vm open-port \
  --resource-group $RESOURCE_GROUP \
  --name $VM_NAME \
  --port 22 \
  --priority 1000

az vm open-port \
  --resource-group $RESOURCE_GROUP \
  --name $VM_NAME \
  --port 80 \
  --priority 1001

az vm open-port \
  --resource-group $RESOURCE_GROUP \
  --name $VM_NAME \
  --port 8000 \
  --priority 1002

echo "✅ Ports 22, 80, 8000 opened"

# ============================================================================
# 4. Install Dependencies on VM
# ============================================================================

echo "📦 Installing Docker and dependencies on VM..."

ssh -o StrictHostKeyChecking=no $ADMIN_USER@$VM_IP << 'ENDSSH'
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt-get install docker-compose-plugin -y

# Install Git
sudo apt-get install git -y

# Install monitoring tools
sudo apt-get install htop iotop ncdu -y

echo "✅ Dependencies installed"
ENDSSH

# ============================================================================
# 5. Clone Repository and Deploy
# ============================================================================

echo "📥 Cloning repository and deploying..."

ssh $ADMIN_USER@$VM_IP << 'ENDSSH'
# Clone repo
git clone https://github.com/Suhxs-Reddy/sg-smart-city-analytics.git
cd sg-smart-city-analytics

# Start services
docker compose up -d

# Wait for services to start
sleep 15

# Health check
curl -f http://localhost:8000/api/health || {
  echo "❌ Health check failed"
  docker compose logs
  exit 1
}

echo "✅ Services deployed and healthy"
docker compose ps
ENDSSH

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "🎉 Deployment Complete!"
echo "======================"
echo "VM IP:           $VM_IP"
echo "SSH Access:      ssh $ADMIN_USER@$VM_IP"
echo "API Endpoint:    http://$VM_IP:8000"
echo "Health Check:    http://$VM_IP:8000/api/health"
echo ""
echo "📊 Next Steps:"
echo "  1. Test API: curl http://$VM_IP:8000/api/health"
echo "  2. View logs: ssh $ADMIN_USER@$VM_IP 'cd sg-smart-city-analytics && docker compose logs -f'"
echo "  3. Monitor resources: ssh $ADMIN_USER@$VM_IP 'htop'"
echo ""
echo "💰 Estimated cost: ~$8/month (from $100 student credits)"
echo ""

# Save connection info
cat > deployment-info.txt << ENDINFO
Deployment Information
======================
Date: $(date)
VM IP: $VM_IP
SSH: ssh $ADMIN_USER@$VM_IP
API: http://$VM_IP:8000
ENDINFO

echo "✅ Deployment info saved to: deployment-info.txt"
