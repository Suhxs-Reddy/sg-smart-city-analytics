#!/bin/bash
set -e

# Singapore Smart City Analytics — Execution Script
# Follow this step-by-step to deploy everything in 2-3 days

echo "🚀 Singapore Smart City Analytics — Deployment Execution"
echo "========================================================="
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Step 1: Check GitHub Actions
echo -e "${BLUE}STEP 1: GitHub Actions CI/CD${NC}"
echo "---------------------------------------------------"
echo "Check: https://github.com/Suhxs-Reddy/sg-smart-city-analytics/actions"
echo ""
echo -e "${YELLOW}Waiting for CI/CD to complete...${NC}"
echo "Expected: All workflows should show green checkmarks ✅"
echo ""
read -p "Press ENTER when all workflows are green..."
echo -e "${GREEN}✅ CI/CD passed${NC}"
echo ""

# Step 2: Azure VM Deployment
echo -e "${BLUE}STEP 2: Deploy Azure VM${NC}"
echo "---------------------------------------------------"
echo "This will create:"
echo "  - Resource Group: sg-smart-city-rg"
echo "  - VM: Standard_B1s (Singapore region)"
echo "  - Cost: $8/month from your $100 Azure credits"
echo ""
read -p "Ready to deploy Azure VM? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Running Azure deployment...${NC}"
    ./deploy/setup-azure-vm.sh

    echo ""
    echo -e "${GREEN}✅ Azure VM deployed${NC}"
    echo ""
    echo -e "${YELLOW}IMPORTANT: Save your VM IP address!${NC}"
    read -p "Enter your VM IP (e.g., 20.205.xxx.xxx): " VM_IP

    # Save VM IP to config
    echo "AZURE_VM_IP=$VM_IP" > .env.deployment
    echo -e "${GREEN}Saved VM IP: $VM_IP${NC}"
    echo ""
else
    echo -e "${RED}Skipped Azure deployment${NC}"
    echo ""
fi

# Step 3: Kaggle Training
echo -e "${BLUE}STEP 3: Train Model on Kaggle${NC}"
echo "---------------------------------------------------"
echo "Manual steps (NO local execution):"
echo ""
echo "1. Go to: https://www.kaggle.com"
echo "2. Sign in with your account"
echo "3. Click 'Create' → 'New Notebook'"
echo "4. Upload: notebooks/train_yolo.ipynb"
echo "5. Settings → Accelerator → GPU T4 x2"
echo "6. Click 'Run All' (takes 2-3 hours)"
echo "7. When done, download 'best.pt' from output"
echo ""
echo -e "${YELLOW}Training will take 2-3 hours. Go do something else!${NC}"
echo ""
read -p "Press ENTER when training is complete and you have best.pt..."
echo ""

# Step 4: Deploy Model to Azure
echo -e "${BLUE}STEP 4: Deploy Trained Model to Azure${NC}"
echo "---------------------------------------------------"

if [ -f .env.deployment ]; then
    source .env.deployment
    echo "VM IP: $AZURE_VM_IP"
    echo ""

    read -p "Path to your downloaded best.pt file: " MODEL_PATH

    if [ -f "$MODEL_PATH" ]; then
        echo -e "${YELLOW}Uploading model to Azure VM...${NC}"
        scp "$MODEL_PATH" "azureuser@$AZURE_VM_IP:~/sg-smart-city-analytics/models/best.pt"

        echo -e "${YELLOW}Restarting API with new model...${NC}"
        ssh "azureuser@$AZURE_VM_IP" 'cd sg-smart-city-analytics && docker compose restart api'

        echo ""
        echo -e "${GREEN}✅ Model deployed to Azure${NC}"
        echo ""

        # Test API
        echo -e "${YELLOW}Testing API...${NC}"
        curl -s "http://$AZURE_VM_IP:8000/api/health" | python3 -m json.tool
        echo ""
        echo -e "${GREEN}✅ API is responding${NC}"
        echo ""
    else
        echo -e "${RED}Error: best.pt not found at $MODEL_PATH${NC}"
        echo "Please download best.pt from Kaggle first"
        exit 1
    fi
else
    echo -e "${RED}Error: VM IP not found. Did you deploy Azure VM in Step 2?${NC}"
    exit 1
fi

# Step 5: Deploy Dashboard to Vercel
echo -e "${BLUE}STEP 5: Deploy Dashboard to Vercel${NC}"
echo "---------------------------------------------------"
echo "Manual steps (NO local execution):"
echo ""
echo "1. Go to: https://vercel.com"
echo "2. Sign in with GitHub"
echo "3. Click 'Add New Project'"
echo "4. Import: Suhxs-Reddy/sg-smart-city-analytics"
echo "5. Root Directory: dashboard/"
echo "6. Environment Variables:"
echo "   NEXT_PUBLIC_API_URL = http://$AZURE_VM_IP:8000"
echo "7. Click 'Deploy'"
echo "8. Wait 5-10 minutes"
echo "9. Copy your Vercel URL"
echo ""
read -p "Press ENTER when Vercel deployment is complete..."
echo ""
read -p "Enter your Vercel URL (e.g., https://sg-smart-city-xxx.vercel.app): " VERCEL_URL

# Save Vercel URL
echo "VERCEL_URL=$VERCEL_URL" >> .env.deployment
echo -e "${GREEN}✅ Dashboard deployed to Vercel${NC}"
echo ""

# Step 6: Verify Everything
echo -e "${BLUE}STEP 6: Verify End-to-End System${NC}"
echo "---------------------------------------------------"
echo ""
echo -e "${YELLOW}Testing backend API...${NC}"
curl -s "http://$AZURE_VM_IP:8000/api/health"
echo ""
echo ""
echo -e "${YELLOW}Opening dashboard in browser...${NC}"
open "$VERCEL_URL" || xdg-open "$VERCEL_URL" || echo "Open manually: $VERCEL_URL"
echo ""

# Final Summary
echo ""
echo "========================================================="
echo -e "${GREEN}🎉 DEPLOYMENT COMPLETE!${NC}"
echo "========================================================="
echo ""
echo "Your System:"
echo "  🌐 Live Dashboard: $VERCEL_URL"
echo "  🔧 API Endpoint: http://$AZURE_VM_IP:8000"
echo "  📚 API Docs: http://$AZURE_VM_IP:8000/docs"
echo "  📊 GitHub: https://github.com/Suhxs-Reddy/sg-smart-city-analytics"
echo ""
echo "Cost Breakdown:"
echo "  💰 Azure VM: \$8/month (from \$100 credits)"
echo "  💰 Vercel: \$0 (free tier)"
echo "  💰 Kaggle: \$0 (free GPU)"
echo "  💰 Total: \$8/month"
echo ""
echo "What You Can Show Recruiters:"
echo "  ✅ Live website with 90 cameras"
echo "  ✅ Real-time traffic detection"
echo "  ✅ Production-grade GitHub repo"
echo "  ✅ Full CI/CD pipeline"
echo "  ✅ Cost-efficient deployment"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "  1. Take screenshots of dashboard"
echo "  2. Record 2-3 minute demo video"
echo "  3. Update README.md with live demo link"
echo "  4. Share with recruiters!"
echo ""
echo -e "${GREEN}Ready to impress! 🚀${NC}"
echo ""
