# 🇸🇬 Singapore Smart City: Zero-Compute Execution Guide

This guide details how to execute the entire Senior-Level Machine Learning Architecture **without utilizing a single CPU or GPU cycle on your local Mac**. 

By leveraging your `.edu` email address (GitHub Student Developer Pack) and Google Colab's free tiers, you can run this entire enterprise system for **$0**, entirely in the cloud.

---

## 1. Running the Digital Twin & Event Streaming (GitHub Codespaces)
Instead of running `docker-compose` on your Mac, you can run the entire backend (Redpanda Kafka, Milvus Vector DB, ML Inference API) on Microsoft's servers.

**Steps:**
1. Push this local repository to your GitHub account:
   ```bash
   git add .
   git commit -m "Initialize Zero-Compute Architecture"
   git push origin main
   ```
2. Navigate to your repository on `github.com`.
3. Click the green **`<> Code`** button.
4. Select the **Codespaces** tab and click **Create codespace on main**.
5. Microsoft will automatically provision a remote Ubuntu Linux machine via the `.devcontainer.json` configuration we created.
6. Once the browser-based VS Code opens, open the terminal and run:
   ```bash
   docker-compose up -d --build
   ```
*Result:* The entire system is now running on a remote cloud server. Your local Mac CPU remains at 0% usage. The API will be exposed to a secure GitHub URL.

## 2. Training the Advanced ML Models (Google Colab / Kaggle)
Do not train Neural ODEs or YOLO Knowledge Distillation on your Mac. It will overheat the machine and take days. We use free Cloud GPUs.

**Steps:**
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload the `scripts/train_yolo_colab.py` file to Colab.
3. In Colab, go to **Runtime \> Change runtime type** and select **T4 GPU** (Free).
4. Run the script. It will automatically download the dataset from the cloud, utilize the T4 GPU to distill the weights, and export an optimized `best.onnx` inference model directly to your Google Drive.
5. (Optional) Repeat the same process for `notebooks/train_stgnn_pinn.ipynb` to train the Neural ODEs dynamically.

---

## 3. Final Deployment (Phase 8 Architecture)
If you ever gain access to AWS Educate credits or GCP Free Tier, the Kubernetes definitions located in the `k8s/` folder and the GitHub Actions pipeline (`.github/workflows/ml_deploy.yml`) will automatically detect the new weights and trigger an enterprise rolling-update onto your active production cluster.

### Summary
* **Local Hardware Risk:** 0%
* **Budget Used:** $0.00
* **Architecture Standard:** Senior ML Engineer (Event Streaming, Vector RAG, Neural ODEs)
