# Singapore Smart City — AWS Deployment Architecture

As the local and Colab resources are hitting limits or encountering timeouts, we need to migrate the core platform to a more resilient cloud architecture.

Given the project requirements for real-time video processing, scalable model inference, and a highly available dashboard backend, here is the recommended **AWS Architecture** to replace the local/Colab setup:

## 1. Data Collection (The Ingestion Layer)
*   **Current:** Colab notebook that times out or Google Drive that fills up.
*   **AWS Solution: Amazon EventBridge + AWS Lambda + Amazon S3**
    *   **How it works:** EventBridge runs a cron schedule every 60 seconds to trigger an AWS Lambda function.
    *   **Lambda:** Runs a lightweight version of `collector.py` that hits the `data.gov.sg` APIs.
    *   **Storage:** The images and JSONL metadata are streamed directly into an **Amazon S3 Bucket** (`s3://sg-smart-city-raw-data`).
    *   **Why:** Serverless, costs pennies, infinitely scalable storage, and cannot "time out" like Colab.

## 2. Model Training (The Deep Learning Layer)
*   **Current:** Kaggle Free T4 GPU.
*   **AWS Solution: Amazon SageMaker (Training Jobs)**
    *   **How it works:** When we have enough data in S3, we trigger a SageMaker Training Job using an `ml.g4dn.xlarge` instance (which contains an NVIDIA T4 GPU).
    *   **Data flow:** SageMaker pulls the raw dataset from S3, runs the `train_yolo.ipynb` equivalent script, and outputs the trained `best.pt` model weights back into S3.
    *   **Why:** You only pay for the exact minutes the GPU is training. No idle costs. Fully reproducible MLflow tracking built-in.

## 3. Real-Time Inference Pipeline (The Processing Layer)
*   **Current:** Local Python script (`pipeline.py`).
*   **AWS Solution: Amazon ECS (Fargate) + Amazon SQS**
    *   **How it works:** As new images hit the S3 bucket, an S3 Event triggers a message to an SQS Queue.
    *   **Workers:** A fleet of containerized Python workers running on ECS Fargate polls the queue, downloads the image, runs the YOLO detection + tracking inference, and calculates the congestion score.
    *   **Why:** Decouples ingestion from processing. If traffic spikes, ECS Auto-Scales the number of worker containers.

## 4. API & Database Layer (The Backend)
*   **Current:** Local FastAPI server writing to local JSON.
*   **AWS Solution: Amazon API Gateway + AWS Lambda + Amazon DynamoDB**
    *   **Database:** The ECS workers save the final congestion scores, vehicle counts, and anomaly alerts into DynamoDB (a fast NoSQL database perfect for time-series/IoT data).
    *   **API:** API Gateway hosts the REST endpoints (e.g., `/api/cameras/congestion`), backed by lightweight Lambda functions that query DynamoDB.
    *   **Why:** Zero server maintenance, sub-millisecond latency for the dashboard.

## 5. Dashboard (The Frontend)
*   **Current:** Vercel (Successfully Deployed: [Live URL](https://dashboard-ecru-sigma-62.vercel.app))
*   **AWS Alternative (Optional): Amazon CloudFront + S3**
    *   Vercel is actually perfect for this React app. But if you want a 100% AWS stack, you would compile the Vite React app into static files and host them in an S3 Bucket distributed via the CloudFront CDN.

---

## The Migration Path
To execute this, we will need to:
1.  **Set up the AWS CLI** locally with an IAM user.
2.  Create the **S3 Bucket** and migrate any existing data from Google Drive.
3.  Deploy the data collector as a **Lambda function**.
4.  Containerize the inference `pipeline.py` using **Docker** and push it to Amazon ECR.
5.  Set up the **DynamoDB** tables.
