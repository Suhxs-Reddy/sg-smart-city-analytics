"""
Singapore Smart City — Enterprise Streaming Producer (AWS Scale)

This module handles asynchronous ingestion of raw traffic frames into 
Apache Kafka / Redpanda topics. This enables the 'Massive Project' scale 
required for a real-world Smart City deployment.
"""

import json
import logging
import time
from pathlib import Path

try:
    from kafka import KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

logger = logging.getLogger(__name__)

class TrafficStreamProducer:
    def __init__(self, bootstrap_servers: list = ["localhost:9092"]):
        self.topic = "raw-traffic-ingestion"
        
        if KAFKA_AVAILABLE:
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    acks='all',
                    retries=5
                )
                logger.info(f"Connected to Redpanda/Kafka clusters at {bootstrap_servers}")
            except Exception as e:
                logger.error(f"Failed to connect to streaming backbone: {e}")
                self.producer = None
        else:
            logger.warning("Kafka-python not installed. Streaming ingest will be mocked.")
            self.producer = None

    def publish_frame(self, camera_id: str, image_path: Path, metadata: dict):
        """
        Publishes a camera event to the streaming bus.
        """
        payload = {
            "camera_id": camera_id,
            "timestamp": time.time(),
            "image_uri": str(image_path.absolute()),
            "metadata": metadata
        }
        
        if self.producer:
            self.producer.send(self.topic, payload)
            logger.info(f"Event published to {self.topic} for {camera_id}")
        else:
            logger.info(f"[MOCK STREAM] camera={camera_id} event recorded.")
            
        return payload

if __name__ == "__main__":
    producer = TrafficStreamProducer()
    print("Enterprise Streaming Infrastructure initialized.")
