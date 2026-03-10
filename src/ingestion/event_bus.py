import os
import json
import asyncio
import logging
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

logger = logging.getLogger(__name__)

# Environment variables to target local Redpanda/Kafka cluster
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
RAW_FRAMES_TOPIC = "sg_traffic_raw_frames"
INFERENCE_TOPIC = "sg_traffic_inferences"


class EventStreamingBus:
    """
    Asynchronous Event Streaming Manager using Apache Kafka / Redpanda.
    This entirely decouples ingestion (downloading images) from processing (running ML models),
    allowing massive fault-tolerance and throughput scaling.
    """

    def __init__(self):
        self.producer = None
        self.consumer = None

    async def start_producer(self):
        """Initializes the producer to write messages to the bus."""
        self.producer = AIOKafkaProducer(
            bootstrap_servers=KAFKA_BROKER, value_serializer=lambda v: json.dumps(v).encode("utf-8")
        )
        await self.producer.start()
        logger.info(f"Kafka Producer connected to {KAFKA_BROKER}")

    async def publish_raw_frame(self, camera_id: str, timestamp: str, image_payload_b64: str):
        """
        Ingestion Layer: Publishes raw images to the queue for the ML Engine to pick up later.
        """
        payload = {"camera_id": camera_id, "timestamp": timestamp, "image_data": image_payload_b64}
        await self.producer.send_and_wait(RAW_FRAMES_TOPIC, value=payload)

    async def publish_inference_result(self, result_dict: dict):
        """
        Inference Layer: Publishes the final output from the 3-Tier engine.
        """
        await self.producer.send_and_wait(INFERENCE_TOPIC, value=result_dict)

    async def start_consumer(self, topic_name: str, group_id: str):
        """Initializes a consumer to read streams from the bus."""
        self.consumer = AIOKafkaConsumer(
            topic_name,
            bootstrap_servers=KAFKA_BROKER,
            group_id=group_id,
            value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        )
        await self.consumer.start()
        logger.info(f"Kafka Consumer connected to {topic_name} acting as {group_id}")
        return self.consumer

    async def close(self):
        if self.producer:
            await self.producer.stop()
        if self.consumer:
            await self.consumer.stop()


# Example usage pattern when ran independently
async def demo():
    bus = EventStreamingBus()
    await bus.start_producer()
    await bus.publish_inference_result({"id": "CAM_1", "l1_yolo_count": 14})
    await bus.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo())
