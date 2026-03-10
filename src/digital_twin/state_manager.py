import os
import json
import asyncio
import logging
from typing import Dict, Any

try:
    import torch
    import torch_geometric
    from torch_geometric.data import Data
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from src.ingestion.event_bus import EventStreamingBus, INFERENCE_TOPIC

logger = logging.getLogger(__name__)

class DigitalTwinStateManager:
    """
    Singapore Smart City - Digital Twin Graph Manager
    
    This continuously listens to the `inference-results` Kafka topic and updates
    a live PyTorch Geometric (PyG) Graph representing the city's traffic state.
    
    Because this is purely asynchronous, it runs in the cloud (AWS/ECS) and is 
    decoupled from the edge devices taking the pictures.
    """
    def __init__(self):
        self.bus = EventStreamingBus()
        self.nodes: Dict[str, dict] = {}
        self.node_mapping: Dict[str, int] = {}
        
        # PyG Graph State
        self.live_graph: Data | None = None
        
        self._running = False

    async def initialize(self):
        """Sets up the state manager to read from the inference stream."""
        if not ML_AVAILABLE:
            logger.warning("PyTorch/PyG not available. Running in mocked state.")
            
        await self.bus.start_consumer(INFERENCE_TOPIC, group_id="digital_twin_manager")
        logger.info("Digital Twin State Manager initialized. Subscribed to inference stream.")

    def _update_live_graph(self):
        """
        Reconstructs the PyTorch Geometric representation of the city 
        based on the latest ingested inference data.
        """
        if not ML_AVAILABLE or not self.nodes:
            return

        # Simplified representation for Phase 3 prototype
        num_nodes = len(self.nodes)
        
        # In a real environment, node features would be a rich multi-dimensional vector
        # (e.g., [vehicle_count, speed, anomaly_encoded_value, time_of_day])
        features = []
        for cam_id, data in self.nodes.items():
            if cam_id not in self.node_mapping:
                self.node_mapping[cam_id] = len(self.node_mapping)
            features.append([
                float(data.get("l1_yolo_count", 0)), 
                1.0 if data.get("l2_vlm_anomaly") else 0.0
            ])
            
        x = torch.tensor(features, dtype=torch.float)
        
        # In production, edge_index would be built dynamically based on lat/lon proximity
        # using Haversine distance, representing the physical road network.
        # Here we mock a basic sequential ring topology for the twin demonstration.
        edges = []
        nodes_list = list(self.node_mapping.values())
        if num_nodes > 1:
            for i in range(num_nodes):
                edges.append([nodes_list[i], nodes_list[(i + 1) % num_nodes]])
                edges.append([nodes_list[(i + 1) % num_nodes], nodes_list[i]])
        else:
            edges = [[0, 0]]
            
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Update the live Digital Twin PyG Data object
        self.live_graph = Data(x=x, edge_index=edge_index)
        logger.debug(f"Updated Digital Twin Graph. Nodes: {num_nodes}")

    async def consume_loop(self):
        """
        Endless loop consuming inference results and updating the Digital Twin graph.
        Never runs locally on user's machine; meant for AWS ECS Fargate or similar cloud service.
        """
        if not self.bus.consumer:
            raise RuntimeError("Consumer not initialized.")
            
        self._running = True
        logger.info("\ud83c\udf10 Digital Twin active. Awaiting real-time inferences...")
        
        try:
            async for msg in self.bus.consumer:
                if not self._running:
                    break
                    
                data = msg.value
                cam_id = data.get("id")
                if cam_id:
                    self.nodes[cam_id] = data
                    self._update_live_graph()
                    logger.info(f"Twin Graph updated with new state for {cam_id}")
                    
        except asyncio.CancelledError:
            logger.info("Digital Twin shutting down...")
        finally:
            await self.bus.close()

    def get_city_state(self) -> Dict[str, Any]:
        """Provides the current digital twin state to the Dashboard API."""
        return {
            "node_count": len(self.nodes),
            "graph_available": self.live_graph is not None,
            "latest_nodes": self.nodes
        }

async def start_digital_twin_service():
    """Entry point for the cloud service."""
    manager = DigitalTwinStateManager()
    await manager.initialize()
    await manager.consume_loop()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # This prevents the script from accidentally hanging a user's local machine
    print("Digital Twin State Manager is designed for Cloud execution.")
    print("To run locally for testing, invoke: asyncio.run(start_digital_twin_service())")
