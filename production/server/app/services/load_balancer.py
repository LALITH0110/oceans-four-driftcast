"""
Load balancing strategies for task distribution
"""
from typing import List, Dict, Any
from abc import ABC, abstractmethod
import random
import time
import logging

logger = logging.getLogger(__name__)

class LoadBalancingStrategy(ABC):
    @abstractmethod
    def select_client(self, clients: List[str], task: Dict[str, Any]) -> str:
        pass

class RoundRobinStrategy(LoadBalancingStrategy):
    def __init__(self):
        self.current_index = 0
    
    def select_client(self, clients: List[str], task: Dict[str, Any]) -> str:
        if not clients:
            raise ValueError("No clients available")
        
        client = clients[self.current_index]
        self.current_index = (self.current_index + 1) % len(clients)
        return client

class RandomStrategy(LoadBalancingStrategy):
    def select_client(self, clients: List[str], task: Dict[str, Any]) -> str:
        if not clients:
            raise ValueError("No clients available")
        return random.choice(clients)

class WeightedStrategy(LoadBalancingStrategy):
    def __init__(self):
        self.client_weights: Dict[str, float] = {}
    
    def update_client_weight(self, client_id: str, weight: float):
        """Update client weight based on performance"""
        self.client_weights[client_id] = weight
    
    def select_client(self, clients: List[str], task: Dict[str, Any]) -> str:
        if not clients:
            raise ValueError("No clients available")
        
        # Use weights if available, otherwise equal probability
        weights = [self.client_weights.get(client, 1.0) for client in clients]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return random.choice(clients)
        
        # Weighted random selection
        r = random.uniform(0, total_weight)
        cumulative = 0
        
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return clients[i]
        
        return clients[-1]  # Fallback

class LoadBalancer:
    def __init__(self, strategy: str = "round_robin"):
        self.strategies = {
            "round_robin": RoundRobinStrategy(),
            "random": RandomStrategy(),
            "weighted": WeightedStrategy()
        }
        self.current_strategy = strategy
        self.client_metrics: Dict[str, Dict[str, Any]] = {}
    
    def select_client(self, clients: List[str], task: Dict[str, Any]) -> str:
        """Select best client for task using current strategy"""
        try:
            strategy = self.strategies.get(
                self.current_strategy, 
                self.strategies["round_robin"]
            )
            return strategy.select_client(clients, task)
        except Exception as e:
            logger.error(f"Error in load balancer: {e}")
            # Fallback to random selection
            return random.choice(clients) if clients else ""
    
    def update_client_metrics(self, client_id: str, metrics: Dict[str, Any]):
        """Update client performance metrics"""
        self.client_metrics[client_id] = {
            **metrics,
            "last_updated": time.time()
        }
        
        # Update weighted strategy if in use
        if isinstance(self.strategies["weighted"], WeightedStrategy):
            # Calculate weight based on performance
            success_rate = metrics.get("success_rate", 0.5)
            avg_execution_time = metrics.get("avg_execution_time", 300)  # seconds
            
            # Higher weight for better performance
            weight = success_rate * (300 / max(avg_execution_time, 1))
            self.strategies["weighted"].update_client_weight(client_id, weight)
    
    def set_strategy(self, strategy: str):
        """Change load balancing strategy"""
        if strategy in self.strategies:
            self.current_strategy = strategy
            logger.info(f"Load balancing strategy changed to: {strategy}")
        else:
            logger.warning(f"Unknown strategy: {strategy}")
    
    def get_client_metrics(self, client_id: str) -> Dict[str, Any]:
        """Get metrics for specific client"""
        return self.client_metrics.get(client_id, {})
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all clients"""
        return self.client_metrics.copy()
