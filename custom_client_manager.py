import threading
import time
from typing import Dict, List, Optional, Union
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager


class CustomClientManager(ClientManager):
    def __init__(self):
        self.clients: Dict[str, ClientProxy] = {}
        self.lock = threading.Lock()
        # Condition to notify when new clients connect
        self.cv = threading.Condition(self.lock)
        
    def num_available(self) -> int:
        """Return the number of available clients."""
        with self.lock:
            return len(self.clients)
        
    def register(self, client: ClientProxy) -> bool:
        """Register a new client."""
        with self.lock:
            self.clients[client.cid] = client
            print(f"Client {client.cid} registered. Total clients: {len(self.clients)}")
            # Notify waiting threads that a new client is available
            self.cv.notify_all()
            return True
        
    def unregister(self, client: ClientProxy) -> None:
        """Unregister a client."""
        with self.lock:
            if client.cid in self.clients:
                del self.clients[client.cid]
                print(f"Client {client.cid} unregistered. Total clients: {len(self.clients)}")
        
    def all(self) -> Dict[str, ClientProxy]:
        """Return all registered clients."""
        with self.lock:
            return self.clients.copy()
        
    def wait_for(self, num_clients: int, timeout: Optional[float] = None) -> bool:
        """Wait until at least `num_clients` are available."""
        with self.cv:
            # Calculate end time if timeout is provided
            end_time = time.time() + timeout if timeout is not None else None
            
            # Wait until we have enough clients or timeout
            while len(self.clients) < num_clients:
                # Check if timeout has expired
                if end_time is not None and time.time() > end_time:
                    return False
                
                # Calculate remaining time if timeout is provided
                wait_time = end_time - time.time() if end_time is not None else None
                
                print(f"Waiting for clients: {len(self.clients)}/{num_clients}")
                
                # Wait for notification with timeout
                if wait_time is not None:
                    if not self.cv.wait(timeout=wait_time):
                        return False
                else:
                    # Wait indefinitely
                    self.cv.wait()
            
            print(f"Required number of clients connected: {len(self.clients)}/{num_clients}")
            return True
        
    def sample(self, num_clients: int, min_num_clients: Optional[int] = None,
               criterion: Optional[Union[Dict, callable]] = None) -> List[ClientProxy]:
        """Sample a number of clients from available clients."""
        with self.lock:
            # Determine minimum number of clients to sample
            min_clients = min_num_clients if min_num_clients is not None else num_clients
            
            # Check if we have enough clients
            if len(self.clients) < min_clients:
                return []
            
            # Sample clients
            available_clients = list(self.clients.values())
            if num_clients >= len(available_clients):
                return available_clients
            
            # Random sampling
            import random
            return random.sample(available_clients, num_clients)