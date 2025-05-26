from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy


class FedAvgStrategy(Strategy):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
    ):
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize global model parameters."""
        # Wait for at least min_available_clients to be available
        print(f"Waiting for {self.min_available_clients} clients before initializing parameters...")
        client_manager.wait_for(self.min_available_clients, timeout=None)
        
        # Let Flower handle the initialization
        return None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # Sample clients
        sample_size = max(
            int(client_manager.num_available() * self.fraction_fit), self.min_fit_clients
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=self.min_fit_clients
        )
        
        print(f"Round {server_round}: {len(clients)} clients sampled for training")
        
        # Create fit instruction for each client
        config = {}
        fit_ins = FitIns(parameters, config)
        
        # Return client/fit_ins pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results."""
        if not results:
            print(f"Round {server_round}: No results to aggregate")
            return None, {}
        
        if failures:
            print(f"Round {server_round}: {len(failures)} clients failed during training")
        
        # Extract weights and num_examples
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) 
            for _, fit_res in results
        ]
        
        # Perform weighted averaging
        total_examples = sum([num_examples for _, num_examples in weights_results])
        weighted_weights = [
            [layer * num_examples / total_examples for layer in weights] 
            for weights, num_examples in weights_results
        ]
        
        # Aggregate weights
        aggregated_weights = [
            np.sum([weights[i] for weights in weighted_weights], axis=0)
            for i in range(len(weighted_weights[0]))
        ]
        
        # Aggregate custom metrics if they exist
        metrics_aggregated = {}
        for _, fit_res in results:
            for key, value in fit_res.metrics.items():
                if key not in metrics_aggregated:
                    metrics_aggregated[key] = [value]
                else:
                    metrics_aggregated[key].append(value)
        
        # Average custom metrics
        metrics_avg = {}
        for key, values in metrics_aggregated.items():
            metrics_avg[key] = float(np.mean(values))
        
        metrics_avg["round"] = server_round
        print(f"Round {server_round}: Training aggregated from {len(results)} clients")
        
        return ndarrays_to_parameters(aggregated_weights), metrics_avg

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Sample clients
        sample_size = max(
            int(client_manager.num_available() * self.fraction_evaluate),
            self.min_evaluate_clients,
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=self.min_evaluate_clients
        )
        
        print(f"Round {server_round}: {len(clients)} clients sampled for evaluation")
        
        # Create evaluate instruction for each client
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)
        
        # Return client/evaluate_ins pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        if not results:
            print(f"Round {server_round}: No evaluation results to aggregate")
            return None, {}
        
        if failures:
            print(f"Round {server_round}: {len(failures)} clients failed during evaluation")
        
        # Extract loss and num_examples
        loss_results = [
            (evaluate_res.loss, evaluate_res.num_examples) 
            for _, evaluate_res in results
        ]
        
        # Perform weighted averaging of loss
        total_examples = sum([num_examples for _, num_examples in loss_results])
        weighted_loss = sum(
            [loss * num_examples / total_examples for loss, num_examples in loss_results]
        )
        
        # Aggregate custom metrics if they exist
        metrics_aggregated = {}
        for _, evaluate_res in results:
            for key, value in evaluate_res.metrics.items():
                if key not in metrics_aggregated:
                    metrics_aggregated[key] = [value]
                else:
                    metrics_aggregated[key].append(value)
        
        # Average custom metrics
        metrics_avg = {}
        for key, values in metrics_aggregated.items():
            metrics_avg[key] = float(np.mean(values))
        
        metrics_avg["round"] = server_round
        print(f"Round {server_round}: Evaluation aggregated from {len(results)} clients")
        
        return weighted_loss, metrics_avg
    
    # Add required evaluation method to match modern API
    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate the current global model parameters.
        
        This is a new method required in recent Flower versions.
        """
        print(f"Round {server_round}: Evaluating global model")
        return None