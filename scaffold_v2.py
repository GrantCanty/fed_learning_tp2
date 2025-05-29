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
import config
import json


class ScaffoldStrategy(Strategy):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = config.NUM_CLIENTS,
        min_evaluate_clients: int = config.NUM_CLIENTS,
        min_available_clients: int = config.NUM_CLIENTS,
    ):
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.c = None
        self.initial_parameters = None

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize global model parameters."""
        # Wait for at least min_available_clients to be available
        print(f"Waiting for {self.min_available_clients} clients before initializing parameters...")
        client_manager.wait_for(self.min_available_clients, timeout=None)
        
        # Let Flower handle the initialization automatically
        # We'll initialize self.c in configure_fit when we first get the parameters
        print("Letting Flower handle parameter initialization...")
        return None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # Initialize c if it's None (shouldn't happen after initialize_parameters, but safety check)
        if self.c is None:
            weights = parameters_to_ndarrays(parameters)
            self.c = [np.zeros_like(w) for w in weights]
            print("Initialized global control variate 'c' with zeros in configure_fit")
        
        # Sample clients
        sample_size = max(
            int(client_manager.num_available() * self.fraction_fit), self.min_fit_clients
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=self.min_fit_clients
        )
        
        print(f"Round {server_round}: {len(clients)} clients sampled for training")
        
        # Create fit instruction for each client
        config = {
            "global_control": json.dumps([arr.tolist() for arr in self.c])
        }
        fit_ins = FitIns(parameters, config)
        
        # Return client/fit_ins pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], 
                          failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Enhanced aggregate_fit with monitoring and safety checks."""
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

        # Initialize c if it's still None (safety check)
        if self.c is None:
            sample_weights = weights_results[0][0]
            self.c = [np.zeros_like(w) for w in sample_weights]
            print("Initialized global control variate 'c' with zeros in aggregate_fit")
        
        # Monitor control variate magnitudes BEFORE update
        if self.c is not None:
            c_norms = [np.linalg.norm(c_layer) for c_layer in self.c]
            max_c_norm = max(c_norms) if c_norms else 0.0
            print(f"Round {server_round}: Max control variate norm (before): {max_c_norm:.4f}")
            
            # Reset if control variates are too large (very aggressive threshold)
            if max_c_norm > 0.1:  # Extremely low threshold
                print(f"Round {server_round}: Resetting control variates due to explosion (norm: {max_c_norm:.6f})")
                sample_weights = weights_results[0][0]
                self.c = [np.zeros_like(w) for w in sample_weights]
        
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
        
        # Check for NaNs or Infs in aggregated weights
        has_nan = any(np.isnan(w).any() or np.isinf(w).any() for w in aggregated_weights)
        if has_nan:
            print(f"Round {server_round}: ERROR - NaN or Inf detected in aggregated weights!")
            return None, {}
        
        # Monitor parameter magnitudes
        param_norms = [np.linalg.norm(w) for w in aggregated_weights]
        max_param_norm = max(param_norms) if param_norms else 0.0
        print(f"Round {server_round}: Max parameter norm: {max_param_norm:.4f}")
        
        # Aggregate custom metrics and control variates
        metrics_aggregated = {}
        cks = []
        for _, fit_res in results:
            for key, value in fit_res.metrics.items():
                if key not in metrics_aggregated:
                    metrics_aggregated[key] = [value]
                else:
                    metrics_aggregated[key].append(value)
            
            if 'ck' in fit_res.metrics:
                try:
                    ck = json.loads(fit_res.metrics['ck'])
                    ck_arrays = [np.array(layer) for layer in ck]
                    
                    # Check for NaNs in client control variates
                    has_ck_nan = any(np.isnan(arr).any() or np.isinf(arr).any() for arr in ck_arrays)
                    if not has_ck_nan:
                        cks.append(ck_arrays)
                    else:
                        print(f"Round {server_round}: WARNING - NaN/Inf in client control variate, skipping")
                        
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Round {server_round}: Error parsing client control variate: {e}")

        # Update global control variate with bounds
        if cks:
            old_c_norms = [np.linalg.norm(c_layer) for c_layer in self.c]
            self.c = update_global_control_variate(self.c, cks, clip_norm=0.01, momentum=0.95, max_change_ratio=0.1)
            
            # Monitor control variate magnitudes AFTER update
            new_c_norms = [np.linalg.norm(c_layer) for c_layer in self.c]
            max_new_c_norm = max(new_c_norms) if new_c_norms else 0.0
            print(f"Round {server_round}: Max control variate norm (after): {max_new_c_norm:.4f}")
        
        # Average custom metrics (excluding 'ck')
        metrics_avg = {}
        for key, values in metrics_aggregated.items():
            if key != 'ck':
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
    
    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate the current global model parameters."""
        print(f"Round {server_round}: Evaluating global model")
        return None


def update_global_control_variate(c_global: List[np.ndarray], c_clients: List[List[np.ndarray]], 
                                clip_norm: float = 0.01, momentum: float = 0.95, 
                                max_change_ratio: float = 0.1) -> List[np.ndarray]:
    """Ultra-conservative control variate update with multiple safety mechanisms."""
    num_clients = len(c_clients)
    updated_c = []

    for i in range(len(c_global)):
        # Calculate individual client differences
        client_diffs = [(ck[i] - c_global[i]) for ck in c_clients]
        
        # Clip each client difference very aggressively
        clipped_diffs = []
        for diff in client_diffs:
            diff_norm = np.linalg.norm(diff)
            if diff_norm > clip_norm:
                diff = diff * (clip_norm / diff_norm)
            clipped_diffs.append(diff)
        
        # Average the clipped differences
        avg_diff = sum(clipped_diffs) / num_clients
        
        # Limit the magnitude of change relative to current control variate
        current_norm = np.linalg.norm(c_global[i])
        if current_norm > 0:
            change_norm = np.linalg.norm(avg_diff)
            max_allowed_change = current_norm * max_change_ratio
            if change_norm > max_allowed_change:
                avg_diff = avg_diff * (max_allowed_change / change_norm)
        
        # Apply very high momentum for stability
        new_c_i = momentum * c_global[i] + (1 - momentum) * (c_global[i] + avg_diff)
        
        # Ultra-aggressive final clipping
        c_norm = np.linalg.norm(new_c_i)
        if c_norm > clip_norm * 5:  # Very tight bound
            new_c_i = new_c_i * (clip_norm * 5 / c_norm)
        
        # Check for any suspicious values
        if np.isnan(new_c_i).any() or np.isinf(new_c_i).any() or c_norm > 1.0:
            print(f"WARNING: Suspicious control variate in layer {i} (norm: {c_norm:.6f}), resetting")
            new_c_i = np.zeros_like(c_global[i])
            
        updated_c.append(new_c_i)

    return updated_c