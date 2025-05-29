from pathlib import Path
import json
import flwr as fl
from flwr.server import ServerConfig, Server
from flwr.server.history import History
import scaffold_v2
import custom_client_manager  # Your custom client manager
from generate_data import generate_distributed_datasets
from config import NUM_CLIENTS, ALPHA_DIRICHLET, SAVE_PATH, NUM_ROUNDS
import argparse
import time


def main():
    # take arg for .json output name
    parser = argparse.ArgumentParser(description="Run Flower Server")
    parser.add_argument("--output", type=str, default=f"{time.time()}", 
                       help="Server address")
    args = parser.parse_args()    

    
    # create dataset that will be used
    generate_distributed_datasets(NUM_CLIENTS, ALPHA_DIRICHLET, SAVE_PATH)
    
    # 1. Define server address
    server_address = "0.0.0.0:8080"  # Listen on all interfaces on port 8080

    # 2. Define federated learning hyperparameters
    num_rounds = NUM_ROUNDS  # As specified in the assignment
    # min_available_clients = 2  # Minimum number of clients required
    
    print(f"Starting Flower server on {server_address}")
    print(f"Number of rounds: {num_rounds}")
    print(f"Waiting for {NUM_CLIENTS} clients to connect...")

    # 3. Instantiate ClientManager 
    client_manager = custom_client_manager.CustomClientManager()
    
    # 4. Configure the strategy with proper minimums
    strategy = scaffold_v2.ScaffoldStrategy(
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS
    )
    
    # 5. Configure server
    config = ServerConfig(num_rounds=num_rounds)

    # 6. Start the Flower server
    try:
        # Start server and wait for completion
        history = fl.server.start_server(
            server_address=server_address,
            config=config,
            strategy=strategy,
            client_manager=client_manager
        )

        print("Training completed! Saving results...")

        # 7. Extract history info
        losses_distributed = history.losses_distributed
        metrics_distributed_fit = history.metrics_distributed_fit
        metrics_distributed = history.metrics_distributed

        # 8. Save results as JSON
        results = {
            "losses_distributed": losses_distributed,
            "metrics_distributed_fit": metrics_distributed_fit,
            "metrics_distributed": metrics_distributed,
        }
        
        save_path = Path(f"fl_history_{args.output}.json")
        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Training history saved to {save_path}")
        
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except Exception as e:
        print(f"Error running server: {e}")


if __name__ == "__main__":
    main()