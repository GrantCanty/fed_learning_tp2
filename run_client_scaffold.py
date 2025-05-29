import argparse
import torch
import flwr as fl
#from client_scaffold_cl import CustomClient  # Your custom client implementation
from client_scaffold_v2 import CustomClient
from load import load_client_data  # Your data loading function
from model_fedprox import CustomFashionModel  # Your model implementation
import config


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Flower Client")
    parser.add_argument("--cid", type=int, required=True, help="Client ID")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080", 
                       help="Server address")
    args = parser.parse_args()

    # Extract client ID
    client_id = args.cid
    print(f"Starting client {client_id}")

    # Load client data
    train_loader, val_loader = load_client_data(client_id, config.SAVE_PATH, config.BATCH_SIZE)
    
    # Create model instance
    model = CustomFashionModel()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create custom client
    client = CustomClient(args.cid, model, train_loader, val_loader, device)
    
    # Start client
    print(f"Connecting to server at {args.server}")
    fl.client.start_client(
        server_address=args.server,
        client=client.to_client(),
        grpc_max_message_length=1024*1024*1024
    )


if __name__ == "__main__":
    main()