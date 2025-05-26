import matplotlib.pyplot as plt
import json
import os
from prettytable import PrettyTable
import argparse


class ResultsVisualizer:
    def __init__(self) -> None:
        self.results = None
    

    def load_simulation_results(self, file_name: str) -> None:
        with open(file_name, 'r') as file:
            self.results = json.load(file)
    
    
    def plot_results(self, fig_directory: str, output_name: str) -> None:
        if self.results is None:
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(fig_directory, exist_ok=True)
        
        # Plot losses_distributed
        if "losses_distributed" in self.results:
            plt.figure(figsize=(10, 6))
            rounds = [item[0] for item in self.results["losses_distributed"]]
            values = [item[1] for item in self.results["losses_distributed"]]
            
            plt.plot(rounds, values, marker='o', linewidth=2, markersize=6)
            plt.title(f'Distributed Losses Over Training Rounds  - {output_name}')
            plt.xlabel('Training Round')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.tight_layout()
            
            filename = os.path.join(fig_directory, f'losses_distributed_plot_{output_name}.png')
            plt.savefig(filename)
            plt.close()
        
        # Plot metrics from metrics_distributed_fit
        if "metrics_distributed_fit" in self.results:
            for metric_name, metric_data in self.results["metrics_distributed_fit"].items():
                if metric_name != "round":  # Skip the round metric as it's redundant
                    plt.figure(figsize=(10, 6))
                    rounds = [item[0] for item in metric_data]
                    values = [item[1] for item in metric_data]
                    
                    plt.plot(rounds, values, marker='o', linewidth=2, markersize=6)
                    plt.title(f'{metric_name.replace("_", " ").title()} (Fit) Over Training Rounds  - {output_name}')
                    plt.xlabel('Training Round')
                    plt.ylabel(metric_name.replace("_", " ").title())
                    plt.grid(True)
                    plt.tight_layout()
                    
                    filename = os.path.join(fig_directory, f'{metric_name}_fit_plot_{output_name}.png')
                    plt.savefig(filename)
                    plt.close()
        
        # Plot metrics from metrics_distributed
        if "metrics_distributed" in self.results:
            for metric_name, metric_data in self.results["metrics_distributed"].items():
                if metric_name != "round":  # Skip the round metric as it's redundant
                    plt.figure(figsize=(10, 6))
                    rounds = [item[0] for item in metric_data]
                    values = [item[1] for item in metric_data]
                    
                    plt.plot(rounds, values, marker='o', linewidth=2, markersize=6)
                    plt.title(f'{metric_name.replace("_", " ").title()} (Evaluation) Over Training Rounds - {output_name}')
                    plt.xlabel('Training Round')
                    plt.ylabel(metric_name.replace("_", " ").title())
                    plt.grid(True)
                    plt.tight_layout()
                    
                    filename = os.path.join(fig_directory, f'{metric_name}_eval_plot_{output_name}.png')
                    plt.savefig(filename)
                    plt.close()
    
    def print_results_table(self) -> None:
        if self.results is None:
            return
        
        # Get all rounds from any available metric
        rounds = set()
        
        # Collect rounds from losses_distributed
        if "losses_distributed" in self.results:
            rounds.update([item[0] for item in self.results["losses_distributed"]])
        
        # Collect rounds from metrics_distributed_fit
        if "metrics_distributed_fit" in self.results:
            for metric_data in self.results["metrics_distributed_fit"].values():
                rounds.update([item[0] for item in metric_data])
        
        # Collect rounds from metrics_distributed
        if "metrics_distributed" in self.results:
            for metric_data in self.results["metrics_distributed"].values():
                rounds.update([item[0] for item in metric_data])
        
        rounds = sorted(list(rounds))
        
        # Create the table
        table = PrettyTable()
        
        # Set up columns
        columns = ['Round']
        
        # Add column for distributed losses
        if "losses_distributed" in self.results:
            columns.append('Loss (Distributed)')
        
        # Add columns for fit metrics
        if "metrics_distributed_fit" in self.results:
            for metric_name in self.results["metrics_distributed_fit"].keys():
                if metric_name != "round":
                    columns.append(f'{metric_name.title()} (Fit)')
        
        # Add columns for evaluation metrics
        if "metrics_distributed" in self.results:
            for metric_name in self.results["metrics_distributed"].keys():
                if metric_name != "round":
                    columns.append(f'{metric_name.title()} (Eval)')
        
        table.field_names = columns
        
        # Add rows for each round
        for round_num in rounds:
            row = [round_num]
            
            # Add distributed loss value
            if "losses_distributed" in self.results:
                loss_value = None
                for item in self.results["losses_distributed"]:
                    if item[0] == round_num:
                        loss_value = item[1]
                        break
                if loss_value is not None:
                    row.append(f"{loss_value:.4f}")
                else:
                    row.append("N/A")
            
            # Add fit metric values
            if "metrics_distributed_fit" in self.results:
                for metric_name, metric_data in self.results["metrics_distributed_fit"].items():
                    if metric_name != "round":
                        metric_value = None
                        for item in metric_data:
                            if item[0] == round_num:
                                metric_value = item[1]
                                break
                        if metric_value is not None:
                            row.append(f"{metric_value:.4f}")
                        else:
                            row.append("N/A")
            
            # Add evaluation metric values
            if "metrics_distributed" in self.results:
                for metric_name, metric_data in self.results["metrics_distributed"].items():
                    if metric_name != "round":
                        metric_value = None
                        for item in metric_data:
                            if item[0] == round_num:
                                metric_value = item[1]
                                break
                        if metric_value is not None:
                            row.append(f"{metric_value:.4f}")
                        else:
                            row.append("N/A")
            
            table.add_row(row)
        
        print(table)


def main():
    parser = argparse.ArgumentParser(description="Run Flower Client")
    parser.add_argument("--input", type=str, required=True, help="name of input json file")
    parser.add_argument("--output", type=str, required=True, help="Name of files for plots")
    args = parser.parse_args()
    
    # Create visualizer instance
    visualizer = ResultsVisualizer()
    
    # Load simulation results
    visualizer.load_simulation_results(f"fl_history_{args.input}.json")
    
    # Print results table
    visualizer.print_results_table()
    
    # Plot results
    visualizer.plot_results("plots", args.output)


if __name__ == "__main__":
    main()