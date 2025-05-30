import torch
import torch.nn as nn
import flwr
import config
import json

from flwr.common import (GetPropertiesIns, GetPropertiesRes,
GetParametersIns, GetParametersRes,
Parameters, FitRes, FitIns,
EvaluateIns, EvaluateRes, Code, Status,
ndarrays_to_parameters,
parameters_to_ndarrays)

class CustomClient(flwr.client.Client):
    def __init__(self, cid, model: torch.nn.Module, train_loader, test_loader, device: torch.device) -> None:
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.LEARNING_RATE)
        self.ck = [torch.zeros_like(p.data) for p in model.parameters()]
        self.c = [torch.zeros_like(p.data) for p in model.parameters()]

    def get_properties(self, instruction: GetPropertiesIns) -> GetPropertiesRes:
        return GetPropertiesRes(
            status=Status(code=Code.OK, message="Success"),
            properties={"framework": "PyTorch", "device": str(self.device)}
        )
    
    def get_parameters(self, instruction: GetParametersIns) -> GetParametersRes:
        parameters = ndarrays_to_parameters(self.model.get_model_parameters())
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=parameters
        )

    def train_epoch(self, train_loader, criterion, optimizer, device, global_control, local_control):
        """Custom training epoch that implements SCAFFOLD algorithm."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0
        correct_total = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # SCAFFOLD: Apply control variates to gradients
            with torch.no_grad():
                for param, c_global, c_local in zip(self.model.parameters(), global_control, local_control):
                    if param.grad is not None:
                        # Apply SCAFFOLD correction: subtract global control and add local control
                        param.grad.data = param.grad.data - c_global + c_local
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.25)
            
            optimizer.step()
            
            # Statistics
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            batch_count += 1
            correct_total += correct
            total_samples += total
            
            if batch_idx % 100 == 0:
                print(f'Client {self.cid}: Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / total
        accuracy = correct_total / total_samples

        param_norm = sum(p.norm().item() for p in self.model.parameters())
        print(f"[{self.cid}] Parameter norm after training: {param_norm}")
        
        return avg_loss, accuracy, batch_count

    def fit(self, instruction: FitIns) -> FitRes:
        print(f"Client {self.cid}: Starting training round")
        self.model.set_model_parameters(parameters_to_ndarrays(instruction.parameters))

        # Deserialize and load global control variate c from server (in config)
        if "global_control" in instruction.config:
            c_np = json.loads(instruction.config["global_control"])
            self.c = [torch.tensor(arr, dtype=torch.float32, device=self.device) for arr in c_np]
            print(f"Client {self.cid}: Loaded global control variate")
        else:
            # Fallback in case it's the first round
            self.c = [torch.zeros_like(p.data) for p in self.model.parameters()]
            print(f"Client {self.cid}: Using zero global control variate")

        # Save initial model weights: wt (before training)
        wt = [p.detach().clone() for p in self.model.parameters()]

        total_batches = 0  # Needed for ck update
        total_loss = 0.0
        total_acc = 0.0

        for epoch in range(config.EPOCHS):
            print(f"Client {self.cid}: Training epoch {epoch+1}/{config.EPOCHS}")
            
            # Use our custom SCAFFOLD training function
            loss, acc, batch_count = self.train_epoch(
                self.train_loader, 
                self.criterion, 
                self.optimizer, 
                self.device,
                global_control=self.c,
                local_control=self.ck
            )
            
            total_batches += batch_count
            total_loss += loss
            total_acc += acc
            
            print(f"Client {self.cid}: Epoch {epoch+1} - Loss: {loss:.4f}, Accuracy: {acc:.2f}%")

        # Average metrics across epochs
        avg_loss = total_loss / config.EPOCHS
        avg_acc = total_acc / config.EPOCHS

        # Save new weights: wt+1 (after training)
        wt_plus_1 = [p.detach().clone() for p in self.model.parameters()]

        # ---------------------
        # Step 4: Update local control variate `ck`
        # ---------------------
        eta = config.LEARNING_RATE
        for i in range(len(self.ck)):
            delta_w = wt[i] - wt_plus_1[i]
            self.ck[i] = self.ck[i] - self.c[i] + (delta_w / (eta * total_batches))

        print(f"Client {self.cid}: Updated local control variate")

        # Serialize ck to send back to server
        ck_serialized = json.dumps([tensor.cpu().numpy().tolist() for tensor in self.ck])

        new_params = ndarrays_to_parameters(self.model.get_model_parameters())
        num_examples = len(self.train_loader.dataset)

        print(f"Client {self.cid}: Training completed - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.2f}%")

        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=new_params,
            num_examples=num_examples,
            metrics={"loss": avg_loss, "accuracy": avg_acc, "ck": ck_serialized}
        )

    def evaluate(self, instruction: EvaluateIns) -> EvaluateRes:
        print(f"Client {self.cid}: Starting evaluation")
        self.model.set_model_parameters(parameters_to_ndarrays(instruction.parameters))
        
        # Use the existing test_epoch method if available, otherwise implement here
        try:
            loss, acc = self.model.test_epoch(self.test_loader, self.criterion, self.device)
        except AttributeError:
            # If test_epoch doesn't exist, implement evaluation here
            loss, acc = self.evaluate_model()
        
        num_examples = len(self.test_loader.dataset)
        print(f"Client {self.cid}: Evaluation completed - Loss: {loss:.4f}, Accuracy: {acc:.2f}%")

        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=loss,
            num_examples=num_examples,
            metrics={"accuracy": acc}
        )

    def evaluate_model(self):
        """Fallback evaluation method if model doesn't have test_epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item() * data.size(0)
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy

    def to_client(self) -> flwr.client.Client:
        return self