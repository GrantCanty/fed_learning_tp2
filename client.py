import torch
import torch.nn as nn
import flwr
import json
import config

from flwr.common import (
    GetPropertiesIns, GetPropertiesRes,
    GetParametersIns, GetParametersRes,
    Parameters, FitRes, FitIns,
    EvaluateIns, EvaluateRes, Code, Status,
    ndarrays_to_parameters, parameters_to_ndarrays
)

class CustomClient(flwr.client.Client):
    def __init__(self, cid, model: torch.nn.Module, train_loader, test_loader, device: torch.device) -> None:
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.LEARNING_RATE)

        # Initialize local control variate ck to zeros
        self.ck = [torch.zeros_like(p.data) for p in self.model.parameters()]

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

    def fit(self, instruction: FitIns) -> FitRes:
        # Set model parameters
        self.model.set_model_parameters(parameters_to_ndarrays(instruction.parameters))

        # Decode global control variate 'c' from config
        c_json = instruction.config.get("global_control", None)
        if c_json is None:
            raise ValueError("Missing global_control in FitIns.config")

        c = [torch.tensor(arr, device=self.device) for arr in json.loads(c_json)]

        self.model.train()
        old_params = [p.clone().detach() for p in self.model.parameters()]
        T = 0  # total number of steps

        for epoch in range(config.EPOCHS):
            for x, y in self.train_loader:
                T += 1
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()

                # Apply SCAFFOLD correction: grad += ck - c
                with torch.no_grad():
                    for i, p in enumerate(self.model.parameters()):
                        if p.grad is not None:
                            p.grad += self.ck[i] - c[i]

                self.optimizer.step()

        # Get updated model parameters
        new_params = [p.clone().detach() for p in self.model.parameters()]

        # Compute accuracy on training set for reporting
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
        acc = correct / total

        # Update ck using: ck = ck - c + (1 / ηT)(w - w_new)
        eta = config.LEARNING_RATE
        for i in range(len(self.ck)):
            delta = (old_params[i] - new_params[i]) / (eta * T)
            self.ck[i] = self.ck[i] - c[i] + delta

        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters([p.cpu().numpy() for p in new_params]),
            num_examples=len(self.train_loader.dataset),
            metrics={
                "loss": loss.item(),
                "accuracy": acc,
                "ck": json.dumps([p.cpu().numpy().tolist() for p in self.ck])
            }
        )

    def evaluate(self, instruction: EvaluateIns) -> EvaluateRes:
        self.model.set_model_parameters(parameters_to_ndarrays(instruction.parameters))
        self.model.eval()
        loss_total, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss_total += loss.item() * x.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)

        avg_loss = loss_total / total
        accuracy = correct / total

        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=avg_loss,
            num_examples=total,
            metrics={"accuracy": accuracy}
        )

    def to_client(self) -> flwr.client.Client:
        return self
