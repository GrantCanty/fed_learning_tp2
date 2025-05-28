import torch
import torch.nn as nn
import flwr
import config
import copy

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

    def fit(self,  instruction: FitIns) -> FitRes:
        self.model.set_model_parameters(parameters_to_ndarrays(instruction.parameters))
        model_copy = copy.deepcopy(self.model)
        self.model.train()
        for epoch in range(config.EPOCHS):
            print(f"client evaluating epoch: {epoch}")
            loss, acc = self.model.train_epoch(self.train_loader, self.criterion, self.optimizer, self.device, model_copy, config.PROXIMAL_TERM)
            new_params = ndarrays_to_parameters(self.model.get_model_parameters())
            num_examples = len(self.train_loader.dataset)

        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=new_params,
            num_examples=num_examples,
            metrics={"loss": loss, "accuracy": acc}
        )

    def evaluate(self, instruction: EvaluateIns) -> EvaluateRes:
        print("client evaluating model >>>>")
        self.model.set_model_parameters(parameters_to_ndarrays(instruction.parameters))
        loss, acc = self.model.test_epoch(self.test_loader, self.criterion, self.device)
        num_examples = len(self.test_loader.dataset)

        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=loss,
            num_examples=num_examples,
            metrics={"accuracy": acc}
        )

    def to_client(self) -> flwr.client.Client:
        return self