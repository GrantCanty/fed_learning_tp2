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

    def fit(self,  instruction: FitIns) -> FitRes:
        self.model.set_model_parameters(parameters_to_ndarrays(instruction.parameters))

        # Deserialize and load global control variate c from server (in config)
        if "global_control" in instruction.config:
            c_np = json.loads(instruction.config["global_control"])
            self.c = [torch.tensor(arr, dtype=torch.float32, device=self.device) for arr in c_np]
        else:
            # Fallback in case it's the first round
            self.c = [torch.zeros_like(p.data) for p in self.model.parameters()]

        # Save initial model weights: wt (before training)
        wt = [p.detach().clone() for p in self.model.parameters()]

        self.model.train()
        total_batches = 0  # Needed for ck update

        for epoch in range(config.EPOCHS):
            print(f"Client {self.cid} training epoch: {epoch}")
            loss, acc, batch_count = self.model.train_epoch(
                self.train_loader, 
                self.criterion, 
                self.optimizer, 
                self.device,
                global_control=self.c,
                local_control=self.ck
            )
            total_batches += batch_count

        # Save new weights: wt+1 (after training)
        wt_plus_1 = [p.detach().clone() for p in self.model.parameters()]

        # ---------------------
        # Step 4: Update local control variate `ck`
        # ---------------------
        eta = config.LEARNING_RATE
        for i in range(len(self.ck)):
            delta_w = wt[i] - wt_plus_1[i]
            self.ck[i] = self.ck[i] - self.c[i] + (delta_w / (eta * total_batches))

        # Serialize ck to send back to server
        ck_serialized = json.dumps([tensor.cpu().numpy().tolist() for tensor in self.ck])

        new_params = ndarrays_to_parameters(self.model.get_model_parameters())
        num_examples = len(self.train_loader.dataset)

        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=new_params,
            num_examples=num_examples,
            metrics={"loss": loss, "accuracy": acc, "ck": ck_serialized}
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