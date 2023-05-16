import torch

class Layer ():
    def __init__ (self):
        # modules
        self.tasks = {}

class EmbeddingTask (torch.nn.Module):
    def __init__ (self, length, features,
        rank=16
    ):
        torch.nn.Module.__init__(self)

        # validation
        assert (rank > 0) and (rank < min(in_features, out_features)), "rank must be between 0 and lowest number of features"

        # parameters
        self.A = torch.nn.parameters.Parameter(torch.zeros(length, rank))
        self.B = torch.nn.parameters.Parameter(torch.zeros(rank, features))

    def forward (self, idx):
        # embedding
        weight = torch.matmul(self.A, self.B)
        embedding = torch.nn.functional.embedding(idx, weight)

        return embedding

    def reset_parameters (self):
        self.A = torch.nn.parameters.Parameter(torch.zeros(length, rank))
        self.B = torch.nn.parameters.Parameter(torch.zeros(rank, features))

class LinearTask (torch.nn.Module):
    def __init__ (self, in_features, out_features,
        rank=16
    ):
        torch.nn.Module.__init__(self)

        # validation
        assert (rank > 0) and (rank < min(in_features, out_features)), "rank must be between 0 and lowest number of features"

        # parameters
        self.A = torch.nn.parameters.Parameter(torch.zeros(in_features, rank))
        self.B = torch.nn.parameters.Parameter(torch.zeros(rank, out_features))

    def forward (self, input):
        # linear
        weight = torch.matmul(self.A, self.B)
        linear = torch.nn.functional(input, weight)

        return linear

    def reset_parameters (self):
        self.A = torch.nn.parameters.Parameter(torch.zeros(in_features, rank))
        self.B = torch.nn.parameters.Parameter(torch.zeros(rank, out_features))

def reset_parameters (model, task):
    for module in model.modules():
        if isinstance(module, Layer):
            module.tasks[task].reset_parameters()

def parameters (model, task):
    for module in model.modules():
        if isinstance(module, Layer):
            yield module.tasks[task].parameters()

def train (model, task, mode=True):
    for module in model.modules():
        if isinstance(module, Layer):
            module.tasks[task].train(mode=mode)

def state_dict (model, task, prefix=""):
    dict = {**state_dict(child, task, prefix=f"{prefix}.{name}") for name,child in model.named_children()}

    if isinstance(model, Layer):
        dict = {**dict, **module.tasks[task].state_dict(prefix=f"{prefix}.{task}")}

    return dict

def load_state_dict (model, task, state_dict, prefix=""):
    for name,child in model.named_children():
        load_state_dict(child, task, prefix=f"{prefix}.{name}")

    if isinstance(model, Layer):
        module.tasks[task]._load_from_state_dict(state_dict=state_dict, prefix=f"{prefix}.{task}", local_metadata={}, strict=False, missing_keys=[], unexpected_keys=[], error_msgs=[])
