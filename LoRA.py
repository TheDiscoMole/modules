import torch

class Layer ():
    def __init__ (self):
        # modules
        self.tasks = {}

class ConvNdTask (torch.nn.Module):
    def __init__ (self, in_channels, out_channels,
        kernel_size=5,
        rank=16,
        **kwargs
    ):
        torch.nn.Module.__init__(self)

        # validation
        assert (rank > 0) and (rank < min(in_channels, out_channels)), "rank must be between 0 and smallest dimension"

        # configs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.rank = rank

        # parameters
        self.A = torch.nn.parameters.Parameter(torch.zeros(in_channels * kernel_size, rank * kernel_size))
        self.B = torch.nn.parameters.Parameter(torch.zeros(rank * kernel_size, out_channels * kernel_size))

        # modules
        match dims:
            case 1: self.convolution = torch.nn.functional.conv1d
            case 2: self.convolution = torch.nn.functional.conv2d
            case 3: self.convolution = torch.nn.functional.conv3d

    def forward (self, input):
        # convolution
        kernel = torch.matmul(self.A, self.B)
        convolution = self.convolution(input, kernel)

        return convolution

    def reset_parameters (self):
        self.A = torch.nn.parameters.Parameter(torch.zeros(self.in_channels * self.kernel_size, self.rank * self.kernel_size))
        self.B = torch.nn.parameters.Parameter(torch.zeros(self.rank * self.kernel_size, self.out_channels * self.kernel_size))

class EmbeddingTask (torch.nn.Module):
    def __init__ (self, length, features,
        rank=16
    ):
        torch.nn.Module.__init__(self)

        # validation
        assert (rank > 0) and (rank < min(length, features)), "rank must be between 0 and smallest dimension"

        # configs
        self.length = length
        self.features = features
        self.rank = rank

        # parameters
        self.A = torch.nn.parameters.Parameter(torch.zeros(length, rank))
        self.B = torch.nn.parameters.Parameter(torch.zeros(rank, features))

    def forward (self, idx):
        # embedding
        weight = torch.matmul(self.A, self.B)
        embedding = torch.nn.functional.embedding(idx, weight)

        return embedding

    def reset_parameters (self):
        self.A = torch.nn.parameters.Parameter(torch.zeros(self.length, self.rank))
        self.B = torch.nn.parameters.Parameter(torch.zeros(self.rank, self.features))

class LinearTask (torch.nn.Module):
    def __init__ (self, in_features, out_features,
        rank=16
    ):
        torch.nn.Module.__init__(self)

        # validation
        assert (rank > 0) and (rank < min(in_features, out_features)), "rank must be between 0 and smallest dimension"

        # configs
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # parameters
        self.A = torch.nn.parameters.Parameter(torch.zeros(in_features, rank))
        self.B = torch.nn.parameters.Parameter(torch.zeros(rank, out_features))

    def forward (self, input):
        # linear
        weight = torch.matmul(self.A, self.B)
        linear = torch.nn.functional(input, weight)

        return linear

    def reset_parameters (self):
        self.A = torch.nn.parameters.Parameter(torch.zeros(self.in_features, self.rank))
        self.B = torch.nn.parameters.Parameter(torch.zeros(self.rank, self.out_features))

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
