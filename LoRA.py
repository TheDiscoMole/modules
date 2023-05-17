import math
import torch

from . import convolution

class ConvNd (torch.nn.Module):
    """
    ConvNd module for LoRA tasks

    dims: number of input dimensions
    in_channels: number of input channels
    out_channels: number of output channels
    kernel_size: size of the convolutional kernel                               (optional|default: 5)
    rank: LoRA rank                                                             (optional|default: 16)
    stride: convolutional stride length                                         (optional|default: 1)
    """
    def __init__ (self, dims, in_channels, out_channels,
        kernel_size=5,
        padding="same",
        rank=16,
        stride=1
    ):
        torch.nn.Module.__init__(self)

        # validation
        assert rank < min(in_channels, out_channels), "rank must be small"

        # configs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = [kernel_size] * dims
        self.padding = padding
        self.stride = stride

        # parameters
        self.A = torch.nn.Parameter(torch.zeros(*self.kernel_size, self.out_channels, rank))
        self.B = torch.nn.Parameter(torch.zeros(*self.kernel_size, rank, self.in_channels))

        # modules
        match dims:
            case 1: self.convolution = torch.nn.functional.conv1d
            case 2: self.convolution = torch.nn.functional.conv2d
            case 3: self.convolution = torch.nn.functional.conv3d

    def forward (self, input):
        # convolution
        kernel = torch.matmul(self.A, self.B).view(self.out_channels, self.in_channels, *self.kernel_size)
        convolution = self.convolution(input, kernel, padding=self.padding, stride=self.stride)

        return convolution

class Embedding (torch.nn.Module):
    """
    Embedding module for LoRA tasks

    num_embeddings: length of the embedding
    embedding_dim: number of embedding features
    rank: LoRA rank                                                             (optional|default: 16)
    """
    def __init__ (self, num_embeddings, embedding_dim,
        rank=16
    ):
        torch.nn.Module.__init__(self)

        # validation
        assert rank < min(length, features), "rank must be less than smallest dimension"

        # configs
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.rank = rank

        # parameters
        self.A = torch.nn.Parameter(torch.zeros(num_embeddings, rank))
        self.B = torch.nn.Parameter(torch.zeros(rank, embedding_dim))

    def forward (self, idx):
        # embedding
        weight = torch.matmul(self.A, self.B)
        embedding = torch.nn.functional.embedding(idx, weight)

        return embedding

class Linear (torch.nn.Module):
    """
    Linear module for LoRA tasks

    in_features: number of input features
    out_features: number of output features
    rank: LoRA rank                                                             (optional|default: 16)
    """
    def __init__ (self, in_features, out_features,
        rank=16
    ):
        torch.nn.Module.__init__(self)

        # validation
        assert rank < min(in_features, out_features), "rank must be less than smallest dimension"

        # configs
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # parameters
        self.A = torch.nn.Parameter(torch.zeros(in_features, rank))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_features))

    def forward (self, input):
        # linear
        weight = torch.matmul(self.A, self.B)
        linear = torch.nn.functional.linear(input, weight)

        return linear

class LoRALayer (torch.nn.Module):
    """
    LoRA layer wrapper

    model: lora model reference
    module: module to lora wrap
    new_task: lambda to generate new task
    """
    def __init__ (self, model, module, new_task):
        torch.nn.Module.__init__(self)

        # task
        self.task = lambda: model.task
        self.tasks = {}
        self.new_task = new_task

        # modules
        self.module = module

    def forward (self, *inputs, **kwargs):
        # output
        output = self.module(*inputs, **kwargs)
        output = output if self.task() is None else output + self.tasks[self.task()](*inputs, **kwargs)

        return output

    def add_task (self, task):
        self.tasks[task] = self.new_task(self.module)

    def remove_task (self, task):
        del self.tasks[task]

    def parameters (self, task=None):
        yield self.module.parameters() if task is None else self.tasks[task].parameters()

    def train (self, task=None, mode=True):
        self.module.train(mode=mode) if task is None else self.tasks[task].train(mode=mode)

class LoRAModel (torch.nn.Module):
    """
    LoRA model wrapper

    model: base model
    rank: function to calculate layer rank                                      (optional|default: sqrt of smallest dimension)
    """
    def __init__ (self, model,
        rank=lambda *features: int(math.sqrt(min(*features))),
        **kwargs
    ):
        torch.nn.Module.__init__(self)

        # compile
        def compile (model):
            for name, module in model.named_children():
                compile(module)

                if isinstance(module, convolution.ConvNd):
                    new_task = lambda module: ConvNd(dims=module.dims, in_channels=module.in_channels, out_channels=module.out_channels, kernel_size=module.kernel_size, padding=module.padding, rank=rank(module.in_channels, module.out_channels), stride=module.stride)
                    module = LoRALayer(self, module, new_task)
                elif isinstance(module, torch.nn.Embedding):
                    new_task = lambda module: Embedding(num_embeddings=module.num_embeddings, embedding_dim=module.embedding_dim, rank=rank(module.num_embeddings, module.embedding_dim))
                    module = LoRALayer(self, module, new_task)
                elif isinstance(module, torch.nn.Linear):
                    new_task = lambda module: Linear(in_features=module.in_features, out_features=module.out_features, rank=rank(module.in_features, module.out_features))
                    module = LoRALayer(self, module, new_task)
                else: continue

                setattr(model, name, module)
        compile(model)

        # model
        self.model = model

    def forward (self, *inputs, task=None, **kwargs):
        # output
        self.task = task
        output = self.model(*inputs, **kwargs)
        self.task = None

        return output

    def add_task (self, task):
        for module in self.modules():
            if isinstance(module, LoRALayer): module.add_task(task)

    def remove_task (self, task):
        for module in self.modules():
            if isinstance(module, LoRALayer): module.remove_task(task)

    def parameters (self, task=None):
        for module in self.modules():
            if isinstance(module, LoRALayer): yield module.parameters(task=task)
            elif task is None: yield module.parameters()

    def train (self, task=None, mode=True):
        for module in self.modules():
            if isinstance(module, LoRALayer): module.train(task=task, mode=mode)
            elif task is None: module.train(mode=mode)

    def state_dict (model, task=None, prefix=""):
        dict = {}

        def df (module, task=None, prefix=""):
            for name,child in module.named_children():
                dict = {**dict, **df(child, task, prefix=f"{prefix}.{name}")}

        if isinstance(model, LoRALayer):
            dict = {**dict, **model.tasks[task].state_dict(prefix=f"{prefix}.{task}")}

        return dict

    def load_state_dict (model, task, state_dict, prefix=""):
        for name,child in model.named_children():
            load_state_dict(child, task, prefix=f"{prefix}.{name}")

        if isinstance(model, LoRALayer):
            module.tasks[task]._load_from_state_dict(state_dict=state_dict, prefix=f"{prefix}.{task}", local_metadata={}, strict=False, missing_keys=[], unexpected_keys=[], error_msgs=[])
