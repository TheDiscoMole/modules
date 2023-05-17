import collections
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

        self.kernel = (self.A @ self.B).view(out_channels, in_channels, *self.kernel_size)

        # modules
        match dims:
            case 1: self.convolution = torch.nn.functional.conv1d
            case 2: self.convolution = torch.nn.functional.conv2d
            case 3: self.convolution = torch.nn.functional.conv3d

    def forward (self, input):
        return self.convolution(input, self.kernel, padding=self.padding, stride=self.stride)

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

        self.weight = self.A @ self.B

    def forward (self, idx):
        return torch.nn.functional.embedding(idx, self.weight)

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

        self.weight = self.A @ self.B

    def forward (self, input):
        return torch.nn.functional.linear(input, self.weight)

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
        return self.module.parameters() if task is None else self.tasks[task].parameters()

    def train (self, task=None, mode=True):
        self.module.train(mode=mode) if task is None else self.tasks[task].train(mode=mode)

    def state_dict (self, task=None, prefix="", **kwargs):
        if task is None:
            return self.module.state_dict(prefix=f"{prefix}module.", **kwargs)
        else:
            return self.tasks[task].state_dict(prefix=f"{prefix}module.", **kwargs)

    def load_state_dict (self, state_dict, task=None, prefix="", **kwargs):
        if task is None:
            self.module.load_state_dict(state_dict, **kwargs)
        else:
            prefix = f"{prefix}module."
            state_dict = {k[len(prefix)]:v for k,v in state_dict.items() if prefix in k}
            self.tasks[task].load_state_dict(state_dict, **kwargs)

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
                if isinstance(module, convolution.ConvNd):
                    new_task = lambda module: ConvNd(dims=module.dims, in_channels=module.in_channels, out_channels=module.out_channels, kernel_size=module.kernel_size, padding=module.padding, rank=rank(module.in_channels, module.out_channels), stride=module.stride)
                    setattr(model, name, LoRALayer(self, module, new_task))
                elif isinstance(module, torch.nn.Embedding):
                    new_task = lambda module: Embedding(num_embeddings=module.num_embeddings, embedding_dim=module.embedding_dim, rank=rank(module.num_embeddings, module.embedding_dim))
                    setattr(model, name, LoRALayer(self, module, new_task))
                elif isinstance(module, torch.nn.Linear):
                    new_task = lambda module: Linear(in_features=module.in_features, out_features=module.out_features, rank=rank(module.in_features, module.out_features))
                    setattr(model, name, LoRALayer(self, module, new_task))
                else: compile(module)
        compile(model)

        # model
        self.model = model

    def forward (self, *inputs, task=None, **kwargs):
        # output
        self.task = task
        output = self.model(*inputs, **kwargs)
        self.task = None

        return output

    def lora_layers (self):
        for module in self.modules():
            if isinstance(module, LoRALayer): yield module

    def add_task (self, task):
        for module in self.lora_layers(): module.add_task(task)

    def remove_task (self, task):
        for module in self.lora_layers(): module.remove_task(task)

    def parameters (self, task=None):
        if task is None:
            for param in self.model.parameters(): yield param
            return

        for module in self.lora_layers():
            for param in module.parameters(task=task): yield param

    def eval (self, task=None):
        self.train(task=task, mode=False)

    def train (self, task=None, mode=True):
        self.training = mode

        if task is None: return self.model.train(mode=mode)
        for module in self.lora_layers(): module.train(task=task, mode=mode)

    def state_dict (self, task=None, prefix="", **kwargs):
        if task is None: return self.model.state_dict(destination=collections.OrderedDict(), prefix=prefix)

        def df_task_state_dict (module, task, prefix):
            task_state_dict = {}

            if isinstance(module, LoRALayer):
                return module.state_dict(task=task, prefix=f"{prefix}{task}.")

            for name,child in module.named_children():
                task_state_dict = {**task_state_dict, **df_task_state_dict(child, task, prefix=f"{prefix}{name}.")}

            return task_state_dict

        return collections.OrderedDict(df_task_state_dict(self.model, task, prefix))

    def load_state_dict (self, state_dict, task=None, prefix="", **kwargs):
        if task is None: return self.model.load_state_dict(state_dict)

        def df_load_task_state_dict (module, state_dict, task, prefix):
            if isinstance(module, LoRALayer):
                return module.load_state_dict(state_dict, task=task, prefix=f"{prefix}{task}.")

            for name,child in module.named_children():
                df_load_task_state_dict (child, state_dict, task, prefix=f"{prefix}{name}.")

        df_load_task_state_dict(self.model, state_dict, task, prefix)
