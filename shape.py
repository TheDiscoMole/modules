import einops
import torch

class ReshapeSimple():
    """
    Wrapper for simple einop rearranges

    format: string format for reshaping
    inverse: string format for inverse reshaping  (optional|default: format flipped)
    """
    def __init__(self, format: str, 
        inverse: str|None=None
    ) -> None:
        # validation
        assert format.count(" -> ") == 1, "invalid einops string format"

        # reshaping
        self.operation = format
        self.inverseOperation = " -> ".join(format.split(" -> ")[::-1]) if inverse is None else inverse

    def __call__(self, tensor: torch.Tensor, **kwargs: dict) -> torch.Tensor:
        return einops.rearrange(tensor, self.operation, **kwargs)

    def inverse(self, tensor: torch.Tensor, **kwargs: dict) -> torch.Tensor:
        return einops.rearrange(tensor, self.inverseOperation, **kwargs)

class ReshapeNdFlatten():
    """
    Reshapes an n-dimensional input tensor into a flattened sequence of patches.

    dims: number of dimensions to flatten/unflatten
    """
    def __init__ (self, dims: int) -> None:
        # configs
        self.dims = dims

        # reshaping
        dimensionNames = ' '.join(f"x{dim}" for dim in range(self.dims))

        self.operation = "b ... f -> b (...) f"
        self.inverseOperation = "b (%s) f -> b %s f" % (dimensionNames, dimensionNames)

    def __call__ (self, tensor: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(tensor, self.operation)

    def inverse (self, tensor: torch.Tensor, dimensionSizes: list[int]) -> torch.Tensor:
        return einops.rearrange(tensor, self.inverseOperation, **{f"x{dimension}":size for dimension,size in enumerate(dimensionSizes[1:-1])})

class ReshapePatchesNdBatches():
    """
    Reshape n-dimensional input tensor into patches across batches

    patches: (n-2)-dimensional patch sizes
    """
    def __init__ (self, patches: list[int]) -> None:
        # validation
        assert len(patches) > 0, "need to provice n dimensional sizes"

        # configs
        self.dims = len(patches)
        self.patches = patches

        # reshaping
        dims = [f"x{dim}" for dim in range(self.dims)]
        patches = {f"p{dim}":patch for dim,patch in enumerate(patches)}

        self.operation = "b %s f -> (b %s) %s f" % (' '.join("(%s %s)" % slice for slice in zip(dims,patches.keys())), ' '.join(dims), ' '.join(patches.keys()))
        self.inverseOperation = "(b %s) %s f -> b %s f " % (' '.join(dims), ' '.join(patches.keys()), ' '.join("(%s %s)" % slice for slice in zip(dims,patches.keys())))

        self.rearrangeKwargs = patches

    def __call__ (self, tensor: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(tensor, self.operation, **self.rearrangeKwargs)

    def inverse (self, tensor: torch.Tensor, dimensionSizes: list[int]) -> torch.Tensor:
        return einops.rearrange(tensor, self.inverseOperation, **{**self.rearrangeKwargs, **{"x%d" % dim:size // patch for dim, (patch, size) in enumerate(zip(self.patches, dimensionSizes[1:-1]))}})

class ReshapePatchesNdFeatures():
    """
    Reshape n-dimensional input tensor into patches across features

    patches: (n-2)-dimensional patch sizes
    """
    def __init__ (self, patches: list[int]) -> None:
        # validation
        assert len(patches) > 0, "need to provice n dimensional sizes"

        # configs
        self.dims = len(patches)
        self.patches = patches

        # reshaping
        dims = ["x%d" % dim for dim in range(self.dims)]
        patches = {"p%d" % dim:patch for dim,patch in enumerate(patches)}

        self.operation = "b %s f -> b %s (f %s)" % (' '.join("(%s %s)" % slice for slice in zip(dims,patches.keys())), ' '.join(dims), ' '.join(patches.keys()))
        self.inverseOperation = "b %s (f %s) -> b %s f " % (' '.join(dims), ' '.join(patches.keys()), ' '.join("(%s %s)" % slice for slice in zip(dims,patches.keys())))

        self.rearrangeKwargs = patches

    def __call__ (self, tensor: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(tensor, self.operation, **self.rearrangeKwargs)

    def inverse (self, tensor: torch.Tensor, dimensionSizes: list[int]) -> torch.Tensor:
        return einops.rearrange(tensor, self.inverseOperation, **{**self.rearrangeKwargs, **{"x%d" % dim:size // patch for dim, (patch, size) in enumerate(zip(self.patches, dimensionSizes[1:-1]))}})
