import einops

class ReshapeSimple ():
    """
    Wrapper for simple einop rearranges

    format: string format for reshaping
    inverse: string format for inverse reshaping                                (optional|default: format flipped)
    """
    def __init__ (self, format,
        inverse=None,
    ):
        # validation
        assert format.count(" -> ") == 1, "invalid einops string format"

        # reshaping
        self.rearrange_forward_format = format
        self.rearrange_inverse_format = " -> ".join(format.split(" -> ")[::-1]) if inverse is None else inverse

    def __call__ (self, tensor, **kwargs):
        return einops.rearrange(tensor, self.rearrange_forward_format, **kwargs)

    def inverse (self, tensor, **kwargs):
        return einops.rearrange(tensor, self.rearrange_inverse_format, **kwargs)

class ReshapeNdFlatten ():
    """
    Reshapes an n-dimensional input tensor into a flattened sequence of patches.

    dims: number of dimensions to flatten/unflatten
    """
    def __init__ (self, dims):
        # configs
        self.dims = dims

        # reshaping
        dims = ' '.join("x%d" % dim for dim in range(self.dims))

        self.rearrange_forward_format = "b ... f -> b (...) f"
        self.rearrange_inverse_format = "b (%s) f -> b %s f" % (dims, dims)

    def __call__ (self, tensor):
        return einops.rearrange(tensor, self.rearrange_forward_format)

    def inverse (self, tensor, dims):
        return einops.rearrange(tensor, self.rearrange_inverse_format, **{"x%d" % dim:size for dim, size in enumerate(dims[1:-1])})

class ReshapePatchesNdBatches ():
    """
    Reshape n-dimensional input tensor into patches across batches

    patches: (n-2)-dimensional patch sizes
    """
    def __init__ (self, patches):
        # validation
        assert len(patches) > 0, "need to provice n dimensional sizes"

        # configs
        self.dims = len(patches)
        self.patches = patches

        # reshaping
        dims = ["x%d" % dim for dim in range(self.dims)]
        patches = {"p%d" % dim:patch for dim,patch in enumerate(patches)}

        self.rearrange_forward_format = "b %s f -> (b %s) %s f" % (' '.join("(%s %s)" % slice for slice in zip(dims,patches.keys())), ' '.join(dims), ' '.join(patches.keys()))
        self.rearrange_inverse_format = "(b %s) %s f -> b %s f " % (' '.join(dims), ' '.join(patches.keys()), ' '.join("(%s %s)" % slice for slice in zip(dims,patches.keys())))

        self.rearrange_kwargs = patches

    def __call__ (self, tensor):
        return einops.rearrange(tensor, self.rearrange_forward_format, **self.rearrange_kwargs)

    def inverse (self, tensor, dims):
        return einops.rearrange(tensor, self.rearrange_inverse_format, **{**self.rearrange_kwargs, **{"x%d" % dim:size // patch for dim, (patch, size) in enumerate(zip(self.patches, dims[1:-1]))}})

class ReshapePatchesNdFeatures ():
    """
    Reshape n-dimensional input tensor into patches across features

    patches: (n-2)-dimensional patch sizes
    """
    def __init__ (self, patches):
        # validation
        assert len(patches) > 0, "need to provice n dimensional sizes"

        # configs
        self.dims = len(patches)
        self.patches = patches

        # reshaping
        dims = ["x%d" % dim for dim in range(self.dims)]
        patches = {"p%d" % dim:patch for dim,patch in enumerate(patches)}

        self.rearrange_forward_format = "b %s f -> b %s (f %s)" % (' '.join("(%s %s)" % slice for slice in zip(dims,patches.keys())), ' '.join(dims), ' '.join(patches.keys()))
        self.rearrange_inverse_format = "b %s (f %s) -> b %s f " % (' '.join(dims), ' '.join(patches.keys()), ' '.join("(%s %s)" % slice for slice in zip(dims,patches.keys())))

        self.rearrange_kwargs = patches

    def __call__ (self, tensor):
        return einops.rearrange(tensor, self.rearrange_forward_format, **self.rearrange_kwargs)

    def inverse (self, tensor, dims):
        return einops.rearrange(tensor, self.rearrange_inverse_format, **{**self.rearrange_kwargs, **{"x%d" % dim:size // patch for dim, (patch, size) in enumerate(zip(self.patches, dims[1:-1]))}})
