import torch

class Decussation(torch.nn.Module):
    """
    A wrapper module which generates decussation gradients.

    model: output module
    criterion: criterion to use for decussation               (optional|default: MSE)
    dropout: rate with which to sample inputs for decussation (optional|default: 0.3)
    lossScale: loss and gradient scaling factor               (optional|default: 0.1)
    """
    def __init__ (self, model: torch.nn.Module,
        criterion: torch.nn.Module=torch.nn.MSELoss,
        dropout: float=3e-1,
        lossScale: float=1e-1,
        **kwargs: dict
    ) -> None:
        super().__init__()

        # configs
        self.lossScale = lossScale

        # modules
        self.criterion = criterion()
        self.dropout = torch.nn.Dropout(dropout)
        self.model = model

    def forward (self, *inputs: list, **kwargs: dict) -> torch.Tensor:
        # output
        output = (self.dropout(input) for input in inputs)
        output = self.model(*output, **kwargs)

        # decussation
        if self.training:
            decussation = (self.dropout(input) for input in inputs)
            decussation = self.model(*decussation, **kwargs)

            loss = self.criterion(decussation, output) * self.lossScale
            loss.backward()

        return output
