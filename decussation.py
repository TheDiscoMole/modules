import copy
import torch

class Decussation (torch.nn.Module):
    """
    A wrapper module which generates decussation gradients.

    model: output module
    criterion: criterion to use for decussation                                 (optional|default: MSE)
    dropout: rate with which to sample inputs for decussation                   (optional|default: 0.3)
    loss_scale: loss and gradient scaling factor                                (optional|default: 0.1)
    """
    def __init__ (self, model,
        criterion=torch.nn.MSELoss,
        dropout=3e-1,
        loss_scale=1e-1
    ):
        super(Decussation, self).__init__()

        # configs
        self.loss_scale = loss_scale

        # modules
        self.criterion = criterion()
        self.dropout = torch.nn.Dropout(dropout)
        self.model = model

    def forward (self, input):
        # output
        output = self.dropout(input)
        output = self.model(output)

        # decussation
        if self.training:
            decussation = self.dropout(input)
            decussation = self.model(decussation)

            loss = self.criterion(decussation, output) * self.loss_scale
            loss.backward()

        return output
