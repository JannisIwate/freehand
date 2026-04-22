import torch

class EfficientNetFeatureRet(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._features = None

        def hook(module, input, output):
            self._features = output

        self.handle = self.model.features.register_forward_hook(hook)

    def forward(self, x):
        self._features = None
        out = self.model(x)
        return out, self._features