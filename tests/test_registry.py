import torch.nn as nn

from ugtsdti.core.registry import MODELS


def test_registry_registration():
    """Test that the core registry pattern correctly registers and builds models"""

    # 1. Register a dummy model
    @MODELS.register("dummy_test_model")
    class DummyModel(nn.Module):
        def __init__(self, hidden_dim: int):
            super().__init__()
            self.hidden_dim = hidden_dim

    assert "dummy_test_model" in MODELS

    # 2. Build the dummy model from a config dict
    config = {"name": "dummy_test_model", "params": {"hidden_dim": 128}}

    instance = MODELS.build(config)
    assert isinstance(instance, DummyModel)
    assert instance.hidden_dim == 128
