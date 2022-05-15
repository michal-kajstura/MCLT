from typing import Optional

from torch import nn
from transformers import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaLayer


def add_adapter_layers(model: RobertaModel):
    for layer in model.encoder.layer:
        layer: RobertaLayer
        adapter_output = AdapterRobertaOutput(model.config)
        keys = adapter_output.load_state_dict(layer.output.state_dict(), strict=False)
        assert all('adapter' in key for key in keys.missing_keys)
        layer.output = adapter_output


class AdapterRobertaOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.adapter = Adapter(config.hidden_size)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        adapted = self.adapter(hidden_states, input_tensor, self.LayerNorm)
        return adapted


class Adapter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        reduction_factor: Optional[int] = 16,
    ):
        super().__init__()
        downsample_size = hidden_size // reduction_factor
        self._bottleneck = nn.Sequential(
            nn.Linear(hidden_size, downsample_size),
            nn.LeakyReLU(),
            nn.Linear(downsample_size, hidden_size),
        )
        self.apply(self.init_bert_weights)

    @staticmethod
    def init_bert_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # std defaults to 0.02, this might need to be changed
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_hidden_states, residual_input, layer_norm):
        adapted = self._bottleneck(input_hidden_states) + input_hidden_states
        return layer_norm(adapted + residual_input)
