import torch
import torch.nn as nn

class ModeloTransformer(nn.Module):
    def __init__(self, input_size=126, num_clases=7, seq_length=40, dim_model=128, num_heads=4, num_layers=2):
        super(ModeloTransformer, self).__init__()

        self.embedding = nn.Linear(input_size, dim_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Linear(dim_model, num_clases)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        x = self.embedding(x)               # (batch, seq, dim_model)
        x = self.transformer(x)             # (batch, seq, dim_model)
        x = x.mean(dim=1)                   # promedio sobre la secuencia
        return self.classifier(x)
