from typing import Optional
import torch
from torch import nn

from transformers import TimesformerModel


class TimeSformer(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.prepare_model()

    def prepare_model(self):
        num_classes = [self.cfg.MODEL.NUM_CLASSES] * self.cfg.DATA.NUM_FRAMES

        self.encoder = TimesformerModel.from_pretrained(  # [B, T, C, H, W]を期待
            "facebook/timesformer-base-finetuned-k400",
            ignore_mismatched_sizes=True,
        )
        embed_dim = self.encoder.config.hidden_size

        self.head = TransformerBasicHead(embed_dim, num_classes)

    def forward(
        self,
        pixel_values: torch.Tensor,  # [B, C, T, H, W]で渡してくる
        video_names: Optional[str] = None,
    ):

        # [B, C, T, H, W] -> [B, T, C, H, W]
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        outputs = self.encoder(pixel_values=pixel_values)
        cls_token_features = outputs.last_hidden_state[:, 0, :]

        return self.head(cls_token_features)


class TransformerBasicHead(nn.Module):
    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        super().__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        if isinstance(num_classes, (list, tuple)):
            self.projection = nn.ModuleList(
                [nn.Linear(dim_in, c, bias=True) for c in num_classes]
            )
            self.multi_classification = True
        else:
            self.projection = nn.Linear(dim_in, num_classes, bias=True)
            self.multi_classification = False

        if act_func == "softmax":
            self.act = nn.Softmax()
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(f"{act_func} is not supported.")

    def forward(self, x):

        if hasattr(self, "dropout"):
            x = self.dropout(x)

        if self.multi_classification:
            out = []
            for proj in self.projection:
                z = proj(x)
                if not self.training:
                    z = self.act(z)
                out.append(z)
            return out

        x = self.projection(x)
        if not self.training:
            x = self.act(x)
        return x
