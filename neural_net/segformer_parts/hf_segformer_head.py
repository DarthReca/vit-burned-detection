# Copyright 2018- The Hugging Face team. All rights reserved.
# MODIFICATIONS TO MADE THE MODEL COMPATIBLE WITH PYTORCH

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegformerMLP(nn.Module):
    """
    Linear Embedding.
    """

    def __init__(self, hidden_size, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class HFSegformerHead(nn.Module):
    def __init__(
        self,
        hidden_sizes: List[int],
        decoder_hidden_size: int,
        classifier_dropout_prob: float,
        num_classes: int,
        reshape_last_stage: bool,
    ):
        super().__init__()
        num_encoder_blocks = len(hidden_sizes)
        self.reshape_last_stage = reshape_last_stage
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        mlps = []
        for i in range(num_encoder_blocks):
            mlp = SegformerMLP(decoder_hidden_size, input_dim=hidden_sizes[i])
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=decoder_hidden_size * num_encoder_blocks,
            out_channels=decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(decoder_hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(classifier_dropout_prob)
        self.classifier = nn.Conv2d(decoder_hidden_size, num_classes, kernel_size=1)

    def forward(self, encoder_hidden_states):
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            if self.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )

            # unify channel dimension
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(
                batch_size, -1, height, width
            )
            # upsample
            encoder_hidden_state = F.interpolate(
                encoder_hidden_state,
                size=encoder_hidden_states[0].size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            all_hidden_states += (encoder_hidden_state,)

        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # logits are of shape (batch_size, num_labels, height/4, width/4)
        logits = self.classifier(hidden_states)

        return logits
