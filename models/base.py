"""
Adapted from UZH-RPG https://github.com/uzh-rpg/rpg_e2vid
"""

from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

    def input_cropping(self, x, crop_size):
        """
        Extracts crops of the corners of the input image and stackes them along the batch dimension.
        """
        top_left = x[:, :, :crop_size, :crop_size]
        top_right = x[:, :, :crop_size, -crop_size:]
        bot_right = x[:, :, -crop_size:, -crop_size:]
        bot_left = x[:, :, -crop_size:, :crop_size]
        crops = [top_left, top_right, bot_right, bot_left]

        return torch.cat(crops, dim=0)
