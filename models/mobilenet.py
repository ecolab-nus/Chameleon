#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" This file contains the model Class used for the exps"""

import torch
import torch.nn as nn

try:
    from pytorchcv.models.mobilenet import DwsConvBlock
except:
    from pytorchcv.models.common import DwsConvBlock
from pytorchcv.model_provider import get_model


def remove_sequential(network, all_layers):

    for layer in network.children():
        if isinstance(layer, nn.Sequential): # if sequential layer, apply recursively to layers in sequential layer
            remove_sequential(layer, all_layers)
        else: # if leaf node, add it to list
            all_layers.append(layer)

def remove_DwsConvBlock(cur_layers):

    all_layers = []
    for layer in cur_layers:
        if isinstance(layer, DwsConvBlock):
            for ch in layer.children():
                all_layers.append(ch)
        else:
            all_layers.append(layer)
    return all_layers


class MyMobilenetV1(nn.Module):
    def __init__(self, pretrained=True, latent_layer_num=21):
        super().__init__()

        model = get_model("mobilenet_w1", pretrained=pretrained)
        for name, param in model.named_parameters():
            print(name)
        model.features.final_pool = nn.AvgPool2d(4)

        all_layers = []
        remove_sequential(model, all_layers)
        all_layers = remove_DwsConvBlock(all_layers)

        lat_list = []
        end_list = []

        for i, layer in enumerate(all_layers[:-1]):
            if i <= latent_layer_num:
                lat_list.append(layer)
            else:
                end_list.append(layer)

        self.lat_features = nn.Sequential(*lat_list)
        self.end_features = nn.Sequential(*end_list)

        self.output = nn.Linear(1024, 50, bias=False)

    def forward(self, x, latent_input=None, return_lat_acts=False, lat=True):
        
        if lat == True:
            orig_acts = self.lat_features(x)
            if latent_input is not None:
                lat_acts = torch.cat((orig_acts, latent_input), 0)
            else:
                lat_acts = orig_acts
        else:
            lat_acts = latent_input

        feat = self.end_features(lat_acts)
        x = feat.view(feat.size(0), -1)
        logits = self.output(x)

        feat_size = feat.shape[-1]
        num_channels = feat.shape[1]
        features2 = feat.permute(0, 2, 3, 1)  # 1 x feat_size x feat_size x num_channels
        features3 = torch.reshape(features2, (
            feat.shape[0], feat_size * feat_size, num_channels))
        feat = features3.mean(1)  # mb x num_channels

        if return_lat_acts:
            return logits, orig_acts, feat
        else:
            return logits



if __name__ == "__main__":

    model = MyMobilenetV1(pretrained=True)
    for i in range(2, 7, 2):
        print(list(model.end_features.children())[:-1 * i])
        print("--------------------")
