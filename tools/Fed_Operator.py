#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import torch
import numpy as np
from utils.config import Args
from tools.FTTQ import fed_ttq


def quantize_server(model_dict):
    """
    Return quantized weights of all model.
    Only possible values of quantized weights are: {0, 1, -1}.
    """

    for key, kernel in model_dict.items():
        # quantize the ternary layer in the global model
        if Args.model is 'MLP' and 'ternary' in key:
            print(key)
            d2 = kernel.size(0) * kernel.size(1)
            delta = 0.05 * kernel.abs().sum() / d2
            tmp1 = (kernel.abs() > delta).sum()
            tmp2 = ((kernel.abs() > delta) * kernel.abs()).sum()
            w_p = tmp2 / tmp1
            a = (kernel > delta).float()
            b = (kernel < -delta).float()
            kernel = w_p * a - w_p * b
            model_dict[key] = kernel
        elif Args.model is not 'MLP' and 'ternary' and 'conv' in key:
            print(key)
            d2 = kernel.size(0) * kernel.size(1)
            delta = 0.05 * kernel.abs().sum() / d2
            tmp1 = (kernel.abs() > delta).sum()
            tmp2 = ((kernel.abs() > delta) * kernel.abs()).sum()
            w_p = tmp2 / tmp1
            a = (kernel > delta).float()
            b = (kernel < -delta).float()
            kernel = w_p * a - w_p * b
            model_dict[key] = kernel

    return model_dict



def ServerUpdate(w, num_samp):
    '''
    :param w: all participating weights, list
    :param num_samp: number of data on each client, np.array
    :return:
    '''

    num_samp = np.array(num_samp)
    frac_c = num_samp / num_samp.sum()
    num_model = len(w)
    w_avg = w[0]
    for key, value in w_avg.items():
        for i in range(0, num_model):
            if i == 0:
                w_avg[key] = frac_c[0] * w[0][key]
            else:
                w_avg[key] += frac_c[i] * w[i][key]

    backup_w = copy.deepcopy(w_avg)

    ter_avg = quantize_server(backup_w)

    return w_avg, ter_avg


class LocalUpdate(object):
    def __init__(self, client_name, c_round, train_iter, test_iter, wp_lists, args):
        self.c_name = client_name
        self.c_round = c_round
        self.wp_lists = wp_lists
        self.args = args
        self.local_train = train_iter
        self.local_test = test_iter
        self.loss_func = torch.nn.CrossEntropyLoss()

    def TFed_train(self, net):
        # train and update
        net_dict, wp_lists = fed_ttq(net, self.local_train, self.local_test, self.c_name, self.c_round, self.wp_lists, self.args)

        return net_dict, wp_lists