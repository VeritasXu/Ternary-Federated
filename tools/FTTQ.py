# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
from tools.Ternary import optimization_step, quantize, ternary_train

from utils.config import Args as config

if config.model == 'CNN':
    from model.CNN import Quantized_CNN as network
elif config.model == 'ResNet':
    from model.resnet import Quantized_resnet as network
elif config.model == 'MLP':
    from model.MLP import Quantized_MLP as network


def initial_scales():
    """
    :return: initialized quantization factor w_p
    """
    return 1.0

def fed_ttq(pre_model, train_iter, test_iter, client_name, current_round, scale_factors, args):

    # model setup
    model, loss_fun, optimizer = network(pre_model=pre_model, args=args)
    model.to(args.device)

    model.train()
    # copy almost all full precision kernels of the model
    all_fp_kernels = [
        kernel.clone().detach().requires_grad_(True)
        for kernel in optimizer.param_groups[1]['params']]

    # init quantification
    initial_scaling_factors = []

    all_kernels = [kernel for kernel in optimizer.param_groups[1]['params']]

    ii = 0
    for k, k_fp in zip(all_kernels, all_fp_kernels):

        w_p_initial = initial_scales()

        initial_scaling_factors += [w_p_initial]
        # quantization
        k.data = quantize(k_fp.data, w_p_initial, args)
        ii += 1



    if config.optimizer == 'Adam':
        # optimizer for updating only all_fp_kernels
        optimizer_fp = optim.Adam(all_fp_kernels, lr=args.lr)

        # optimizer for updating only scaling factors
        optimizer_sf = optim.Adam([
            torch.tensor(w_p).to(args.device).requires_grad_(True)
            for w_p in initial_scaling_factors
        ], lr=args.lr)

    else:
        # optimizer for updating only all_fp_kernels
        optimizer_fp = optim.SGD(all_fp_kernels, lr=args.lr)

        # optimizer for updating only scaling factors
        optimizer_sf = optim.SGD([
            torch.tensor(w_p).to(args.device).requires_grad_(True)
            for w_p in initial_scaling_factors
        ], lr=args.lr)


    optimizer_list = [optimizer, optimizer_fp, optimizer_sf]

    def optimization_step_fn(p_model, loss_f, x_batch, y_batch, arg):
        return optimization_step(p_model, loss_f, x_batch, y_batch, current_round,
                                 optimizer_list, arg)

    model_dict, wp_lists = ternary_train(model, loss_fun, optimization_step_fn, train_iter, test_iter, client_name, args)

    return model_dict, wp_lists

