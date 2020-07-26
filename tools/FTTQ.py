# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
from model.MLP import Ternary_MLP
from torch.utils.data import DataLoader
from tools.Ternary import optimization_step, quantize, initial_scales, ternary_train




def fed_ttq(pre_model, train_set, test_set, client_name, current_round, args):

    # dataset setup
    train_iter = DataLoader(train_set, batch_size=args.batch_size, num_workers=2, shuffle=True)
    val_iter = DataLoader(test_set, batch_size=args.batch_size, num_workers=2, shuffle=False)

    # model setup
    model, loss_fun, optimizer = Ternary_MLP(pre_model=pre_model, args=args)
    model.to(args.device)

    model.train()
    # copy almost all full precision kernels of the model
    all_fp_kernels = [
        kernel.clone().detach().requires_grad_(True)
        for kernel in optimizer.param_groups[1]['params']]

    # init quantification
    initial_scaling_factors = []

    all_kernels = [kernel for kernel in optimizer.param_groups[1]['params']]

    for k, k_fp in zip(all_kernels, all_fp_kernels):
        #intialize w_p
        w_p_initial = initial_scales()
        initial_scaling_factors += [w_p_initial]
        # quantization
        k.data = quantize(k_fp.data, w_p_initial, args)

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

    model_dict, local_loss = ternary_train(model, loss_fun, optimization_step_fn, train_iter, val_iter, client_name, args)

    return model_dict, local_loss

