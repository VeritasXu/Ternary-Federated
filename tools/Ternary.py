# -*- coding: utf-8 -*-
import time
import torch
import numpy as np
import torch.nn.functional as F
from utils.Evaluate import evaluate





def quantize(kernel, w_p, args):
    """
    :return quantized weights of a certain layer, which is: w_p * [1, 0, -1].
    """
    prob = np.random.rand()
    if prob > 0.5:
        T_k = args.T_thresh + 0.01 * prob
    else:
        T_k = args.T_thresh + 0.01 * 0.5

    if args.ada_thresh is False:
        delta = T_k * kernel.abs().max()
    else:
        T_a = 0.07
        d2 = kernel.size(0) * kernel.size(1)
        delta = T_a * kernel.abs().sum() / d2

        tmp1 = (kernel.abs() > delta).sum()
        tmp2 = ((kernel.abs() > delta)*kernel.abs()).sum()
        w_p = tmp2 / tmp1


    a = (kernel > delta).float()
    b = (kernel < -delta).float()

    return w_p*a + (-w_p*b)


def get_grads(kernel_grad, kernel, w_p, args):
    """
    Arguments:
        kernel_grad: gradient with respect to quantized kernel.
        kernel: corresponding full precision kernel.
        w_p: quantization factor.
        args: arguments

    Returns:
        1. gradient for the full precision kernel.
        2. gradient for w_p.
    """
    T_k = args.T_thresh

    delta = T_k * kernel.abs().max()

    # masks
    a = (kernel > delta).float().to(args.device)
    b = (kernel < -delta).float().to(args.device)

    c = torch.ones(kernel.size()).to(args.device) - a - b

    return w_p*a*kernel_grad + w_p*b*kernel_grad + 1.0*c*kernel_grad, (a*kernel_grad).sum()



def optimization_step(model, loss, x_batch, y_batch, current_round, optimizer_list, args):
    """Make forward pass and update model parameters with gradients."""

    total_epoch = current_round*args.local_e

    optimizer, optimizer_fp, optimizer_sf = optimizer_list



    x_batch = x_batch.to(args.device)
    y_batch = y_batch.to(args.device)
    logits = model(x_batch)

    loss_value = loss(logits, y_batch)
    batch_loss = loss_value.item()

    pred = F.softmax(logits, dim=1)

    optimizer.zero_grad()
    optimizer_fp.zero_grad()
    optimizer_sf.zero_grad()
    loss_value.backward()

    all_kernels = optimizer.param_groups[1]['params']
    all_fp_kernels = optimizer_fp.param_groups[0]['params']
    scaling_factors = optimizer_sf.param_groups[0]['params']

    wp_lists = [0 for i in range(len(all_kernels))]

    for i in range(len(all_kernels)):
        k = all_kernels[i]
        k_fp = all_fp_kernels[i]
        f = scaling_factors[i]
        w_p = f.data
        k_fp_grad, w_p_grad = get_grads(k.grad.data, k_fp.data, w_p, args)
        k_fp.grad = k_fp_grad
        k.grad.data.zero_()
        f.grad = w_p_grad.to(args.device)



    optimizer.step()
    # update all full precision kernels
    optimizer_fp.step()
    # update all scaling factors
    optimizer_sf.step()


    for i in range(len(all_kernels)):
        k = all_kernels[i]
        k_fp = all_fp_kernels[i]
        f = scaling_factors[i]
        w_p = f.data
        # re-quantize a quantized kernel using updated full precision weights
        k.data = quantize(k_fp.data, w_p, args)

        wp_lists[i] = f.clone()

    return wp_lists


def ternary_train(model, loss, optimization_step_fn, train_iterator, val_iterator, client_name, args):
    """
    Train 'model' by minimizing 'loss' using 'optimization_step_fn'
    for parameter updates.
    """

    # collect losses and accuracies here
    all_losses = []
    start_time = time.time()
    model.train()

    acc = []
    wp_lists = []
    record_list = np.zeros((100, 5))
    for epoch in range(0, args.local_e):

        # main training loop
        for ind, (x_batch, y_batch) in enumerate(train_iterator):
            wp_lists = optimization_step_fn(model, loss, x_batch, y_batch, args)


    end_time = time.time()
    test_loss, test_acc, test_top5_acc = evaluate(model, loss, val_iterator, args)

    all_losses += [(
        client_name,
        test_loss,
        test_acc,
        test_top5_acc
    )]

    acc.append(test_acc*100)

    out_str = 'Client:{0: d}, test loss:{1:.3f}, ' + \
              'test acc:{2:.3f}, ' + \
              'test top5:{3:.3f}, elapsed time:{4:.3f}'
    print(out_str.format(*all_losses[-1], end_time - start_time))

    return model.state_dict(), wp_lists
