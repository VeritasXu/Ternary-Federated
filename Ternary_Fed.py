#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import sys
import copy
import torch
import numpy as np
import pandas as pd
from utils.config import Args
from model.MLP import MLP
from utils.Evaluate import evaluate
from torch.utils.data import DataLoader
from utils.load_mnist import M_part, mnist_test
from tools.Fed_Operator import ServerUpdate, LocalUpdate

def choose_model(f_dict, ter_dict):
    tmp_net1 = MLP()
    tmp_net2 = MLP()
    tmp_net1.load_state_dict(f_dict)
    tmp_net2.load_state_dict(ter_dict)

    _, acc_1, _ = evaluate(tmp_net1, G_loss_fun, test_iter, Args)
    _, acc_2, _ = evaluate(tmp_net2, G_loss_fun, test_iter, Args)
    print('F: ', acc_1, 'TF: ', acc_2)

    flag = False
    if np.abs(acc_1-acc_2) <= 3:
        flag = True
        return ter_dict, flag
    else:
        return f_dict, flag


if __name__ == '__main__':

    if Args.device == 'cuda':
        torch.cuda.manual_seed(Args.seed)

    # build global network
    G_net = MLP()

    G_net.train()
    G_loss_fun = torch.nn.CrossEntropyLoss()

    test_iter = DataLoader(mnist_test, batch_size=Args.batch_size, shuffle=False)

    # copy weights
    w_glob = G_net.state_dict()


    m = max(int(Args.frac * Args.num_C), 1)


    gv_acc = []

    net_best = None
    val_acc_list, net_list = [], []
    num_s2 = 0
    # training
    for rounds in range(Args.rounds):
        w_locals, loss_locals = [], []
        client_id = np.random.choice(range(Args.num_C), m, replace=False)
        print('Round {:d} start'.format(rounds, client_id))
        num_samp = []
        for idx in client_id:
            local = LocalUpdate(client_name = idx, c_round = rounds, train_set = M_part[str(idx)], test_set = mnist_test, args=Args)
            w, idx_loss = local.TFed_train(net=copy.deepcopy(G_net).to(Args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(idx_loss)
            num_samp.append(len(M_part[str(idx)]))
        # update global weights
        w_glob, ter_glob = ServerUpdate(w_locals, num_samp)

        w_glob, tmp_flag = choose_model(w_glob, ter_glob)
        if tmp_flag:
            num_s2 += 1
            print('S1')

        # reload global network weights
        G_net.load_state_dict(w_glob)

        #verify accuracy on test set
        g_loss, g_acc, g_acc5 = evaluate(G_net, G_loss_fun, test_iter, Args)
        gv_acc.append(g_acc)


        print('Round {:3d}, Global loss {:.3f}, Global Acc {:.3f}'.format(rounds, g_loss, g_acc))

