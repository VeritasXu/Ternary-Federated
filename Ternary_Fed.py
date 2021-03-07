#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import sys
import copy
import torch
import numpy as np
from utils.config import Args
from utils.Evaluate import evaluate
import utils.data_utils as data_utils
from tools.Fed_Operator import ServerUpdate, LocalUpdate

if Args.model == 'MLP':
    from model.MLP import MLP as Fed_Model
elif Args.model == 'CNN':
    from model.CNN import CNN as Fed_Model
elif Args.model == 'ResNet':
    from model.resnet import ResNet18 as Fed_Model



def choose_model(f_dict, ter_dict):
    tmp_net1 = Fed_Model()
    tmp_net2 = Fed_Model()
    tmp_net1.load_state_dict(f_dict)
    tmp_net2.load_state_dict(ter_dict)

    _, acc_1, _ = evaluate(tmp_net1, G_loss_fun, test_iter, Args)
    _, acc_2, _ = evaluate(tmp_net2, G_loss_fun, test_iter, Args)
    print('F: %.3f' % acc_1, 'TF: %.3f' % acc_2)

    flag = False
    if np.abs(acc_1-acc_2) < 0.03:
        flag = True
        return ter_dict, flag
    else:
        return f_dict, flag


if __name__ == '__main__':

    torch.manual_seed(Args.seed)

    C_iter, train_iter, test_iter, stats = data_utils.get_dataset(args=Args)
    # build global network
    G_net = Fed_Model()
    print(G_net)
    G_net.train()
    G_loss_fun = torch.nn.CrossEntropyLoss()


    # copy weights
    w_glob = G_net.state_dict()

    m = max(int(Args.frac * Args.num_C), 1)

    gv_acc = []

    net_best = None
    val_acc_list, net_list = [], []
    num_s2 = 0
    # training
    c_lists = [[] for i in range(Args.num_C)]
    for rounds in range(Args.rounds):
        w_locals = []
        client_id = np.random.choice(range(Args.num_C), m, replace=False)
        print('Round {:d} start'.format(rounds, client_id))
        num_samp = []
        for idx in client_id:
            local = LocalUpdate(client_name = idx, c_round = rounds, train_iter = C_iter[idx], test_iter = test_iter, wp_lists= c_lists[idx], args=Args)
            w, wp_lists = local.TFed_train(net=copy.deepcopy(G_net).to(Args.device))
            c_lists[idx] = wp_lists
            w_locals.append(copy.deepcopy(w))

            num_samp.append(len(C_iter[idx].dataset))
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

