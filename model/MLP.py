import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.config import Args


class MLP(nn.Module):
    """
    define MLP model
    """
    def __init__(self):
        super(MLP, self).__init__()
        self.fp_layer1 = nn.Linear(784, 30, bias=False)

        self.ternary_layer1 = nn.Linear(30, 20, bias=False)

        self.fp_layer2 = nn.Linear(20, 10)


    def forward(self, x):
        x = x.cpu().view(-1, 784).to(Args.device)

        x = self.fp_layer1(x)
        x = self.ternary_layer1(x)
        x = self.fp_layer2(x)

        output = F.log_softmax(x, dim=1)
        return output






def Quantized_MLP(pre_model, args):
    """
    quantize the MLP model
    :param pre_model:
    :param args:
    :return:
    """

    #full-precision first and last layer
    weights = [p for n, p in pre_model.named_parameters() if 'fp_layer' in n and 'weight' in n]
    biases = [pre_model.fp_layer2.bias]

    #layers that need to be quantized
    ternary_weights = [p for n, p in pre_model.named_parameters() if 'ternary' in n]

    params = [
        {'params': weights},
        {'params': ternary_weights},
        {'params': biases}
    ]

    optimizer = optim.SGD(params, lr=args.lr)
    loss_fun = nn.CrossEntropyLoss()

    return pre_model, loss_fun, optimizer