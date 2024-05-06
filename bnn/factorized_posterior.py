from os.path import join
import csv
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.nn import Module, Parameter
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import _reduction as _Reduction
import math
import torch.optim as optim
import random

from scipy.stats import norm
import matplotlib.pyplot as plt

datadir = "data/bank-note"

parser = argparse.ArgumentParser()
parser.add_argument('--datafile', default="data/data.jsonl")
parser.add_argument('--lr', type=float, default=1.0, help='Learning rate.')
parser.add_argument('--batch_size', type=int, default=4, help='Training batch size.')
parser.add_argument("--layer_size", type=int, default=20, help="Size of each hidden layer")
parser.add_argument('--num_classes', type=int, default=2, help='Number of classes.')
parser.add_argument('--n_train_steps', type=int, default=10, help='Number of training steps.')
parser.add_argument('--beta', type=float, default=1.0, help='ELBO kl divergence weight.')
parser.add_argument('--num_samples', type=int, default=10, help='Nb. of params samples in inference.')
parser.add_argument('--activation', choices=['tanh', 'relu'], default='tanh', help="Activation function: tanh or relu")
args = parser.parse_args()


class BayesLinear(Module):

    __constants__ = ['in_features', 'out_features']
                                                                           
    def __init__(self, in_features, out_features):
        super(BayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weights_mu = Parameter(torch.zeros((out_features, in_features)))
        self.weights_rho = Parameter(torch.zeros((out_features, in_features)))

        mean = torch.zeros(self.weights_mu.size())
        std = torch.ones(self.weights_rho.size())
        self.prior_mu = torch.normal(mean, std)
        self.prior_rho = torch.normal(mean, std)
    
    def reset_parameters(self):
        # Initialization method of BNN: sample weights from
        # normal distribution
        weights_sigma = torch.log(1 + torch.exp(self.weights_rho))
        self.weights = torch.normal(mean=self.weights_mu, std=weights_sigma)

    def forward(self, input):

        eps_mean = torch.zeros(self.weights_mu.size())
        eps_std = torch.ones(self.weights_mu.size())
        eps = torch.normal(mean=eps_mean, std=eps_std)

        sigma = (torch.log(1 + torch.exp(self.weights_rho)))
        w_offset = eps * sigma
        w = self.weights_mu + w_offset

        return F.linear(input, w)
        

def factorized_kl_loss(mu_0, rho_0, mu_1, rho_1) :
    """
    Sum of KL divergences between `n` distributions

    Arguments are all tensor [Float] of size `n`
    """

    sigma_0 = torch.log(1 + torch.exp(rho_0))
    sigma_1 = torch.log(1 + torch.exp(rho_1))

    term1 = torch.log(sigma_1 / sigma_0)
    term2 = ((sigma_0**2) + (mu_0-mu_1)**2) / (2 * sigma_1**2)
    kl = term1 + term2 - 0.5

    return kl.sum()


def gaussian_entropy(mu, rho):
    """
    Sum of gaussian entropies between `n` distributions
    """

    sigma = torch.log(1 + torch.exp(rho))
    entropies = 2 * math.pi * math.e * sigma
    return entropies.sum()


def bayesian_kl_loss(model) :
    """
    An method for calculating KL divergence of whole layers in the model. (KL + bayesian entropy)

    Arguments:
        model (nn.Module): a model to be calculated for KL-divergence.        
    """

    device = torch.device("cpu")
    kl_sum = torch.Tensor([0]).to(device)
    n = torch.Tensor([0]).to(device)

    for m in model.modules() :
        if isinstance(m, BayesLinear):
            kl = factorized_kl_loss(m.weights_mu,
                                    m.weights_rho,
                                    m.prior_mu,
                                    m.prior_rho)
            entropy = gaussian_entropy(m.weights_mu,
                                       m.weights_rho)
            kl_sum += kl + entropy
            n += len(m.weights_mu.view(-1))

    return kl_sum


class _Loss(Module):
    def __init__(self):
        super(_Loss, self).__init__()


class BKLLoss(_Loss):
    """
    Loss for calculating KL divergence of baysian neural network model.
    """

    def __init__(self):
        super(BKLLoss, self).__init__()

    def forward(self, model):
        """
        Arguments:
            model (nn.Module): a model to be calculated for KL-divergence.
        """
        return bayesian_kl_loss(model)

    
def load_data(filename):

    def tofloat(row):
        return [float(x) for x in row]
    
    with open(join(datadir, filename)) as f:
        data = list(csv.reader(f))

    features = [tofloat(row[:4]) for row in data]
    labels = [[float(row[4])] for row in data]

    f_tensor = torch.tensor(features)
    l_tensor = torch.tensor(labels)
    return f_tensor, l_tensor


def init_model():

    hsize = args.layer_size
    if args.activation == 'relu':
        model = nn.Sequential(
            BayesLinear(in_features=4, out_features=hsize),
            nn.ReLU(),
            BayesLinear(hsize, hsize),
            nn.ReLU(),
            BayesLinear(hsize, hsize),
            nn.ReLU(),
            BayesLinear(hsize, 1),
            nn.Sigmoid())
    elif args.activation == 'tanh':
        model = nn.Sequential(
            BayesLinear(in_features=4, out_features=hsize),
            nn.Tanh(),
            BayesLinear(hsize, hsize),
            nn.Tanh(),
            BayesLinear(hsize, hsize),
            nn.Tanh(),
            BayesLinear(hsize, 1),
            nn.Tanh())
    return model


def get_accuracy(model, features, labels):

    pre = model(features)
    predicted = torch.bernoulli(pre)
    total = labels.size(0)

    correct = 0.0
    likelihood = 0.0
    for i in range(total):
        if predicted[i] == labels[i]:
            correct += 1
        
    acc = 100 * correct / total
    
    return acc


def get_accuracy_average(n, model, features, labels):

    accs = [get_accuracy(model, features, labels) for i in range(n)]
    avg_acc = sum(accs) / len(accs)
    return avg_acc


def run():

    features, labels = load_data("train.csv")
    test_features, test_labels = load_data("test.csv")

    model = init_model()
    
    ce_loss = nn.CrossEntropyLoss()
    kl_loss = BKLLoss()
    kl_weight = args.beta

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    trainaccs = []
    testaccs = []
    
    for i in range(args.n_train_steps):
        pre = model(features)
        ce = ce_loss(pre, labels)
        kl = kl_loss(model)
        cost = ce + kl_weight*kl

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # accuracy = get_accuracy(model, features, labels)
        # print('- Train Accuracy: %f %%' % accuracy)
        print('- CE : %2.2f, KL : %2.2f' % (ce.item(), kl.item()))

        if i % 5 == 0:
            trainacc = get_accuracy_average(100, model, features, labels)
            testacc = get_accuracy_average(100, model, test_features, test_labels)
            trainaccs.append(trainacc)
            testaccs.append(testacc)
            
    return trainaccs, testaccs

    
if __name__ == "__main__":

    trainaccs, testaccs = run()

    plt.plot([5*x for x in range(1,21)], trainaccs)
    plt.savefig("trainacc.png")
    plt.plot([5*x for x in range(1,21)], testaccs)
    plt.savefig("testacc.png")
                
    
