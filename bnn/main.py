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
import jsonlines
import random


parser = argparse.ArgumentParser()
parser.add_argument('--datafile', default="data/data.jsonl")
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--batch_size', type=int, default=4, help='Training batch size.')
parser.add_argument('--num_classes', type=int, default=2, help='Number of classes.')
parser.add_argument('--n_train_steps', type=int, default=10, help='Number of training steps.')
parser.add_argument('--beta', type=float, default=0.001, help='ELBO kl divergence weight.')
parser.add_argument('--num_samples', type=int, default=10, help='Nb. of params samples in inference.')
parser.add_argument('--log_interval', type=int, default=5, help='Training logging interval.')
parser.add_argument('--ckpt_path', default='./out/bayes_params.pkl', help='Checkpoint path.')
parser.add_argument('--ckpt_interval', type=int, default=5, help='Params checkpoint interval.')
args = parser.parse_args()


class Datapoint:

    def __init__(self, prompt_or_gen, datapoint):

        if prompt_or_gen == "prompt":
            self.init_prompt(datapoint)
        elif prompt_or_gen == "gen":
            self.init_gen(datapoint)

    def init_prompt(self, datapoint):

        self.text = datapoint["text"]

        self.scores = [datapoint["sexually_explicit"],
                       datapoint["profanity"],
                       datapoint["threat"],
                       datapoint["flirtation"],
                       datapoint["identity_attack"],
                       datapoint["toxicity"],
                       datapoint["insult"],
                       datapoint["severe_toxicity"]]

    def init_gen(self, datapoint):

        self.text = " | ".join([g["text"] for g in datapoint])

        all_scores = []
        for generation in datapoint:
            maybe_scores = [generation["sexually_explicit"],
                            generation["profanity"],
                            generation["threat"],
                            generation["flirtation"],
                            generation["identity_attack"],
                            generation["toxicity"],
                            generation["insult"],
                            generation["severe_toxicity"]]
            if None not in maybe_scores:
                all_scores.append(maybe_scores)

        self.scores = []

        for i in range(8):
            only_scores = [row[i] for row in all_scores]
            avg_score = sum(only_scores) / len(only_scores)
            self.scores.append(avg_score)

    def __repr__(self):

        s = ""
        s += self.text + "\n"
        s += "--------------------------------\n"
        s += str(self.scores) + "\n"
        s += "================================\n"
        return s

    
def load_data(datafile):

    def has_scores(gens):

        for gen in gens:
            for id in gen:
                if gen[id] == None:
                    return False
        return True

    def has_prompt_scores(prompt):
        for id in prompt:
            if prompt[id] == None:
                return False
        return True 

    prompts, generations = [], []
    with jsonlines.open(datafile) as reader:
        for n, obj in enumerate(reader):
            try:
                if has_scores(obj["generations"]) and has_prompt_scores(obj["prompt"]):
                    prompts.append(Datapoint("prompt", obj["prompt"]))
                    generations.append(Datapoint("gen", obj["generations"]))
            except ZeroDivisionError:
                continue

    features = torch.Tensor([p.scores for p in prompts])
    to_label = lambda x: 1 if (x > 0.2) else 0
    gen_avg = [np.average(obj.scores) for obj in generations]
    gen_labels = [[to_label(s)] for s in gen_avg]
    labels = torch.Tensor(gen_labels)

    return features, labels


class Dataset:

    def __init__(self, features, labels, batch_size):
        self.batch_size = batch_size
        self.i = 0
        self.features = features
        self.labels = labels

    def __next__(self):
        to = self.i + self.batch_size
        return_value = (self.features[self.i : to], self.labels[self.i : to])
        self.i += self.batch_size
        return return_value
    
    def __iter__(self):
        return self


class BayesLinear(Module):

    __constants__ = ['prior_mu', 'prior_sigma', 'bias', 'in_features', 'out_features']
                                                                           
    def __init__(self, prior_mu, prior_sigma, in_features, out_features, bias=True):
        super(BayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.prior_log_sigma = math.log(prior_sigma)
        
        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_sigma = Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_eps', None)
                
        if bias is None or bias is False :
            self.bias = False
        else :
            self.bias = True
            
        if self.bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_log_sigma = Parameter(torch.Tensor(out_features))
            self.register_buffer('bias_eps', None)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sigma', None)
            self.register_buffer('bias_eps', None)
            
        self.reset_parameters()

    
    def reset_parameters(self):
        # Initialization method of Adv-BNN
        stdv = 1. / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_log_sigma.data.fill_(self.prior_log_sigma)
        if self.bias :
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_log_sigma.data.fill_(self.prior_log_sigma)

    def forward(self, input):
        r"""
        Overriden.
        """
        if self.weight_eps is None :
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * torch.randn_like(self.weight_log_sigma)
        else :
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * self.weight_eps
        
        if self.bias:
            if self.bias_eps is None :
                bias = self.bias_mu + torch.exp(self.bias_log_sigma) * torch.randn_like(self.bias_log_sigma)
            else :
                bias = self.bias_mu + torch.exp(self.bias_log_sigma) * self.bias_eps                
        else :
            bias = None
            
        return F.linear(input, weight, bias)


def _kl_loss(mu_0, log_sigma_0, mu_1, log_sigma_1) :
    """
    An method for calculating KL divergence between two Normal distribtuion.

    Arguments:
        mu_0 (Float) : mean of normal distribution.
        log_sigma_0 (Float): log(standard deviation of normal distribution).
        mu_1 (Float): mean of normal distribution.
        log_sigma_1 (Float): log(standard deviation of normal distribution).
   
    """
    kl = log_sigma_1 - log_sigma_0 + \
    (torch.exp(log_sigma_0)**2 + (mu_0-mu_1)**2)/(2*math.exp(log_sigma_1)**2) - 0.5
    return kl.sum()


def bayesian_kl_loss(model, reduction='mean', last_layer_only=False) :
    """
    An method for calculating KL divergence of whole layers in the model.


    Arguments:
        model (nn.Module): a model to be calculated for KL-divergence.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'mean'``: the sum of the output will be divided by the number of
            elements of the output.
            ``'sum'``: the output will be summed.
        last_layer_only (Bool): True for return only the last layer's KL divergence.    
        
    """
    device = torch.device("cpu")
    kl = torch.Tensor([0]).to(device)
    kl_sum = torch.Tensor([0]).to(device)
    n = torch.Tensor([0]).to(device)

    for m in model.modules() :
        if isinstance(m, BayesLinear):
            kl = _kl_loss(m.weight_mu, m.weight_log_sigma, m.prior_mu, m.prior_log_sigma)
            kl_sum += kl
            n += len(m.weight_mu.view(-1))

            if m.bias :
                kl = _kl_loss(m.bias_mu, m.bias_log_sigma, m.prior_mu, m.prior_log_sigma)
                kl_sum += kl
                n += len(m.bias_mu.view(-1))
            
    if last_layer_only or n == 0 :
        return kl
    
    if reduction == 'mean' :
        return kl_sum/n
    elif reduction == 'sum' :
        return kl_sum
    else :
        raise ValueError(reduction + " is not valid")


class _Loss(Module):
    def __init__(self, reduction='mean'):
        super(_Loss, self).__init__()
        self.reduction = reduction


class BKLLoss(_Loss):
    """
    Loss for calculating KL divergence of baysian neural network model.

    Arguments:
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'mean'``: the sum of the output will be divided by the number of
            elements of the output.
            ``'sum'``: the output will be summed.
        last_layer_only (Bool): True for return only the last layer's KL divergence.    
    """
    __constants__ = ['reduction']

    def __init__(self, reduction='mean', last_layer_only=False):
        super(BKLLoss, self).__init__(reduction)
        self.last_layer_only = last_layer_only

    def forward(self, model):
        """
        Arguments:
            model (nn.Module): a model to be calculated for KL-divergence.
        """
        return bayesian_kl_loss(model, reduction=self.reduction, last_layer_only=self.last_layer_only)


def get_accuracy(model, features, labels):

    pre = model(features)
    _, predicted = torch.max(pre.data, 1)
    total = labels.size(0)

    correct = 0.0
    for i in range(total):
        if predicted[i] == labels[i]:
            correct += 1
    acc = 100 * correct / total
    return acc


def get_split(dataset):
    """4 batches train, 1 batch test"""

    batches = [next(dataset) for i in range(5)]
    random.shuffle(batches)

    return batches[:4], batches[4]


if __name__ == "__main__":

    # feats = [torch.rand((args.batch_size, 8), requires_grad=True) for i in range(100)]
    # labels = [torch.Tensor([[0.0]]) for i in range(100)]
    # dataset = Dataset(feats, labels)
    # features, label = next(dataset)

    f, l = load_data(args.datafile)
    dataset = Dataset(f, l, args.batch_size)

    model = nn.Sequential(
        BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=8, out_features=16),
        nn.ReLU(),
        BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=16, out_features=1),
        )

    ce_loss = nn.CrossEntropyLoss()
    kl_loss = BKLLoss(reduction='mean', last_layer_only=False)
    kl_weight = args.beta

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    crossval_accuracies = []
    
    for n in range(5):
        train_batches, test_batch = get_split(dataset)
    
        for i in range(10):
            features, labels = random.choice(train_batches)
       
            pre = model(features)
            ce = ce_loss(pre, labels)
            kl = kl_loss(model)
            cost = ce + kl_weight*kl
    
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            accuracy = get_accuracy(model, features, labels)
            print('- Train Accuracy: %f %%' % accuracy)
            print('- CE : %2.2f, KL : %2.2f' % (ce.item(), kl.item()))
    

        testacc = get_accuracy(model, test_batch[0], test_batch[1])
        print("Test accuracy: ", testacc)
        crossval_accuracies.append(testacc)

    print("Average cross-validated test accuracy:", sum(crossval_accuracies) / len(crossval_accuracies))
    """
    _, predicted = torch.max(pre.data, 1)
    total = labels.size(0)

    correct = 0.0
    for i in range(total):
        if predicted[i] == labels[i]:
            correct += 1
    print(correct)
    print('- Accuracy: %f %%' % (100 * float(correct) / total))
    print('- CE : %2.2f, KL : %2.2f' % (ce.item(), kl.item()))
    """
