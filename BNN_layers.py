"""
@author: david

Oriented at

https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/src/Bayes_By_Backprop/model.py
https://github.com/facebookresearch/RandomizedValueFunctions/blob/master/qlearn/commun/bayes_backprop_layer.py
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from uncertainty.normalizing_flows import MaskedNVPFlow


class _MCDropoutNd(nn.Module):
    __constants__ = ['p', 'inplace']
    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super(_MCDropoutNd, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def extra_repr(self) -> str:
        return 'p={}, inplace={}'.format(self.p, self.inplace)


class MCDropout(_MCDropoutNd):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.dropout(input, self.p, True, self.inplace)


class MNFLinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hparams):
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hparams = hparams
        
        self.weight_mu = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.weight_logstd = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.bias_mu = nn.Parameter(torch.Tensor(output_dim))
        self.bias_logvar = nn.Parameter(torch.Tensor(output_dim))

        self.qzero_mu = nn.Parameter(torch.Tensor(input_dim))
        self.qzero_logvar = nn.Parameter(torch.Tensor(input_dim))
        # auxiliary variable c, b1 and b2 are defined in equation (9) and (10)
        self.rzero_c = nn.Parameter(torch.Tensor(input_dim))
        self.rzero_b1 = nn.Parameter(torch.Tensor(input_dim))
        self.rzero_b2 = nn.Parameter(torch.Tensor(input_dim))

        self.flow_q = MaskedNVPFlow(input_dim, hidden_dim, self.hparams.n_hidden_mnf, 
                                    self.hparams.n_flows_q)
        self.flow_r = MaskedNVPFlow(input_dim, hidden_dim, self.hparams.n_hidden_mnf, 
                                    self.hparams.n_flows_r)

        self.register_buffer('epsilon_z', torch.Tensor(input_dim))
        self.register_buffer('epsilon_weight', torch.Tensor(output_dim, input_dim))
        self.register_buffer('epsilon_bias', torch.Tensor(output_dim))
        self.reset_parameters()
        self.reset_noise()

    def reset_noise(self):
        epsilon_z = torch.randn(self.input_dim)
        epsilon_weight = torch.randn(self.output_dim, self.input_dim)
        epsilon_bias = torch.randn(self.output_dim)
        self.epsilon_z.copy_(epsilon_z)
        self.epsilon_weight.copy_(epsilon_weight)
        self.epsilon_bias.copy_(epsilon_bias)
        self.flow_q.reset_noise()
        self.flow_r.reset_noise()

    def reset_parameters(self):

        in_stdv = np.sqrt(4.0 / self.input_dim)
        out_stdv = np.sqrt(4.0 / self.output_dim)
        stdv2 = np.sqrt(4.0 / (self.input_dim + self.output_dim))
        self.weight_mu.data.normal_(0, stdv2)
        self.weight_logstd.data.normal_(-9, 1e-3 * stdv2)
        self.bias_mu.data.zero_()
        self.bias_logvar.data.normal_(-9, 1e-3 * out_stdv)

        self.qzero_mu.data.normal_(1 if self.hparams.n_flows_q == 0 else 0, in_stdv)
        self.qzero_logvar.data.normal_(np.log(0.1), 1e-3 * in_stdv)
        self.rzero_c.data.normal_(0, in_stdv)
        self.rzero_b1.data.normal_(0, in_stdv)
        self.rzero_b2.data.normal_(0, in_stdv)

    def sample_z(self, kl=True):
        qzero_std = torch.exp(0.5 * self.qzero_logvar)
        z =  self.qzero_mu + qzero_std * self.epsilon_z
        if kl:
            z, logdets = self.flow_q(z, kl=True)
            return z, logdets
        else:
            z = self.flow_q(z, kl=False)
            return z

    def kl_div(self, logdets, z, weight):
        weight_mu = z.view(1, -1) * self.weight_mu
        kldiv_weight = 0.5 * (- 2 * self.weight_logstd + \
                              torch.exp(2 * self.weight_logstd)
                              + weight_mu * weight_mu - 1).sum()
        kldiv_bias = 0.5 * (- self.bias_logvar + torch.exp(self.bias_logvar)
                            + self.bias_mu * self.bias_mu - 1).sum()
        logq = - 0.5 * self.qzero_logvar.sum()
        logq -= logdets

        cw = torch.tanh(torch.matmul(self.rzero_c, weight.t()))

        mu_tilde = torch.mean(self.rzero_b1.ger(cw), dim=1)
        neg_log_var_tilde = torch.mean(self.rzero_b2.ger(cw), dim=1)

        z, logr = self.flow_r(z)

        z_mu_square = (z - mu_tilde) * (z - mu_tilde)
        logr += 0.5 * (- torch.exp(neg_log_var_tilde) * z_mu_square
                       + neg_log_var_tilde).sum()

        kldiv = kldiv_weight + kldiv_bias + logq - logr
        
        return kldiv
            
    def forward(self, input, kl=True):
        if kl:
            z, logdets = self.sample_z(kl=True)
        else:
            z = self.sample_z(kl=False)
        weight_std = torch.clamp(torch.exp(self.weight_logstd), 0, self.hparams.threshold_var)
        bias_std = torch.clamp(torch.exp(0.5 * self.bias_logvar), 0, self.hparams.threshold_var)
        weight_mu = z.view(1, -1) * self.weight_mu
        weight = weight_mu + weight_std * self.epsilon_weight
        bias = self.bias_mu + bias_std * self.epsilon_bias
        out = F.linear(input, weight, bias)
        if not kl:
            return out
        else:
            kldiv = self.kl_div(logdets, z, weight)
            return out, kldiv