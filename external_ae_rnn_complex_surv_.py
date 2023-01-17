# -*- coding: utf-8 -*-
"""External_AE-RNN-COMPLEX-SURV

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Sb1MzNgrqf9Fek_4LM1y-87bmKhu6oao
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install pytorch_lightning

import torch
import pytorch_lightning as pl
from torch import nn

class ExternalAE(pl.LightningModule):

  """
  @Author: Martin Pius
  ----------------------
  -This class implement the Inspired-Unsupervised Autoencoder for noise reduction

  Parameters:
  -----------
  input_dim: Int ==> dimension of the input data
  hidden_dim: Int ==> hidden dim for the FC net
  output_dim: Int ==> Embedding dimension

  Returns:
  ---------
  out: torch.Tensor--> Reconstructed input with shape [batch_size, input_dim]
  embedding: torch.Tensor--> The hidden representation with shape [batch_size, output_dim]

  """

  def __init__(self, input_dim,
               hidden_dim,
               output_dim):
    
    super(ExternalAE, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim

    self.encoder = nn.Sequential(
        nn.Linear(in_features = self.input_dim,
                  out_features = self.hidden_dim),
        nn.ReLU(),
        nn.Linear(in_features = self.hidden_dim,
                  out_features = self.output_dim),
        nn.ReLU())
    
    self.decoder = nn.Sequential(
        nn.Linear(in_features = self.output_dim,
                  out_features = self.hidden_dim),
        nn.ReLU(),
        nn.Linear(in_features = self.hidden_dim,
                  out_features = self.input_dim),
        nn.Tanh())
    
  def forward(self, input):
    embedding = self.encoder(input)
    out = self.decoder(embedding)
    return out, embedding

####---Sanity check for the denoising autoencoder---####
batch_size = 32
input_dim = 200
hidden_dim = 128
output_dim = 64
input = torch.randn(size = (batch_size, input_dim))
external_ae = ExternalAE(input_dim, hidden_dim, output_dim)
out, embedding = external_ae(input)
assert out.shape == (batch_size, input_dim)
assert embedding.shape == (batch_size, output_dim)