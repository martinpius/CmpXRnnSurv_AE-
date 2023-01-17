# -*- coding: utf-8 -*-
"""CoMpleX_surv_Attention.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wEdbs2B23nGQBnvfK735uXWwLjYwB6HR
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install pytorch-Lightning

import torch, time
from torch import nn, mm
import pytorch_lightning as pl
from argparse import ArgumentTypeError



class __AdditiveAttentionV1__(pl.LightningModule):
  def __init__(self, rnn_hidden_dim, rnn_out_dim, bidirectional = True):
    """
    @Author: Martin Pius
    This method compute the weights (prabalities of feature's contribution)

    Parameters:
    -----------
    rnn_hidden_dim: Fetched from the return of the 
    rnn_encoder (hidden) which has the shape of [batch_size, hidden_dim]
    rnn_out_dim: Fetched from the the return of the rnn_encoder
    (rnn_output) which has the shape [seq_len, batch_size, output_dim]:
    -----------
    """
    super().__init__()
    if bidirectional:
      rnn_out_dim = 2 * rnn_hidden_dim
    else: rnn_out_dim = rnn_hidden_dim
    self.attn = nn.Linear(rnn_hidden_dim + rnn_out_dim, rnn_hidden_dim)
    self.v = nn.Linear(rnn_hidden_dim, 1, bias = False)
  
  def forward(self, rnn_hidden, rnn_output):
    seq_len = rnn_output.shape[0] 
    rnn_hidden = rnn_hidden.unsqueeze(1) # new_shape == [batch_size, 1, hidden_dim]
    rnn_hidden1 = rnn_hidden.repeat(1, seq_len, 1) # new shape == [batch_size, seq_len, hidden_dim]
    rnn_output = rnn_output.permute(1, 0, 2) # new shape == [batch_size, seq_len, output_dim]
    # We combine the rnn_out and hidden to be the input to the linear layer
    att_input = torch.cat((rnn_hidden1, rnn_output), dim = 2) # shape == [batch_size, seq_len, output_dim + hidden_dim]
    alphas = torch.tanh(self.attn(att_input))
    attention = self.v(alphas).squeeze()
    return nn.functional.softmax(attention, dim = 1)

input_dim = 100
hidden_dim = 128
num_layers = 2
seq_len = 5
rnn_type = ["RNN", "GRU", "LSTM"]
batch_size = 32


#---*----*----* Model sanity check---*----*-----*
model = Encoder_RNN4Surv(
    input_dim,
    hidden_dim,
    num_layers,
    seq_len,
    rnn_type[0],
    bidirectional= True
) 

rnn_input = torch.randn(size = (batch_size, input_dim))
rnn_output, rnn_hidden = model(rnn_input)

rnn_output_dim = rnn_output.shape[2]
rnn_hidden_dim = rnn_hidden.shape[1]
attention = __AdditiveAttentionV1__(rnn_hidden_dim, 
                                    rnn_output_dim, 
                                    bidirectional = True)
attn = attention(rnn_hidden, rnn_output)
assert attn.shape == (batch_size, seq_len)

#---*----*----* Model sanity check---*----*-----* The decoder network

fc_hidden = 512
max_time_horizon = 100

# Testing the attention class:
batch_size = 32
seq_len = 5
rnn_hidden_dim = 128
rnn_out_dim = 256
rnn_output = torch.randn(size = (seq_len, batch_size, rnn_out_dim))
rnn_hidden = torch.randn(size = (batch_size, rnn_hidden_dim))
additive_attention = __AdditiveAttentionV1__(rnn_hidden_dim, rnn_out_dim)
attention = additive_attention(rnn_hidden, rnn_output)
assert attention.shape == (batch_size, seq_len)

