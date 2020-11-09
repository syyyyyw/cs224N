#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f
    def __init__(self,in_dim,out_dim,dropout_rate=0.5):
        super(Highway,self).__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.dropout_rate=dropout_rate
        self.W_p=nn.Linear(self.in_dim,self.out_dim)
        self.W_g=nn.Linear(self.in_dim,self.out_dim)
        self.Dropout=nn.Dropout(self.dropout_rate)

    def forward(self,x):
        x_proj=F.relu(self.W_p(x))
        x_gate=F.sigmoid(self.W_g(x))
        x_highway=x_gate*x_proj+(1-x_gate)*x
        x_word_embed=self.Dropout(x_highway)
        return x_word_embed


    ### END YOUR CODE

