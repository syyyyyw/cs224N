#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ### YOUR CODE HERE for part 1h
        self.vocab=vocab
        self.word_embed_size=word_embed_size
        self.char_embed_size=50
        self.dropout_rate=0.3
        self.kernel_size=5
        self.padding=1
        self.embebding=nn.Embedding(len(self.vocab.char2id),self.char_embed_size,self.vocab.char_pad)
        self.cnn=CNN(self.char_embed_size,self.word_embed_size,self.kernel_size,self.padding)
        self.highway=Highway(self.word_embed_size,self.word_embed_size,self.dropout_rate)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character
            x_padded

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
            x_word_embedded
        """
        ### YOUR CODE HERE for part 1h
        x=self.embebding(input)  #(sentence_length,batch_size,max_word_length,char_embed_size)
        sent_len,batch_size,max_word_lenght,char_embeded_size=x.shape
        x_reshape=x.view(-1,max_word_lenght,char_embeded_size).transpose(1,2)
        x_conv_out=self.cnn(x_reshape)
        x_word_embed=self.highway(x_conv_out)
        x_word_embed=x_word_embed.view(sent_len,batch_size,-1)
        return x_word_embed

        ### END YOUR CODE

