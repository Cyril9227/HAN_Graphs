import sys
import json
import numpy as np


from AttentionWithContext import AttentionWithContext

import keras.backend as K
 
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Embedding, Dropout, Bidirectional, GRU, CuDNNGRU, TimeDistributed, Dense, Flatten, Lambda, BatchNormalization

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

##############################################################################################################################
#
# Utility file to generate deep learning model : easier to train and then predict from same model
#
##############################################################################################################################

def bidir_gru(my_seq, n_units, is_GPU, drop_rate=0, rec_drop_rate=0):
    '''
    Just a convenient wrapper for bidirectional RNN with GRU units
    enables CUDA acceleration on GPU
    # regardless of whether training is done on GPU, model can be loaded on CPU
    # see: https://github.com/keras-team/keras/pull/9112
    '''
    if is_GPU:
        return Bidirectional(CuDNNGRU(units=n_units, return_sequences=True),
                             merge_mode='concat', 
                             weights=None)(my_seq)
    else:
        return Bidirectional(GRU(units=n_units,
                                 activation='tanh', 
                                 dropout=drop_rate,
                                 recurrent_dropout=rec_drop_rate,
                                 implementation=1,
                                 return_sequences=True,
                                 reset_after=True,
                                 recurrent_activation='sigmoid'),
                                 merge_mode='concat', weights=None)(my_seq)


def make_model(n_units, drop_rate, embeddings, docs_train, is_GPU):
    
    '''
    Convenient wrapper for generating same model for training and inference 
    
    n_units : number of units in bidirectional GRU layer
    drop_rate : dropout rate (set to 0 at inference time)
    embeddings : embedding matrix
    docs_train : training documents
    is_GPU : boolean, wether we're using gpu or not
    
    '''
    
    sent_ints = Input(shape=(docs_train.shape[2], ))

    sent_wv = Embedding(input_dim=embeddings.shape[0],
                        output_dim=embeddings.shape[1],
                        weights=[embeddings],
                        input_length=docs_train.shape[2],
                        trainable=False,
                        )(sent_ints)
    
    sent_wv = Dropout(0.1)(sent_wv)
    sent_wa = bidir_gru(sent_wv, n_units, is_GPU)
    sent_wa = BatchNormalization()(sent_wa)
    sent_att_vec = AttentionWithContext()(sent_wa)
    sent_att_vec_dr = Dropout(drop_rate)(sent_att_vec) 
    sent_encoder = Model(sent_ints, sent_att_vec_dr)

    doc_ints = Input(shape=(docs_train.shape[1], docs_train.shape[2], ))
    sent_att_vecs_dr = TimeDistributed(sent_encoder)(doc_ints)
    doc_sa = bidir_gru(sent_att_vecs_dr, n_units, is_GPU)
    doc_sa = BatchNormalization()(doc_sa)
    doc_att_vec = AttentionWithContext()(doc_sa)
    doc_att_vec_dr = Dropout(drop_rate)(doc_att_vec)
    preds = Dense(units=1)(doc_att_vec_dr)
    model = Model(doc_ints, preds)
   
    return model
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    