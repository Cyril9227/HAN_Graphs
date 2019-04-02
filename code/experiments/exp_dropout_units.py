import sys
import json
import numpy as np

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.models import Model
from keras.layers import Input, Embedding, Dropout, Bidirectional, GRU, CuDNNGRU, TimeDistributed, Dense

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# = = = = = = = = = = = = = = =


is_GPU = True
save_weights = True
save_history = False
np.random.seed(1997)

path_root = 'D:/Scolaire/Code/Python/Machine_Learning/Kaggle_Challenges_M2/Altegrad'
path_to_code = path_root + '/code/'
path_to_data = path_root + '/data/'

sys.path.insert(0, path_to_code)

# = = = = = = = = = = = = = = =

from AttentionWithContext import AttentionWithContext
from make_model import make_model


# = = = = = hyper-parameters = = = = =

n_units = 60
drop_rate = 0.1
batch_size = 96
nb_epochs = 15
my_optimizer = 'adam'
my_patience = 4

# = = = = = data loading = = = = =

docs = np.load(path_to_data + 'documents.npy')
embeddings = np.load(path_to_data + 'embeddings.npy')

with open(path_to_data + 'train_idxs.txt', 'r') as file:
    train_idxs = file.read().splitlines()
    
train_idxs = [int(elt) for elt in train_idxs]
    
# create validation set

idxs_select_train = np.random.choice(range(len(train_idxs)), size=int(len(train_idxs) * 0.80), replace=False)
idxs_select_val = np.setdiff1d(range(len(train_idxs)), idxs_select_train)

train_idxs_new = [train_idxs[elt] for elt in idxs_select_train]
val_idxs = [train_idxs[elt] for elt in idxs_select_val]

docs_train = docs[train_idxs_new, :, :]
docs_val = docs[val_idxs, :, :]

for tgt in range(4):

    with open(path_to_data + 'targets/train/target_' + str(tgt) + '.txt', 'r') as file:
        target = file.read().splitlines()
    
    target_train = np.array([target[elt] for elt in idxs_select_train]).astype('float')
    target_val = np.array([target[elt] for elt in idxs_select_val]).astype('float')

    # print('data loaded')
    
    model = make_model(n_units, drop_rate, embeddings, docs_train, is_GPU)
 
    model.compile(loss='mean_squared_error',
                  optimizer=my_optimizer,
                  metrics=['mae'])

    # print('model compiled')

    # = = = = = training = = = = =

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=my_patience,
                                   mode='min')

    # save model corresponding to best epoch
    checkpointer = ModelCheckpoint(filepath=path_to_data + 'model_' + str(tgt), 
                                   verbose=1, 
                                   save_best_only=True,
                                   save_weights_only=True)

    if save_weights:
        my_callbacks = [early_stopping, checkpointer]
    else:
        my_callbacks = [early_stopping]

    model.fit(docs_train, 
              target_train,
              batch_size = batch_size,
              epochs = nb_epochs,
              validation_data = (docs_val, target_val),
              callbacks = my_callbacks,
              verbose = 0)
    

    if save_history:
        hist = model.history.history
        with open(path_to_data + 'model_history_' + str(tgt) + '.json', 'w') as file:
            json.dump(hist, file, sort_keys=False, indent=4)

    print('* * * * * * * target', tgt, 'done * * * * * * *')    
    
