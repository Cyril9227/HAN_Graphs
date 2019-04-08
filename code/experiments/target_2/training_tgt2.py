import sys
import json
import numpy as np

from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tgt = 2

is_GPU = True
save_hist = False
save_weights = True
path_root = os.path.dirname(os.path.abspath('').replace('\\', '/'))
path_to_data = path_root + '/data/'
path_to_code = path_root + '/code/experiments/target_' + str(tgt) + '/'
sys.path.insert(0, path_to_code)

from make_model_tgt2 import make_model

# = = = = = hyper-parameters = = = = =


n_units = 30
drop_rate = 0.5
batch_size = 120
nb_epochs = 60
my_optimizer = Nadam()
my_patience = 6

# = = = = = data loading = = = = =

# Use baseline sampled documents
docs = np.load(path_to_data + 'documents.npy')
embeddings = np.load(path_to_data + 'embeddings.npy')

with open(path_to_data + 'train_idxs.txt', 'r') as file:
    train_idxs = file.read().splitlines()
    
train_idxs = [int(elt) for elt in train_idxs]

idxs_select_train = np.random.choice(range(len(train_idxs)), size=int(len(train_idxs) * 0.80), replace=False)
idxs_select_val = np.setdiff1d(range(len(train_idxs)), idxs_select_train)

train_idxs_new = [train_idxs[elt] for elt in idxs_select_train]
val_idxs = [train_idxs[elt] for elt in idxs_select_val]

docs_train = docs[train_idxs_new, :, :]
docs_val = docs[val_idxs, :, :]


with open(path_to_data + 'targets/train/target_' + str(tgt) + '.txt', 'r') as file:
    target = file.read().splitlines()
    
target_train = np.array([target[elt] for elt in idxs_select_train]).astype('float')
target_val = np.array([target[elt] for elt in idxs_select_val]).astype('float')

model = make_model(n_units, drop_rate, embeddings, docs_train, is_GPU)
print(model.summary())
 
model.compile(loss='mean_squared_error',
                  optimizer=my_optimizer,
                  metrics=['mse'])

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
              callbacks = my_callbacks)
    

if save_hist:
    hist = model.history.history
    with open(path_to_data + 'model_history_' + str(tgt) + '.json', 'w') as file:
        json.dump(hist, file, sort_keys=False, indent=4)

print('* * * * * * * target', tgt, 'done * * * * * * *')    
