import os
import sys
import json
import numpy as np
from keras.optimizers import Nadam, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



# = = = = = = = = = = = = = = = = = = =

# = = = Paths and custom imports = = = 

# = = = = = = = = = = = = = = = = = = =

path_root = os.path.dirname(os.path.abspath('').replace('\\', '/'))
path_to_data = path_root + '/data/'
path_to_code = path_root + '/code/'
path_to_experiments = path_to_code + '/experiments/'

sys.path.insert(0, path_root)
sys.path.insert(0, path_to_code)

for tgt in range(4):
    sys.path.insert(0, path_to_experiments + 'target_' + str(tgt))

# load the custom architectures for each target
from make_model_tgt0 import make_model_tgt0
from make_model_tgt1 import make_model_tgt1
from make_model_tgt2 import make_model_tgt2
from make_model_tgt3 import make_model_tgt3
# load the custom attention mechanism
from AttentionWithContextCustom import AttentionWithContext


# = = = = = = = = = = = = = = = = = = =

# = = = = = Global Variables = = = = =

# = = = = = = = = = = = = = = = = = = =


is_GPU = True
save_weights = True
save_history = True
models = [make_model_tgt0, make_model_tgt1, make_model_tgt2, make_model_tgt3]
np.random.seed(1997)


# = = = = = = = = = = = = = = = = = = =

# = = = = = Hyper-Parameters = = = = =

# = = = = = = = = = = = = = = = = = = =


# number of GRU units
n_units = [60, 60, 45, 60]
# mode by which outputs of the forward and backward RNNs will be combined.
merge_mode = ['sum', 'sum', 'concat', 'concat']
drop_rate = [0.1, 0.1, 0.5, 0.1] 
# droupout rate after embedding layer
drop_rate_emb = [None, None, 0.1, None]
# whether to use cosine sim or not (unormalized dot product)
att_cosine = [False, False, True, False]
# whether to use a MLP for computing hidden attention state
use_dense_layer = [True, True, False, True]
# the activation function used by the MLP 
att_activation = ['tanh', 'sigmoid', None, 'tanh' ]
batch_size = [96, 96, 120, 96]
nb_epochs = 10
my_optimizer = [Adam(), Adam(), Nadam(), Adam()]
my_patience = 4


# = = = = = = = = = = = = = = = = = 

# = = = = = Data Loading = = = = =

# = = = = = = = = = = = = = = = = = 

docs = np.load(path_to_data + 'documents.npy')
docs_0 = np.load(path_to_data + 'documents_0.npy')
docs_3 = np.load(path_to_data + 'documents_3.npy')
dict_docs = {0 : docs_0, 1 : docs, 2 : docs, 3 : docs_3}

embeddings = np.load(path_to_data + 'embeddings.npy')

with open(path_to_data + 'train_idxs.txt', 'r') as file:
    train_idxs = file.read().splitlines()
    
train_idxs = [int(elt) for elt in train_idxs]
    
# create validation set
idxs_select_train = np.random.choice(range(len(train_idxs)),size=int(len(train_idxs)*0.80),replace=False)
idxs_select_val = np.setdiff1d(range(len(train_idxs)),idxs_select_train)

train_idxs_new = [train_idxs[elt] for elt in idxs_select_train]
val_idxs = [train_idxs[elt] for elt in idxs_select_val]


# = = = = = = = = = = = = = = = = = = =

# = = = = = = = Training = = = = = = =

# = = = = = = = = = = = = = = = = = = =


for tgt in range(4):

    with open(path_to_data + 'targets/train/target_' + str(tgt) + '.txt', 'r') as file:
        target = file.read().splitlines()
    
    target_train = np.array([target[elt] for elt in idxs_select_train]).astype('float')
    target_val = np.array([target[elt] for elt in idxs_select_val]).astype('float')
    
    # training and validation set according to the correct target
    data = dict_docs[tgt]
    docs_train = data[train_idxs_new,:,:]
    docs_val = data[val_idxs,:,:]
    print('data loaded')

    # = = = = = defining architecture = = = = =
    
    make_model = models[tgt]
    model =  make_model(n_units[tgt], merge_mode[tgt], drop_rate[tgt], drop_rate_emb[tgt], 
                   att_cosine[tgt], att_activation[tgt], use_dense_layer[tgt], 
                   embeddings, docs_train, is_GPU)
    

    model.compile(loss='mean_squared_error',
                  optimizer=my_optimizer[tgt],
                  metrics=['mae'])

    print('model compiled')

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
        my_callbacks = [early_stopping,checkpointer]
    else:
        my_callbacks = [early_stopping]

    model.fit(docs_train, 
              target_train,
              batch_size = batch_size[tgt],
              epochs = nb_epochs,
              validation_data = (docs_val,target_val),
              callbacks = my_callbacks,
              verbose = 0)

    

    if save_history:
        hist = model.history.history
        with open(path_to_data + 'model_history_' + str(tgt) + '.json', 'w') as file:
            json.dump(hist, file, sort_keys=False, indent=4)

    print('* * * * * * * target', tgt, 'done * * * * * * *')    
    
