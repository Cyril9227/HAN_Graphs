import sys
import json
import numpy as np

from keras.models import Model
from keras.layers import Input, Embedding, Dropout, Bidirectional, GRU, CuDNNGRU, TimeDistributed, Dense

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# = = = = = = = = = = = = = = =

is_GPU = True
save_hist = False
path_root = os.path.dirname(os.path.abspath('').replace('\\', '/'))
path_to_data = path_root + '/data/'
path_to_code = path_root + '/code/'
sys.path.insert(0, path_to_code)

# = = = = = = = = = = = = = = =

from AttentionWithContext import AttentionWithContext
from make_model_layers import make_model
# = = = = = = = = = = = = = = =

docs = np.load(path_to_data + 'documents.npy')
embeddings = np.load(path_to_data + 'embeddings.npy')

with open(path_to_data + 'train_idxs.txt', 'r') as file:
    train_idxs = file.read().splitlines()

with open(path_to_data + 'test_idxs.txt', 'r') as file:
    test_idxs = file.read().splitlines()

train_idxs = [int(elt) for elt in train_idxs]
test_idxs = [int(elt) for elt in test_idxs]

docs_test = docs[test_idxs,:,:]

# = = = = = TRAINING RESULTS = = = = = 

if save_hist:

    for tgt in range(4):

        print('* * * * * * target number : ', tgt, '* * * * * * *')

        with open(path_to_data + '/model_history_' + str(tgt) + '.json', 'r') as file:
            hist = json.load(file)

        val_mse = hist['val_loss']
        val_mae = hist['val_mean_absolute_error']

        min_val_mse = min(val_mse)
        min_val_mae = min(val_mae)

        best_epoch = val_mse.index(min_val_mse) + 1

        print('best epoch:', best_epoch)
        print('best val MSE', round(min_val_mse,3))
        print('best val MAE', round(min_val_mae,3))

# = = = = = PREDICTIONS = = = = =     


all_preds_han = []

for tgt in range(4):
    
    # * * * mean baseline * * * 
    
    with open(path_to_data + 'targets/train/target_' + str(tgt) + '.txt', 'r') as file:
        target = file.read().splitlines()
    
    target = np.array(target).astype('float')
    
    # * * * HAN * * * 
    
    # relevant hyper-parameters
    n_units = 60
    drop_rate = 0 # prediction mode
 
    model = make_model(n_units, drop_rate, embeddings, docs_test, is_GPU)
    
    model.load_weights(path_to_data + 'model_' + str(tgt))
    all_preds_han.append(model.predict(docs_test).tolist())

# flatten
all_preds_han = [elt[0] for sublist in all_preds_han for elt in sublist]


with open(path_to_data + 'predictions_han.txt', 'w') as file:
    file.write('id,pred\n')
    for idx, pred in enumerate(all_preds_han):
        pred = format(pred, '.7f')
        file.write(str(idx) + ',' + pred + '\n')
