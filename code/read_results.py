import json
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# = = = = = = = = = = = = = = =

is_GPU = True
path_root = os.path.dirname(os.path.abspath('').replace('\\', '/'))
path_to_data = path_root + '/data/'

docs = np.load(path_to_data + 'documents.npy')
embeddings = np.load(path_to_data + 'embeddings.npy')

with open(path_to_data + 'train_idxs.txt', 'r') as file:
    train_idxs = file.read().splitlines()

with open(path_to_data + 'test_idxs.txt', 'r') as file:
    test_idxs = file.read().splitlines()

train_idxs = [int(elt) for elt in train_idxs]
test_idxs = [int(elt) for elt in test_idxs]

docs_test = docs[test_idxs, :, :]

grid = (2, 2)
fig = plt.figure(figsize=(8 * grid[1], 5 * grid[0]))
axs = []

for tgt in range(4):

    with open(path_to_data + '/model_history_' + str(tgt) + '.json', 'r') as file:
        hist = json.load(file)

    val_mse = hist['val_loss']
    val_mae = hist['val_mean_absolute_error']

    min_val_mse = min(val_mse)
    min_val_mae = min(val_mae)

    best_epoch = val_mse.index(min_val_mse)
    
    axs.append(plt.subplot2grid(grid, (int(tgt / 2), int(tgt % 2)), fig=fig))
    axs[-1].set_title('Target ' + str(tgt + 1) +  
       ' (MSE: ' + str(round(min_val_mse, 3)) + 
       ', MAE: ' + str(round(min_val_mae, 3)) + ')', fontsize=15)
    axs[-1].title.set_position([0.5, 1.02])
    axs[-1].plot(hist['loss'])
    axs[-1].plot(hist['val_loss'], linestyle='dashed')
    axs[-1].set_ylabel('loss')
    axs[-1].set_xlabel('epoch (best ' + str(best_epoch) + ')')
    axs[-1].legend(['train', 'validation'])

plt.subplots_adjust(hspace=0.3)
plt.show()