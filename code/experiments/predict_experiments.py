import os
import sys
import json
import numpy as np
import pandas as pd
from keras.optimizers import Nadam, Adam
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
all_preds_han = []

indx_tg0 = [i for i in range(0, 18744)]
indx_tg1 = [i for i in range(18744, 37488)]
indx_tg2 = [i for i in range(37488, 56232)]
indx_tg3 = [i for i in range(56232,74976)]

idx = [indx_tg0, indx_tg1, indx_tg2, indx_tg3]
models = [make_model_tgt0, make_model_tgt1, make_model_tgt2, make_model_tgt3]


# = = = = = = = = = = = = = = = = = = =

# = = = = = Hyper-Parameters = = = = =

# = = = = = = = = = = = = = = = = = = =


# number of GRU units
n_units = [60, 60, 45, 60]
# mode by which outputs of the forward and backward RNNs will be combined.
merge_mode = ['sum', 'sum', 'concat', 'concat']
# prediction mode
drop_rate = 0.
drop_rate_emb = 0.
# whether to use cosine sim or not (unormalized dot product)
att_cosine = [False, False, True, False]
# whether to use a MLP for computing hidden attention state
use_dense_layer = [True, True, False, True]
# the activation function used by the MLP 
att_activation = ['tanh', 'sigmoid', None, 'tanh' ]
batch_size = [96, 96, 120, 96]
nb_epochs = 1
my_optimizer = [Adam(), Adam(), Nadam(), Adam()]
my_patience = 12


# = = = = = = = = = = = = = = = = = 

# = = = = = Data Loading = = = = =

# = = = = = = = = = = = = = = = = = 


docs = np.load(path_to_data + 'documents.npy')
docs_0 = np.load(path_to_data + 'documents_0.npy')
docs_3 = np.load(path_to_data + 'documents_3.npy')
dict_docs = {0 : docs_0, 1 : docs, 2 : docs, 3 : docs_3}

embeddings = np.load(path_to_data + 'embeddings.npy')

# Load test set
with open(path_to_data + 'test_idxs.txt', 'r') as file:
    test_idxs = file.read().splitlines()

test_idxs = [int(elt) for elt in test_idxs]


# = = = = = = = = = = = = = = = = = = = 

# = =  Convert txt file to csv file = = 

# = = = = = = = = = = = = = = = = = = = 

def from_txt_to_csv(file_name, folder_name):
    """
    Transform the output of the read_results_predict.py in a proper Kaggle Submission, i.e : a well formated csv file
    
    inputs : 
    - file_name is the name (string) of the txt file generated by the read_results_predict.py (without the .txt extension)
    - folder_name is the desired or existing name (string) of the folder where the submission will be stored into 
    
    output : None
    
    """
    path_root = os.path.dirname(os.path.abspath('').replace('\\', '/')) + "/data/"
    path_file = path_root  + file_name + ".txt"
    path_folder = path_root + folder_name
    # create a directory if it doesn't exist yet
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)
      
    df = pd.read_csv(path_file)
    df.to_csv(path_folder + "/" + file_name + ".csv", index=False)
    print("Submission saved in '{}'".format(path_folder))  
    

    
# = = = = = = = = = = = = = = = = = 

# = = = = Predict Targets = = = =

# = = = = = = = = = = = = = = = = = 

for tgt in range(4):
    data = dict_docs[tgt]
    docs_test = data[test_idxs, :, :]
    make_model = models[tgt]
    model =  make_model(n_units[tgt], merge_mode[tgt], drop_rate, drop_rate_emb, 
                   att_cosine[tgt], att_activation[tgt], use_dense_layer[tgt], 
                   embeddings, docs_test, is_GPU)


    model.load_weights(path_to_data + 'model_' + str(tgt))
    all_preds_han.append(model.predict(docs_test).tolist())
    
# flatten
all_preds_han = [elt[0] for sublist in all_preds_han for elt in sublist]


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

# = Format the predictions into proper Kaggle submission file =

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

for tgt in range(4):
    # write the predictions of a single target with the corrects indx
    with open(path_to_data + 'predictions_han_' + str(tgt) + '.txt', 'w') as file:
        if tgt == 0:
            file.write('id,pred\n')
        for id, pred in zip(idx[tgt], all_preds_han):
            pred = format(pred, '.7f')
            file.write(str(id) + ',' + pred + '\n')
            
            
            
filenames = [path_to_data + 'predictions_han_' + str(tgt) + '.txt' for tgt in range(4)]
with open(path_to_data + 'predictions_all.txt', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)

from_txt_to_csv('predictions_all', '')