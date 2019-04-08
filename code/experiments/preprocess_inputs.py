from preprocessing_biased import create_documents
tgt_hyperparams = [
        (160, 'uniform', (8, 11), 8, 2, 2),
        (80, 'uniform', (8, 11), 5, 2, 0.5),
        (100, 'uniform', (4, 13), 5, 1, 2),
        (100, 'uniform', (4, 13), 5, 1, 2)]
        

for i in range(4):
    create_documents(i, *tgt_hyperparams[i])