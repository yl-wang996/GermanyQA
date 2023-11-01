import os
class Config():

    ### path
    train_path = 'GermanQuAD/GermanQuAD_train.json'
    test_path = 'GermanQuAD/GermanQuAD_test.json'
    logdir = './logs'
    model_dir = '/data/1wang/models'

    # check for existing
    for pth in [logdir,model_dir]:
        if not os.path.exists(pth):
            os.mkdir(pth)
    ### END path

    #### preprocessing
    truncation = True
    padding = True
    ### END preprocessing


    # other
    display_freq = 100
    batch_size = 6
    epochs = 1000
    seed = 666
    operation = 'prediction'  # prediction   train
    ### END training

    ### model
    from_scratch = False
    tokenizer = 'bert-base-german-cased'
    model = 'bert-base-german-cased'
    lr = 1e-5
    ### END model



