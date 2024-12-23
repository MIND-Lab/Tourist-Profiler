import torch

class Setting():
    def __init__(self):
        ## Hyper-parameters about the dataset.
        self.dataset_path = '../dataset/movielens-100k'
        self.batch_size = 256
        self.windowed_dataset = True
        self.fast_testing = True
        self.device = torch.device('cpu')
