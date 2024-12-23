import torch

class Setting():
    def __init__(self):
        self.dataset_path = './movielens-100k'
        self.batch_size = 256
        self.window = 10
        self.fast_testing = True
        self.device = torch.device('cpu')
