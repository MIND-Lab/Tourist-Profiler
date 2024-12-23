import torch
import torch.nn as nn
import pandas as pd


class Environment(nn.Module):
    def __init__(self, args, dataset, padding_number):
        super(Environment, self).__init__()
        self.args = args
        self.lists_full = dataset.lists_full
        self.dataset_path = args.dataset_path
        self.n_categories = dataset.n_categories
        self.padding_number = padding_number
        self.device = args.device
        self.import_data()


    def import_data(self):
        self.movies_data = pd.read_csv(f'{self.dataset_path}/preprocessed/movies_data.csv')
        self.data_tr = pd.read_csv(f'{self.dataset_path}/preprocessed/data_tr.csv')
        self.data_te = pd.read_csv(f'{self.dataset_path}/preprocessed/data_te.csv')
        self.data_full = pd.read_csv(f'{self.dataset_path}/preprocessed/data_full.csv')
        self.train_seq_base_data = pd.read_csv(f"{self.dataset_path}/preprocessed/train_seq_base_data.csv")
        self.test_seq_base_data = pd.read_csv(f"{self.dataset_path}/preprocessed/test_seq_base_data.csv")
        self.train_seq_target_data = pd.read_csv(f"{self.dataset_path}/preprocessed/train_seq_target_data.csv")
        self.test_seq_target_data = pd.read_csv(f"{self.dataset_path}/preprocessed/test_seq_target_data.csv")
        self.data_tr_sep = pd.read_csv(f'{self.dataset_path}/preprocessed/data_tr_sep.csv')
        self.data_te_sep = pd.read_csv(f'{self.dataset_path}/preprocessed/data_te_sep.csv')
        self.data_full_sep = pd.read_csv(f'{self.dataset_path}/preprocessed/data_full_sep.csv')
        self.train_sep_seq_base_data = pd.read_csv(f"{self.dataset_path}/preprocessed/train_sep_seq_base_data.csv")
        self.test_sep_seq_base_data = pd.read_csv(f"{self.dataset_path}/preprocessed/test_sep_seq_base_data.csv")
        self.train_sep_seq_target_data = pd.read_csv(f"{self.dataset_path}/preprocessed/train_sep_seq_target_data.csv")
        self.test_sep_seq_target_data = pd.read_csv(f"{self.dataset_path}/preprocessed/test_sep_seq_target_data.csv")

        if self.args.windowed_dataset == False:
            self.train_users = pd.read_csv(f'{self.dataset_path}/preprocessed/train_users.csv')
            self.test_users = pd.read_csv(f'{self.dataset_path}/preprocessed/test_users.csv')
            self.train_base_data = pd.read_csv(f"{self.dataset_path}/preprocessed/train_base_data.csv")
            self.test_base_data = pd.read_csv(f"{self.dataset_path}/preprocessed/test_base_data.csv")
            self.train_target_data = pd.read_csv(f"{self.dataset_path}/preprocessed/train_target_data.csv")
            self.test_target_data = pd.read_csv(f"{self.dataset_path}/preprocessed/test_target_data.csv")
            self.train_sep_users = pd.read_csv(f'{self.dataset_path}/preprocessed/train_sep_users.csv')
            self.test_sep_users = pd.read_csv(f'{self.dataset_path}/preprocessed/test_sep_users.csv')
            self.train_sep_base_data = pd.read_csv(f"{self.dataset_path}/preprocessed/train_sep_base_data.csv")
            self.test_sep_base_data = pd.read_csv(f"{self.dataset_path}/preprocessed/test_sep_base_data.csv")
            self.train_sep_target_data = pd.read_csv(f"{self.dataset_path}/preprocessed/train_sep_target_data.csv")
            self.test_sep_target_data = pd.read_csv(f"{self.dataset_path}/preprocessed/test_sep_target_data.csv")
        else:
            self.train_users = pd.read_csv(f'{self.dataset_path}/preprocessed/train_win_users.csv')
            self.test_users = pd.read_csv(f'{self.dataset_path}/preprocessed/test_win_users.csv')
            self.train_base_data = pd.read_csv(f"{self.dataset_path}/preprocessed/train_win_base_data.csv")
            self.test_base_data = pd.read_csv(f"{self.dataset_path}/preprocessed/test_win_base_data.csv")
            self.train_target_data = pd.read_csv(f"{self.dataset_path}/preprocessed/train_win_target_data.csv")
            self.test_target_data = pd.read_csv(f"{self.dataset_path}/preprocessed/test_win_target_data.csv")
            self.train_sep_users = pd.read_csv(f'{self.dataset_path}/preprocessed/train_win_sep_users.csv')
            self.test_sep_users = pd.read_csv(f'{self.dataset_path}/preprocessed/test_win_sep_users.csv')
            self.train_sep_base_data = pd.read_csv(f"{self.dataset_path}/preprocessed/train_win_sep_base_data.csv")
            self.test_sep_base_data = pd.read_csv(f"{self.dataset_path}/preprocessed/test_win_sep_base_data.csv")
            self.train_sep_target_data = pd.read_csv(f"{self.dataset_path}/preprocessed/train_win_sep_target_data.csv")
            self.test_sep_target_data = pd.read_csv(f"{self.dataset_path}/preprocessed/test_win_sep_target_data.csv")

        self.train_users = torch.tensor(self.train_users.values, dtype=torch.float32).to(self.device)
        self.test_users = torch.tensor(self.test_users.values, dtype=torch.float32).to(self.device)
        self.train_base_data = torch.tensor(self.train_base_data.values, dtype=torch.float32).to(self.device)
        self.test_base_data = torch.tensor(self.test_base_data.values, dtype=torch.float32).to(self.device)
        self.train_target_data = torch.tensor(self.train_target_data.values, dtype=torch.float32).to(self.device)
        self.test_target_data = torch.tensor(self.test_target_data.values, dtype=torch.float32).to(self.device)
        self.train_seq_base_data = torch.tensor(self.train_seq_base_data.values, dtype=torch.float32).to(self.device)
        self.test_seq_base_data = torch.tensor(self.test_seq_base_data.values, dtype=torch.float32).to(self.device)
        self.train_seq_target_data = torch.tensor(self.train_seq_target_data.values, dtype=torch.float32).to(self.device)
        self.test_seq_target_data = torch.tensor(self.test_seq_target_data.values, dtype=torch.float32).to(self.device)
        self.train_sep_users = torch.tensor(self.train_sep_users.values, dtype=torch.float32).to(self.device)
        self.test_sep_users = torch.tensor(self.test_sep_users.values, dtype=torch.float32).to(self.device)
        self.train_sep_base_data = torch.tensor(self.train_sep_base_data.values, dtype=torch.float32).to(self.device)
        self.test_sep_base_data = torch.tensor(self.test_sep_base_data.values, dtype=torch.float32).to(self.device)
        self.train_sep_target_data = torch.tensor(self.train_sep_target_data.values, dtype=torch.float32).to(self.device)
        self.test_sep_target_data = torch.tensor(self.test_sep_target_data.values, dtype=torch.float32).to(self.device)
        self.train_sep_seq_base_data = torch.tensor(self.train_sep_seq_base_data.values, dtype=torch.float32).to(self.device)
        self.test_sep_seq_base_data = torch.tensor(self.test_sep_seq_base_data.values, dtype=torch.float32).to(self.device)
        self.train_sep_seq_target_data = torch.tensor(self.train_sep_seq_target_data.values, dtype=torch.float32).to(self.device)
        self.test_sep_seq_target_data = torch.tensor(self.test_sep_seq_target_data.values, dtype=torch.float32).to(self.device)
    
    
    def mask_multivae_data(self, original_data):
        masked_original_data = []
        mask_matrix = []
        
        for i in range(len(original_data)):
            original_vector = original_data[i]
            nonzero_idxs = torch.nonzero(original_vector != 0).view(-1)
            mask = torch.full_like(original_vector, False)
            if len(nonzero_idxs) >= 4:
                idxs = torch.randperm(len(nonzero_idxs))[:3]
                mask[nonzero_idxs[idxs]] = True
            mask = mask.to(dtype=torch.bool)
            new_original_vector = torch.where(mask, torch.tensor(0, dtype=torch.long), original_vector)
            masked_original_data.append(new_original_vector)
            mask_matrix.append(mask)
        
        masked_original_data = torch.vstack(masked_original_data).to(self.device)
        mask_matrix = torch.vstack(mask_matrix).to(self.device)
            
        return masked_original_data, mask_matrix
    
    
    def mask_data(self, current_data, next_data):
        mask_matrix = []
        
        for i in range(len(current_data)):
            current_vector = current_data[i]
            next_vector = next_data[i]
            mask = torch.where((current_vector == 0) & (next_vector != 0))[0]
            boolean_vector = torch.full_like(current_vector, False)
            boolean_vector[mask] = True
            boolean_vector = boolean_vector.to(dtype=torch.bool)
            mask_matrix.append(boolean_vector)
        
        mask_matrix = torch.vstack(mask_matrix).to(self.device)
            
        return current_data, mask_matrix
        
                   
                
                        
            