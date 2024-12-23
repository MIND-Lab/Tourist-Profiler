import torch
import torch.nn as nn
import numpy as np
import pandas as pd


class Environment(nn.Module):
    def __init__(self, args, dataset, padding_number, rnn):
        super(Environment, self).__init__()
        self.args = args
        self.lists_full = dataset.lists_full
        self.lists_full_sep = dataset.lists_full_sep
        self.dataset_path = args.dataset_path
        self.n_users = dataset.n_users
        self.n_items = dataset.num_items
        self.n_categories = dataset.n_categories
        self.padding_number = padding_number
        self.device = args.device
        self.rnn = rnn
        self.import_data()
        self.one_hot_categories = self.get_one_hot_categories()
        

    def import_data(self):
        self.movies_data = pd.read_csv(f'{self.dataset_path}/preprocessed/movies_data.csv') # *
        self.data_tr = pd.read_csv(f'{self.dataset_path}/preprocessed/data_tr.csv') #!
        self.data_te = pd.read_csv(f'{self.dataset_path}/preprocessed/data_te.csv') #!
        self.data_full = pd.read_csv(f'{self.dataset_path}/preprocessed/data_full.csv') #*
        self.train_seq_base_data = pd.read_csv(f"{self.dataset_path}/preprocessed/train_seq_base_data.csv") #*
        self.test_seq_base_data = pd.read_csv(f"{self.dataset_path}/preprocessed/test_seq_base_data.csv") #*
        self.train_seq_target_data = pd.read_csv(f"{self.dataset_path}/preprocessed/train_seq_target_data.csv") #*
        self.test_seq_target_data = pd.read_csv(f"{self.dataset_path}/preprocessed/test_seq_target_data.csv") #*
        self.data_tr_sep = pd.read_csv(f'{self.dataset_path}/preprocessed/data_tr_sep.csv') #!
        self.data_te_sep = pd.read_csv(f'{self.dataset_path}/preprocessed/data_te_sep.csv') #!
        self.data_full_sep = pd.read_csv(f'{self.dataset_path}/preprocessed/data_full_sep.csv') #!
        self.train_sep_seq_base_data = pd.read_csv(f"{self.dataset_path}/preprocessed/train_sep_seq_base_data.csv") #*
        self.test_sep_seq_base_data = pd.read_csv(f"{self.dataset_path}/preprocessed/test_sep_seq_base_data.csv") #*
        self.train_sep_seq_target_data = pd.read_csv(f"{self.dataset_path}/preprocessed/train_sep_seq_target_data.csv") #*
        self.test_sep_seq_target_data = pd.read_csv(f"{self.dataset_path}/preprocessed/test_sep_seq_target_data.csv") #*

        if self.args.windowed_dataset == False:
            self.train_users = pd.read_csv(f'{self.dataset_path}/preprocessed/train_users.csv') #!
            self.test_users = pd.read_csv(f'{self.dataset_path}/preprocessed/test_users.csv') #!
            self.train_base_data = pd.read_csv(f"{self.dataset_path}/preprocessed/train_base_data.csv") #*
            self.test_base_data = pd.read_csv(f"{self.dataset_path}/preprocessed/test_base_data.csv") #*
            self.train_target_data = pd.read_csv(f"{self.dataset_path}/preprocessed/train_target_data.csv") #*
            self.test_target_data = pd.read_csv(f"{self.dataset_path}/preprocessed/test_target_data.csv") #*
            self.train_sep_users = pd.read_csv(f'{self.dataset_path}/preprocessed/train_sep_users.csv') #!
            self.test_sep_users = pd.read_csv(f'{self.dataset_path}/preprocessed/test_sep_users.csv') #!
            self.train_sep_base_data = pd.read_csv(f"{self.dataset_path}/preprocessed/train_sep_base_data.csv")   #*
            self.test_sep_base_data = pd.read_csv(f"{self.dataset_path}/preprocessed/test_sep_base_data.csv") #*
            self.train_sep_target_data = pd.read_csv(f"{self.dataset_path}/preprocessed/train_sep_target_data.csv") #*
            self.test_sep_target_data = pd.read_csv(f"{self.dataset_path}/preprocessed/test_sep_target_data.csv") #*
        else:
            self.train_users = pd.read_csv(f'{self.dataset_path}/preprocessed/train_win_users.csv') #!
            self.test_users = pd.read_csv(f'{self.dataset_path}/preprocessed/test_win_users.csv') #!
            self.train_base_data = pd.read_csv(f"{self.dataset_path}/preprocessed/train_win_base_data.csv") #*
            self.test_base_data = pd.read_csv(f"{self.dataset_path}/preprocessed/test_win_base_data.csv") #*
            self.train_target_data = pd.read_csv(f"{self.dataset_path}/preprocessed/train_win_target_data.csv") #*
            self.test_target_data = pd.read_csv(f"{self.dataset_path}/preprocessed/test_win_target_data.csv") #*
            self.train_sep_users = pd.read_csv(f'{self.dataset_path}/preprocessed/train_win_sep_users.csv') #!
            self.test_sep_users = pd.read_csv(f'{self.dataset_path}/preprocessed/test_win_sep_users.csv') #*
            self.train_sep_base_data = pd.read_csv(f"{self.dataset_path}/preprocessed/train_win_sep_base_data.csv") #!
            self.test_sep_base_data = pd.read_csv(f"{self.dataset_path}/preprocessed/test_win_sep_base_data.csv") #* 
            self.train_sep_target_data = pd.read_csv(f"{self.dataset_path}/preprocessed/train_win_sep_target_data.csv") #*
            self.test_sep_target_data = pd.read_csv(f"{self.dataset_path}/preprocessed/test_win_sep_target_data.csv") #*

        self.train_users = torch.tensor(self.train_users.values, dtype=torch.float32).to(self.device) #!
        self.test_users = torch.tensor(self.test_users.values, dtype=torch.float32).to(self.device) #!
        self.train_base_data = torch.tensor(self.train_base_data.values, dtype=torch.float32).to(self.device) #*
        self.test_base_data = torch.tensor(self.test_base_data.values, dtype=torch.float32).to(self.device) #*
        self.train_target_data = torch.tensor(self.train_target_data.values, dtype=torch.float32).to(self.device) #*
        self.test_target_data = torch.tensor(self.test_target_data.values, dtype=torch.float32).to(self.device) #*
        self.train_seq_base_data = torch.tensor(self.train_seq_base_data.values, dtype=torch.float32).to(self.device) #*
        self.test_seq_base_data = torch.tensor(self.test_seq_base_data.values, dtype=torch.float32).to(self.device) #*
        self.train_seq_target_data = torch.tensor(self.train_seq_target_data.values, dtype=torch.float32).to(self.device) #*
        self.test_seq_target_data = torch.tensor(self.test_seq_target_data.values, dtype=torch.float32).to(self.device) #*
        self.train_sep_users = torch.tensor(self.train_sep_users.values, dtype=torch.float32).to(self.device) #!
        self.test_sep_users = torch.tensor(self.test_sep_users.values, dtype=torch.float32).to(self.device) #!
        self.train_sep_base_data = torch.tensor(self.train_sep_base_data.values, dtype=torch.float32).to(self.device) #*
        self.test_sep_base_data = torch.tensor(self.test_sep_base_data.values, dtype=torch.float32).to(self.device) #*
        self.train_sep_target_data = torch.tensor(self.train_sep_target_data.values, dtype=torch.float32).to(self.device) #*
        self.test_sep_target_data = torch.tensor(self.test_sep_target_data.values, dtype=torch.float32).to(self.device) #*
        self.train_sep_seq_base_data = torch.tensor(self.train_sep_seq_base_data.values, dtype=torch.float32).to(self.device) #*
        self.test_sep_seq_base_data = torch.tensor(self.test_sep_seq_base_data.values, dtype=torch.float32).to(self.device) #*
        self.train_sep_seq_target_data = torch.tensor(self.train_sep_seq_target_data.values, dtype=torch.float32).to(self.device) #*
        self.test_sep_seq_target_data = torch.tensor(self.test_sep_seq_target_data.values, dtype=torch.float32).to(self.device) #*
        
        
    def get_past_history_embedding(self, user_id, item_id, window):
        category_one_hot = np.array(self.movies_data.values, dtype=np.int32)

        data_user = self.data_full[self.data_full["UserID"]==user_id]
        data_user = data_user.sort_values(by=["Timestamp", "MovieID"])

        # We add 1 to avoid a zero division.
        scaled_user_id = (user_id + 1) / self.n_users
        # ? scaled_user_id = user_id
        scaled_padding_number = self.padding_number / self.n_items

        history_embedding = []
        if self.args.sep_test_users:
            full_user_history = torch.tensor(self.lists_full_sep[user_id]).to(self.device)
        else:
            full_user_history = torch.tensor(self.lists_full[user_id]).to(self.device)
        
        if item_id in full_user_history: 
            # We get the index of the current item in the user history.		
            item_index = torch.where(full_user_history == item_id)
            item_index = item_index[0]
            # We get the user history before the current item. 
            curr_user_history = full_user_history[:item_index]
            # We limit the user history to the last items. We keep a number of items
            # that is equal to the window. We also consider the case where there are
            # less items than the window.
            curr_user_history = curr_user_history[-window:]
            for item in curr_user_history:
                data_item_rating = data_user[data_user["MovieID"]==int(item)]["Rating"].values
                if len(data_item_rating) == 0:
                    data_item_rating = 0
                else:
                    data_item_rating = data_item_rating[0] / 5.
                # We add 1 to avoid a zero division.
                #? scaled_item_id = (item + 1) / self.n_items
                scaled_item_id = 0.5
                boolean_vector_categories = category_one_hot[item][1:]
                # We create a vector that contains: [uid, iid, category one-hot, rating].
                new_one_hot = torch.cat((torch.tensor(boolean_vector_categories, dtype=torch.float32).to(self.device), torch.tensor([data_item_rating], dtype=torch.float32).to(self.device)), 0)
                history_embedding.append(new_one_hot)
        
        # We fill the window with padding.      
        while len(history_embedding) < window:
            history_embedding.insert(0, torch.cat((torch.zeros(self.n_categories, dtype=torch.float32).to(self.device),
                                                   torch.tensor([0], dtype=torch.float32).to(self.device)), 0))
        
        input_data = torch.vstack(history_embedding).to(self.device)
        input_data = torch.reshape(input_data, (1, len(history_embedding), self.n_categories + 1)).to(self.device)
        
        return input_data
        
    
    def get_current_history_embedding(self, user_id, item_id, window):
        category_one_hot = np.array(self.movies_data.values, dtype=np.int32)

        data_user = self.data_full[self.data_full["UserID"]==user_id]
        data_user = data_user.sort_values(by=["Timestamp", "MovieID"])

        # We add 1 to avoid a zero division.
        scaled_user_id = (user_id + 1) / self.n_users
        #? scaled_user_id = user_id
        scaled_padding_number = self.padding_number / self.n_items
        
        history_embedding = []
        if self.args.sep_test_users:
            full_user_history = torch.tensor(self.lists_full_sep[user_id]).to(self.device)
        else:
            full_user_history = torch.tensor(self.lists_full[user_id]).to(self.device)
        if item_id in full_user_history: 
            # We get the index of the current item in the user history.		
            item_index = torch.where(full_user_history == item_id)
            item_index = item_index[0]
            # We get the user history including the current item. 
            curr_user_history = full_user_history[:item_index+1]
            # We limit the user history to the last items. We keep a number of items
            # that is equal to the window. We also consider the case where there are
            # less items than the window.
            curr_user_history = curr_user_history[-window:]
            for item in curr_user_history:
                data_item_rating = data_user[data_user["MovieID"]==int(item)]["Rating"].values
                if len(data_item_rating) == 0:
                    data_item_rating = 0
                else:
                    data_item_rating = data_item_rating[0] / 5.
                # We add 1 to avoid a zero division.
                # ? scaled_item_id = (item + 1) / self.n_items
                scaled_item_id = 0.5
                boolean_vector_categories = category_one_hot[item][1:]
                # We create a vector that contains: [uid, iid, category one-hot, rating].
                new_one_hot = torch.cat((torch.tensor(boolean_vector_categories, dtype=torch.float32).to(self.device),
                                 torch.tensor([data_item_rating], dtype=torch.float32).to(self.device)), 0)
                history_embedding.append(new_one_hot)
        
        # We fill the window with padding.      
        while len(history_embedding) < window:            
            history_embedding.insert(0, torch.cat((torch.zeros(self.n_categories, dtype=torch.float32).to(self.device),
                                                   torch.tensor([0], dtype=torch.float32).to(self.device)), 0))
        
        input_data = torch.vstack(history_embedding).to(self.device)
        input_data = torch.reshape(input_data, (1, len(history_embedding), self.n_categories + 1)).to(self.device) 
        return input_data   
    
    
    def get_one_hot_categories(self):
        one_hot_categories = []
        item_id = self.padding_number
        for i in range(self.n_categories):
            one_hot_category = self.get_one_hot_category(item_id, category=i)
            one_hot_categories.append(one_hot_category)
        return one_hot_categories
        
    
    def get_one_hot_category(self, item_id, category=None):
        if category == None:
            # We get the item values from the dataset.
            item_values = torch.tensor(self.movies_data.loc[item_id].values).to(self.device)
            
            # We remove the item id.
            item_values = item_values[1:]
            
            # We get the indices of non-zero values.
            non_zero_values = torch.where(item_values != 0)[0]
            
            # We choose one category among the categories of the item.
            random_index = torch.randint(0, non_zero_values.size(0), (1,)).item()
            category = non_zero_values[random_index].item()
        
        # We build a one-hot encoding with the category.
        one_hot_category = torch.zeros(self.n_categories, dtype=torch.float32).to(self.device)
        one_hot_category[category] = 1
        one_hot_category = torch.unsqueeze(one_hot_category, 0).to(self.device)    
    
        return one_hot_category    
                
                
    def get_reward(self, target, action):
        # We add 1 since the action gives an index from 0 to 4, whereas we
        # need a rating between 1 and 5.
        prediction = action + 1
        reward = - torch.abs(target - prediction).to(self.device)
        return reward
        
    
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
    

    def get_states(self, batch_users, batch_items):
        # We build the input for the RNN with the user and item identifiers for
        # each test instance.
        batch_past_rnn_input, batch_curr_rnn_input = [], []
        for i in range(len(batch_users)):
            user_id = batch_users[i].item()
            item_id = batch_items[i].item()
            
            past_rnn_input = self.get_past_history_embedding(user_id, item_id, self.args.window).to(self.device)
            curr_rnn_input = self.get_current_history_embedding(user_id, item_id, self.args.window).to(self.device)
            past_rnn_input = past_rnn_input.squeeze(0)
            curr_rnn_input = curr_rnn_input.squeeze(0)
            batch_past_rnn_input.append(past_rnn_input)
            batch_curr_rnn_input.append(curr_rnn_input)
        batch_past_rnn_input = torch.stack(batch_past_rnn_input).to(self.device)
        batch_curr_rnn_input = torch.stack(batch_curr_rnn_input).to(self.device)

        # We compute the RNN output.
        batch_past_rnn_output = self.rnn(batch_past_rnn_input).to(self.device)
        batch_curr_rnn_output = self.rnn(batch_curr_rnn_input).to(self.device)
    
        # Now we build the second part of the state, that is the one-hot encoding
        # of the categories. 
        batch_categories = []
        for i in range(len(batch_users)):
            item_id = batch_items[i].item()
            one_hot_category = self.get_one_hot_category(item_id)
            batch_categories.append(one_hot_category)
        batch_categories = torch.vstack(batch_categories).to(self.device)

        # We create the state for the agent.
        batch_state = torch.cat((batch_past_rnn_output, batch_categories), dim=1)

        # We create the next state for the agent.
        batch_next_state = torch.cat((batch_curr_rnn_output, batch_categories), dim=1)

        return batch_state, batch_next_state, batch_categories, batch_past_rnn_output, batch_curr_rnn_output
    

    def get_repeated_states(self, batch_past_rnn_output):
        # We repeat the RNN output for the number of categories.
        # The first row of RNN output is repeated 18 times, then the second row
        # is repeated 18 times, and so on.
        batch_past_rnn_output = torch.repeat_interleave(batch_past_rnn_output, self.n_categories, dim=0)

        # We repeat the one-hot encodings of the categories for the number
        # of instances of the batch.
        batch_categories = []
        batch_n_elements = int(batch_past_rnn_output.shape[0] / self.n_categories)
        for i in range(self.n_categories):
            batch_categories.append(self.one_hot_categories[i])
        batch_categories = torch.vstack(batch_categories).to(self.device)
        batch_categories = batch_categories.repeat(batch_n_elements, 1)

        # We create the state for the agent.
        batch_state = torch.cat((batch_past_rnn_output, batch_categories), dim=1)

        return batch_state, batch_n_elements
