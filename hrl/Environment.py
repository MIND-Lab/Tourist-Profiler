import os
import sys
import copy
import torch
import numpy as np
import pandas as pd

sys.path.append('.')
sys.path.append('../../')
sys.path.append('../profile_models/')

notebook_path = os.getcwd()
project_path = os.path.abspath(os.path.join(notebook_path, '..'))
sys.path.append(project_path)

from profile_models.utils.models import MatrixFactorization, MultiVAE
from profile_models.utils import metrics

'''
Code adapted from: https://github.com/jerryhao66/HRL
'''


class Environment():
    def __init__(self, args, dataset):
        self.gamma = 0.5
        self.args = args
        self.dataset = dataset
        self.import_data()


    def initilize_state(self, recommender, traindata, testdata, high_state_size, low_state_size, padding_number):
        self.high_state_size = high_state_size
        self.low_state_size = low_state_size
        self.padding_number = padding_number
        self.item_embedding_user, self.item_embedding_item = recommender.get_item_embedding()
        self.origin_train_rewards = recommender.get_rewards(traindata)
        self.origin_test_rewards = recommender.get_rewards(testdata)
        self.embedding_size = len(self.item_embedding_user[0])  
        self.set_train_original_rewards()
        self.profile_model = self.get_profile_model()
        self.n_categories = self.dataset.n_categories

        
    def import_data(self):
        if self.args.sep_test_users:
            self.data_full = pd.read_csv('../dataset/movielens-100k/preprocessed/data_full_sep.csv')
            self.movies_data = pd.read_csv('../dataset/movielens-100k/preprocessed/movies_data.csv')
            if self.args.windowed_dataset == False:
                self.train_users = pd.read_csv('../dataset/movielens-100k/preprocessed/train_sep_users.csv')
                self.test_users = pd.read_csv('../dataset/movielens-100k/preprocessed/test_sep_users.csv')
                self.train_base_data = pd.read_csv("../dataset/movielens-100k/preprocessed/train_sep_base_data.csv")
                self.test_base_data = pd.read_csv("../dataset/movielens-100k/preprocessed/test_sep_base_data.csv")
                self.train_target_data = pd.read_csv("../dataset/movielens-100k/preprocessed/train_sep_target_data.csv")
                self.test_target_data = pd.read_csv("../dataset/movielens-100k/preprocessed/test_sep_target_data.csv")
            else:
                self.train_users = pd.read_csv('../dataset/movielens-100k/preprocessed/train_win_sep_users.csv')
                self.test_users = pd.read_csv('../dataset/movielens-100k/preprocessed/test_win_sep_users.csv')
                self.train_base_data = pd.read_csv("../dataset/movielens-100k/preprocessed/train_win_sep_base_data.csv")
                self.test_base_data = pd.read_csv("../dataset/movielens-100k/preprocessed/test_win_sep_base_data.csv")
                self.train_target_data = pd.read_csv("../dataset/movielens-100k/preprocessed/train_win_sep_target_data.csv")
                self.test_target_data = pd.read_csv("../dataset/movielens-100k/preprocessed/test_win_sep_target_data.csv")
            self.train_users = np.array(self.train_users.values)
            self.test_users = np.array(self.test_users.values)
            self.train_base_data = np.array(self.train_base_data.values)
            self.test_base_data = np.array(self.test_base_data.values)
            self.train_target_data = np.array(self.train_target_data.values)
            self.test_target_data = np.array(self.test_target_data.values)
        else:
            self.data_full = pd.read_csv('../dataset/movielens-100k/preprocessed/data_full.csv')
            self.movies_data = pd.read_csv('../dataset/movielens-100k/preprocessed/movies_data.csv')
            if self.args.windowed_dataset == False:
                self.train_users = pd.read_csv('../dataset/movielens-100k/preprocessed/train_users.csv')
                self.test_users = pd.read_csv('../dataset/movielens-100k/preprocessed/test_users.csv')
                self.train_base_data = pd.read_csv("../dataset/movielens-100k/preprocessed/train_base_data.csv")
                self.test_base_data = pd.read_csv("../dataset/movielens-100k/preprocessed/test_base_data.csv")
                self.train_target_data = pd.read_csv("../dataset/movielens-100k/preprocessed/train_target_data.csv")
                self.test_target_data = pd.read_csv("../dataset/movielens-100k/preprocessed/test_target_data.csv")
            else:
                self.train_users = pd.read_csv('../dataset/movielens-100k/preprocessed/train_win_users.csv')
                self.test_users = pd.read_csv('../dataset/movielens-100k/preprocessed/test_win_users.csv')
                self.train_base_data = pd.read_csv("../dataset/movielens-100k/preprocessed/train_win_base_data.csv")
                self.test_base_data = pd.read_csv("../dataset/movielens-100k/preprocessed/test_win_base_data.csv")
                self.train_target_data = pd.read_csv("../dataset/movielens-100k/preprocessed/train_win_target_data.csv")
                self.test_target_data = pd.read_csv("../dataset/movielens-100k/preprocessed/test_win_target_data.csv")
            self.train_users = np.array(self.train_users.values)
            self.test_users = np.array(self.test_users.values)
            self.train_base_data = np.array(self.train_base_data.values)
            self.test_base_data = np.array(self.test_base_data.values)
            self.train_target_data = np.array(self.train_target_data.values)
            self.test_target_data = np.array(self.test_target_data.values)
        self.train_base_profiles = self.get_train_base_profiles()
        self.train_target_profiles = self.get_train_target_profiles()
        self.compute_mask_train_data()
        self.compute_mask_test_data()
    
        
    def get_profile_model(self):
        if self.args.null_values_model == "multivae":
            p_dims = [8, 16, self.dataset.n_categories]
            model = MultiVAE(p_dims)
            if self.args.windowed_dataset == False:
                if self.args.sep_test_users:
                    with open('../profile_models/models/model_multivae_predict_sep.pt', 'rb') as file:
                        model.load_state_dict(torch.load(file))
                else:
                    with open('../profile_models/models/model_multivae_predict.pt', 'rb') as file:
                        model.load_state_dict(torch.load(file))
            else:
                if self.args.sep_test_users:
                    with open('../profile_models/models/model_multivae_predict_win_sep.pt', 'rb') as file:
                        model.load_state_dict(torch.load(file))
                else:
                    with open('../profile_models/models/model_multivae_predict_win.pt', 'rb') as file:
                        model.load_state_dict(torch.load(file))
        else:
            model = MatrixFactorization(self.dataset.num_users, self.dataset.n_categories)
            if self.args.windowed_dataset == False:
                if self.args.sep_test_users:
                    with open('../profile_models/models/model_mf_predict_sep.pt', 'rb') as file:
                        model.load_state_dict(torch.load(file))
                else:
                    with open('../profile_models/models/model_mf_predict.pt', 'rb') as file:
                        model.load_state_dict(torch.load(file))
            else:
                if self.args.sep_test_users:
                    with open('../profile_models/models/model_mf_predict_win_sep.pt', 'rb') as file:
                        model.load_state_dict(torch.load(file))
                else:
                    with open('../profile_models/models/model_mf_predict_win.pt', 'rb') as file:
                        model.load_state_dict(torch.load(file))
        return model
    
    
    def get_train_base_profiles(self):
        train_base_profiles = np.zeros((self.dataset.num_users, self.dataset.n_categories))
        for i in range(len(self.train_users)):
            # We find the user of the current item.
            user_id = int(self.train_users[i][0].item())
            # We add the past profile of the user to the train data.
            train_base_profiles[user_id] = self.train_base_data[i]
        return train_base_profiles
    
    
    def get_train_target_profiles(self):
        train_target_profiles = np.zeros((self.dataset.num_users, self.dataset.n_categories))
        for i in range(len(self.train_users)):
            # We find the user of the current item.
            user_id = int(self.train_users[i][0].item())
            # We add the past profile of the user to the train data.
            train_target_profiles[user_id] = self.train_target_data[i]
        return train_target_profiles
    
    
    def compute_mask_train_data(self):
        train_base_data, masked_train_ref = self.mask_data(self.train_base_data, self.train_target_data)
        new_masked_train_ref = np.zeros((self.dataset.num_users, self.dataset.n_categories))
        for i in range(len(self.train_users)):
            # We find the user of the current item.
            user_id = int(self.train_users[i][0].item())
            # We add the masked ref to the row of the specific user in the train set,
            # since each user appears only once in the train set.
            new_masked_train_ref[user_id] = masked_train_ref[i]

        self.masked_train_ref = new_masked_train_ref
    
    
    def compute_mask_test_data(self):
        test_base_data, masked_test_ref = self.mask_data(self.test_base_data, self.test_target_data)
        self.masked_test_ref = masked_test_ref
        
    
    def set_train_original_rewards(self):
        self.origin_rewards = self.origin_train_rewards


    def set_test_original_rewards(self):
        self.origin_rewards = self.origin_test_rewards

    
    def reset_state(self, user_input, num_idx, item_input, labels, batch_size, max_item_num, batch_index):
        self.user_input = user_input
        self.num_idx = num_idx
        self.item_input = np.reshape(item_input, (-1,))
        self.labels = labels
        self.batch_size = batch_size
        self.max_item_num = max_item_num
        self.batch_index = batch_index

        self.origin_prob = np.zeros((self.batch_size, 1), dtype=np.float32)

        self.dot_product_sum = np.zeros((self.batch_size, 1), dtype=np.float32)
        self.dot_product_mean = np.zeros((self.batch_size, 1), dtype=np.float32)

        self.element_wise_mean = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.element_wise_sum = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)

        self.vector_sum = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.vector_mean = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.num_selected = np.zeros(self.batch_size, dtype=np.int)
        self.action_matrix = np.zeros((self.batch_size, self.max_item_num), dtype=np.int)
        self.state_matrix = np.zeros((self.batch_size, self.max_item_num, self.low_state_size), dtype=np.float32)
        self.selected_input = np.full((self.batch_size, self.max_item_num), self.padding_number)


    def get_overall_state(self):

        def _mask(i):
            return [True]*i[0] + [False]*(self.max_item_num - i[0])

        origin_prob = np.reshape(self.origin_rewards[self.batch_index], (-1, 1)) #(batch_size, 1)
        self.num_idx = np.reshape(self.num_idx, (-1,1))

        dot_product = self.rank_dot_product_bymatrix(self.user_input, self.item_input)
        element_wise = self.rank_element_wise_bymatrix(self.user_input, self.item_input)
        mask_mat = np.array(list(map(_mask, np.reshape(self.num_idx, (self.batch_size, 1)))))
        dot_product = np.reshape(np.sum(dot_product * mask_mat, 1), (-1,1)) / self.num_idx
        mask_mat = np.repeat(np.reshape(mask_mat, (self.batch_size, self.max_item_num, 1)), self.embedding_size, 2)
        element_wise = np.sum(element_wise * mask_mat, 1) / self.num_idx

        return np.concatenate((dot_product, element_wise, origin_prob),1)

   
    def get_state(self, step_index):
        self.origin_prob = np.reshape(self.origin_rewards[self.batch_index], (-1, 1))  # (batch_size, 1)
        self.dot_product = self.rank_dot_product(self.user_input, self.item_input, step_index)
        self.element_wise_current = self.rank_element_wise(self.user_input, self.item_input, step_index)
        self.vector_current = self.item_embedding_user[self.user_input[:, step_index]]
        self.vector_item = self.item_embedding_item[self.item_input]
        self.vector_current = np.abs(self.vector_current - self.vector_item)
        return np.concatenate((self.vector_mean, self.vector_current, self.dot_product, self.dot_product_mean), 1)


    def rank_element_wise(self, batched_user_input, item_input, step_index):
        self.train_item_ebd = self.item_embedding_user[batched_user_input[:, step_index]]
        self.test_item_ebd = np.reshape(self.item_embedding_item[item_input], (self.batch_size, self.embedding_size))
        return np.multiply(self.train_item_ebd, self.test_item_ebd)  # (batch_size, embedding_size)


    def rank_dot_product(self, batched_user_input, item_input, step_index):
        self.train_item_ebd = self.item_embedding_user[batched_user_input[:, step_index]]
        self.test_item_ebd = np.reshape(self.item_embedding_item[item_input], (self.batch_size, self.embedding_size))
        norm_user = np.sqrt(np.sum(np.multiply(self.train_item_ebd, self.train_item_ebd),1))
        norm_item = np.sqrt(np.sum(np.multiply(self.test_item_ebd, self.test_item_ebd),1))
        norm = np.multiply(norm_user, norm_item)
        dot_prod = np.sum(np.multiply(self.train_item_ebd, self.test_item_ebd), 1)
        cos_similarity = np.where(norm != 0, dot_prod/norm, dot_prod)
        return np.reshape(cos_similarity, (-1, 1))  # (batch_size, 1)


    def rank_element_wise_bymatrix(self, batched_user_input, item_input):
        self.train_item_ebd = self.item_embedding_user[np.reshape(batched_user_input, (-1,1))]  # (batch_size, embedding_size)
        self.test_item_ebd =  self.item_embedding_item[np.reshape(np.tile(item_input, (1,self.max_item_num)), (-1,1))]  # (batch_size, embedding_size)
        return np.reshape(np.multiply(self.train_item_ebd, self.test_item_ebd), (-1,self.max_item_num, self.embedding_size))  # (batch_size, embedding_size)


    def rank_dot_product_bymatrix(self, batched_user_input, item_input):
        self.train_item_ebd = self.item_embedding_user[np.reshape(batched_user_input, (-1,))]  # (batch_size, embedding_size)
        self.test_item_ebd =  self.item_embedding_item[np.reshape(np.tile(item_input, (1,self.max_item_num)), (-1,))]  # (batch_size, embedding_size)
        norm_user = np.sqrt(np.sum(np.multiply(self.train_item_ebd, self.train_item_ebd),1))
        norm_item = np.sqrt(np.sum(np.multiply(self.test_item_ebd, self.test_item_ebd),1))
        norm = np.multiply(norm_user, norm_item)
        dot_prod = np.sum(np.multiply(self.train_item_ebd, self.test_item_ebd), 1)
        cos_similarity = np.where(norm != 0, dot_prod/norm, dot_prod)
        return np.reshape( cos_similarity , (-1, self.max_item_num))  # (batch_size, 1)


    def update_state(self, low_action, low_state, step_index):
        self.action_matrix[:, step_index] = low_action
        self.state_matrix[:, step_index] = low_state

        self.num_selected = self.num_selected + low_action
        self.vector_sum = self.vector_sum + np.multiply(np.reshape(low_action, (-1, 1)), self.vector_current)
        self.element_wise_sum = self.element_wise_sum + np.multiply(np.reshape(low_action, (-1, 1)), self.element_wise_current)
        self.dot_product_sum = self.dot_product_sum + np.multiply(np.reshape(low_action, (-1,1)), self.dot_product)
        num_selected_array = np.reshape(self.num_selected, (-1, 1))
        self.element_wise_mean = np.where(num_selected_array != 0, self.element_wise_sum / num_selected_array, self.element_wise_sum)
        self.vector_mean = np.where(num_selected_array != 0, self.vector_sum / num_selected_array, self.vector_sum)
        self.dot_product_mean = np.where(num_selected_array != 0, self.dot_product_sum / num_selected_array, self.dot_product_sum)


    def get_action_matrix(self):
        return self.action_matrix


    def get_state_matrix(self):
        return self.state_matrix


    def get_selected_items(self, high_action):
        notrevised_index = []
        revised_index = []
        delete_index = []
        keep_index = []
        select_user_input = np.zeros((self.batch_size, self.max_item_num), dtype=np.int)
        for index in range(self.batch_size):

            # We select only the items for which the action matrix has value 1.
            selected = []
            for item_index in range(self.max_item_num):
                if self.action_matrix[index, item_index] == 1:
                    selected.append(self.user_input[index, item_index])

            # revise
            if high_action[index] == 1:
                # delete
                # If there are no selected items, we delete it and build it with random items.
                if len(selected) == 0:
                    delete_index.append(index)
                # keep
                # If we selected all the items rated by the user, we don't change the items.
                if len(selected) == self.num_idx[index]:
                    keep_index.append(index)
                # If we don't delete or keep the profile, we need to revise it.
                revised_index.append(index)
            # not revise
            if high_action[index] == 0:
                notrevised_index.append(index)

            # random select one item from the original enrolled items if no item is selected by the agent 
            # change the number of selected items as 1 at the same time
            if len(selected) == 0:
                original_item_set = list(set(self.user_input[index]))
                if self.padding_number in original_item_set:
                    original_item_set.remove(self.padding_number)
                random_item = np.random.choice(original_item_set, 1)[0]
                selected.append(random_item)
                self.num_selected[index] = 1

            # If we are revising the profile, we keep the selected items and fill the rest with padding.
            for item_index in range(self.max_item_num - len(selected)):
                selected.append(self.padding_number)
            select_user_input[index, :] = np.array(selected)
        
        nochanged = notrevised_index + keep_index
        select_user_input[nochanged] = self.user_input[nochanged]
        self.num_selected[nochanged] = np.reshape(self.num_idx[nochanged],(-1,))
        return select_user_input, self.num_selected, notrevised_index, revised_index, delete_index, keep_index


    def get_reward(self, recommender, batch_index, high_actions, selected_user_input, batched_num_idx, 
                   batched_item_input, batched_label_input, batched_user_idx, batched_user_input, test_flag=False):
        batch_size = selected_user_input.shape[0]

        # difference between likelihood
        loglikelihood = recommender.get_reward(selected_user_input, np.reshape(self.num_selected, (-1, 1)), batched_item_input, batched_label_input)
        old_likelihood = self.origin_rewards[batch_index]
        likelihood_diff = loglikelihood - old_likelihood
        likelihood_diff = np.where(high_actions == 1, likelihood_diff, np.zeros(batch_size))

        #difference between average dot_product
        dot_product = self.rank_dot_product_bymatrix(selected_user_input, batched_item_input)
        new_dot_product = np.sum(np.multiply(dot_product, self.action_matrix),1) / self.num_selected
        old_dot_product = np.sum(dot_product, 1) /batched_num_idx

        dot_product_diff = new_dot_product - old_dot_product
        reward1 = likelihood_diff + self.gamma * dot_product_diff
        
        if test_flag == True:
            original_rmse_total, original_rmse_explicit, original_rmse_implicit = self.get_rmse(batched_user_input, batched_user_idx, test_flag=True)
            revised_rmse_total, revised_rmse_explicit, revised_rmse_implicit = self.get_rmse(selected_user_input, batched_user_idx, test_flag=True)
        else:
            original_rmse_total, original_rmse_explicit, original_rmse_implicit = self.get_rmse(batched_user_input, batched_user_idx, test_flag=False)
            revised_rmse_total, revised_rmse_explicit, revised_rmse_implicit = self.get_rmse(selected_user_input, batched_user_idx, test_flag=False)
        rmse_diff = original_rmse_total - revised_rmse_total 
        rmse_diff = np.array(rmse_diff) 

        return rmse_diff, old_dot_product, rmse_diff
    
    
    def get_rmse(self, revised_user_history, user_idx, test_flag):
        if test_flag == True:
            test_base_data = torch.tensor(self.test_base_data)
            test_target_data = torch.tensor(self.test_target_data)
            masked_test_ref = self.masked_test_ref
            
            revised_base_data = self._get_profile(user_idx, revised_user_history)
            
            if self.args.null_values_model == "multivae":
                self.profile_model.eval()
                recon_data, mu, logvar = self.profile_model(revised_base_data)
                recon_data = recon_data.detach().numpy()
                recon_data = torch.tensor(recon_data)
            else:
                batch_user_data = []
                for i in range(len(user_idx)):
                    user = int(user_idx[i])
                    user_data = []
                    # For each user we add every possible category. In this way we
                    # can compute the rating for each category of a user profile.
                    for category in range(self.n_categories):
                        user_data.append([user, category])
                    user_data = torch.tensor(user_data)
                    batch_user_data.append(user_data)
                batch_user_data = torch.vstack(batch_user_data)
                recon_data = self.profile_model(batch_user_data)
                recon_data = torch.reshape(recon_data.detach(), (len(user_idx), self.n_categories))
            
            mask = revised_base_data == 0
            filled_revised_base_data = torch.where(mask, recon_data, revised_base_data)
            
            rmse_total, rmse_explicit, rmse_implicit = metrics.rmse_with_mask(test_base_data, test_base_data, 
                                                                              filled_revised_base_data, test_target_data, masked_test_ref)
        else:
            user_idx = np.squeeze(user_idx)
            train_base_data = torch.tensor(self.train_base_profiles[user_idx])
            train_target_data = torch.tensor(self.train_target_profiles[user_idx])
            masked_train_ref = torch.tensor(self.masked_train_ref[user_idx])
            
            revised_base_data = self._get_profile(user_idx, revised_user_history)
            
            if self.args.null_values_model == "multivae":
                self.profile_model.eval()
                recon_data, mu, logvar = self.profile_model(revised_base_data)
                recon_data = recon_data.detach().numpy()
                recon_data = torch.tensor(recon_data)
            else:
                batch_user_data = []
                for i in range(len(user_idx)):
                    user = int(user_idx[i])
                    user_data = []
                    # For each user we add every possible category. In this way we
                    # can compute the rating for each category of a user profile.
                    for category in range(self.n_categories):
                        user_data.append([user, category])
                    user_data = torch.tensor(user_data)
                    batch_user_data.append(user_data)
                batch_user_data = torch.vstack(batch_user_data)
                recon_data = self.profile_model(batch_user_data)
                recon_data = torch.reshape(recon_data.detach(), (len(user_idx), self.n_categories))
            
            mask = revised_base_data == 0
            filled_revised_base_data = torch.where(mask, recon_data, revised_base_data)
            
            rmse_total, rmse_explicit, rmse_implicit = metrics.rmse_with_mask(train_base_data, train_base_data, 
                                                                              filled_revised_base_data, train_target_data, masked_train_ref)
            
        return rmse_total, rmse_explicit, rmse_implicit
    
    
    def _get_profile(self, user_idx, revised_user_history):
        revised_base_data = []
        for i in range(len(revised_user_history)):
            user_id = user_idx[i]
            history = revised_user_history[i]
            user_dense_array = self._create_user_dense_array(user_id, history)
            user_tensor = torch.tensor(user_dense_array)
            revised_base_data.append(user_tensor)
        revised_base_data = torch.vstack(revised_base_data)
        return revised_base_data
    
    
    def _create_user_dense_array(self, user_id, user_history):
        user_id = user_id.item()
        user_history_copy = copy.deepcopy(user_history)
        data_full = copy.deepcopy(self.data_full)

        # We remove paddings from user history.
        user_history_copy = user_history_copy[user_history_copy != self.padding_number]
        user_history_copy = np.array(user_history_copy)
        
        # We get the relevant data for user and items.
        # We assume that each user interacts with each item only one time.
        user_data = data_full[data_full['UserID'] == user_id]
        selected_rows_history = user_data[user_data['MovieID'].isin(user_history_copy)]
        profile_movies = self.movies_data[self.movies_data['MovieID'].isin(user_history_copy)]
        
        merged_df = pd.merge(selected_rows_history, profile_movies, on='MovieID', how='inner')
        merged_df  = merged_df.drop('Timestamp', axis=1) 
        merged_df  = merged_df.drop('UserID', axis=1)
        merged_df  = merged_df.drop('CategoryID', axis=1)
        
        # We assign the rating of the item to each of its categories.  
        for i in range(self.n_categories):
            category_col = f'Category_{i}'
            rating_col = 'Rating'
            merged_df[category_col] *= merged_df[rating_col]
        merged_df = merged_df.drop('Rating', axis=1)
        merged_df = merged_df.drop_duplicates()
        
        # We compute the mean value for each category. This is the user profile.
        categories = [f'Category_{i}' for i in range(self.n_categories)]
        category_values = []
        for category in categories:
            # We compute the mean only for the values that are not 0,
            # since we consider 0 as an absence of value.
            values = merged_df[category].values
            values = values[values != 0]
            if len(values) > 0:
                values = np.mean(values)
                values = round(values)
            else:
                values = 0
            category_values.append(values)
            
        array = np.array([category_values])

        return array
    

    def mask_data(self, current_data, next_data):
        mask_matrix = []
        current_data = torch.tensor(current_data)
        next_data = torch.tensor(next_data)
        
        for i in range(len(current_data)):
            current_vector = current_data[i]
            next_vector = next_data[i]
            mask = torch.where((current_vector == 0) & (next_vector != 0))[0]
            boolean_vector = torch.full_like(current_vector, False)
            boolean_vector[mask] = True
            boolean_vector = boolean_vector.to(dtype=torch.bool)
            mask_matrix.append(boolean_vector)
        
        mask_matrix = torch.vstack(mask_matrix)
            
        return current_data, mask_matrix
    