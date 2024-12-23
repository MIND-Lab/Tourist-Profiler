import os
import copy
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split



class Dataset(object):
    def __init__(self, dataset_path, batch_size, fast_testing=False):
        self.dataset_path = dataset_path
        self.fast_testing = fast_testing
        self.batch_size = int(batch_size)
        self.data_full, self.data_full_bal, self.data_tr, self.data_tr_bal, self.data_val, self.data_te = self.load_dataset_as_dataframe()
        self.lists_full, self.lists_full_bal, self.lists_tr, self.lists_tr_bal, self.lists_val, self.lists_te = self.load_dataset_as_lists()
        self._save_sequential_data()
        
        self.data_full_sep, self.data_tr_sep, self.data_val_sep, self.data_te_sep = self.load_separated_dataset_as_dataframe()
        self.lists_full_sep, self.lists_tr_sep, self.lists_val_sep, self.lists_te_sep = self.load_separated_dataset_as_lists()
        self._save_sequential_separated_data()
        
        

    ########################  DATASET PREPROCESSING  ###########################
    def load_dataset_as_dataframe(self):
        # We preprocess the ratings and movies dataset to get a single dataset.
        raw_data = self._preprocess_dataset()
        
        # We find the number of categories of the dataset.
        self.n_categories = len(np.unique(raw_data['CategoryID'].values))
        
        # We find the number of items of the dataset. This is also used as
        # the padding number.
        self.num_items = len(np.unique(raw_data['MovieID'].values))
        self.padding_number = self.num_items
        
        # We create dictionaries that map movies and users identifiers to
        # numerical identifiers. This is done because movies and users
        # identifiers don't follow a sequential order. Some values are missing.
        unique_movie_id = pd.unique(raw_data['MovieID'])
        unique_user_id = pd.unique(raw_data['UserID'])
        movie2id = dict((movie_id, i) for (i, movie_id) in enumerate(unique_movie_id))
        user2id = dict((uid, i) for (i, uid) in enumerate(unique_user_id))
        
        # We apply the mappings to the dataset. 
        data_full = self._create_dataframe_with_mapping(raw_data, user2id, movie2id)

        # We save the updated data of the movies.
        self._save_movies_dataframe(raw_data)
        
        # We split the dataset in train, validation and test sets.
        data_tr, data_tr_bal, data_val, data_te = self._split_train_val_test(data_full)
        data_full = pd.concat([data_tr, data_val, data_te], axis=0)
        data_full.to_csv(f"{self.dataset_path}/preprocessed/data_full.csv", index=False)
        
        # We save separately the full dataset that uses the balanced train set.
        data_full_bal = pd.concat([data_tr_bal, data_val, data_te], axis=1)
        data_full_bal.to_csv(f"{self.dataset_path}/preprocessed/data_full_bal.csv", index=False)

        return data_full, data_full_bal, data_tr, data_tr_bal, data_val, data_te 
    
    
    def _preprocess_dataset(self):
        # We load the ratings dataset.
        ratings = pd.read_csv(
            f'{self.dataset_path}/u.data', 
            sep='\t', 
            engine='python', 
            encoding='latin-1', 
            names=['UserID', 'MovieID', 'Rating', 'Timestamp']
        )
        
        # We load the movies dataset.
        movies = pd.read_csv(
            f'{self.dataset_path}/u.item', 
            sep='|', engine='python', 
            encoding='latin-1', 
            names=['MovieID', 'Title', 'ReleaseData', 'VideoReleaseData', 'IMDBurl', 
                   'Unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 
                   'Comedy','Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                   'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
                   'War', 'Western'])
        
        # We keep only the relevant columns from the movies dataset.        
        movies = movies[['MovieID', 'Title', 'Action', 'Adventure', 'Animation', 
                         'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 
                         'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
                         'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']]
        genre_columns = ['Action', 'Adventure', 'Animation', 'Children\'s', 
                         'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
                         'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
                         'Sci-Fi', 'Thriller', 'War', 'Western']
        
        
        # We allow writing over the dataframe reference.
        pd.options.mode.chained_assignment = None
        
        # We convert the one-hot encoding of genres to a single string value.
        movies['Genres'] = movies.apply(
            lambda row: '|'.join([genre for genre in genre_columns if row[genre] == 1]), 
            axis=1
        )
        movies = movies[['MovieID', 'Title', 'Genres']]
        
        # We replicate the row for each genre of the rated movie. 
        genres_split = movies['Genres'].str.split('|', expand=True).stack()
        expanded_dataframe = movies.loc[genres_split.index.get_level_values(0)].copy()
        expanded_dataframe['Genres'] = genres_split.values
        expanded_dataframe.reset_index(drop=True, inplace=True)
        merged_df = ratings.merge(expanded_dataframe[['MovieID', 'Genres']], on='MovieID', how='left')
        merged_df = merged_df[['UserID', 'MovieID', 'Genres', 'Rating', 'Timestamp']]
        
        # We convert the string values of the genres to numeric values.        
        genre_mapping = {
            'Action': 0,
            'Adventure': 1,
            'Comedy': 2,
            'Drama': 3,
            'Romance': 4,
            'Thriller': 5,
            'Animation': 6,
            'Children\'s': 6,
            'Crime': 6,
            'Documentary': 6,
            'Fantasy': 6,
            'Film-Noir': 6,
            'Horror': 6,
            'Musical': 6,
            'Mystery': 6,
            'Sci-Fi': 6,
            'War': 6,
            'Western': 6
        }
        merged_df['Genres'] = merged_df['Genres'].apply(lambda genre: genre_mapping.get(genre, 0))
        
        merged_df = merged_df[merged_df['Genres'] != 6]
        
        # We rename the 'Genres' column to 'CategoryID'.
        merged_df.rename(columns={'Genres': 'CategoryID'}, inplace=True)
        
        merged_df = merged_df.drop([], axis=1)
        
        # We find the number of users for the whole dataset.
        self.n_users = len(np.unique(merged_df['UserID'].values))
        
        return merged_df
        
        
    def _create_dataframe_with_mapping(self, in_dataframe, user2id, movie2id):
        in_dataframe["UserID"] = in_dataframe["UserID"].map(user2id)
        in_dataframe["MovieID"] = in_dataframe["MovieID"].map(movie2id)
        out_dataframe = in_dataframe
        return out_dataframe
    
    
    def _save_movies_dataframe(self, movies):
        # We drop irrelevant information and duplicates.
        movies_data = movies.drop('UserID', axis=1)
        movies_data = movies_data.drop('Rating', axis=1)
        movies_data = movies_data.drop('Timestamp', axis=1)
        movies_data = movies_data.drop_duplicates()
        
        # We convert the category column in a one-hot encoding.
        movies_data = pd.get_dummies(movies_data, columns=['CategoryID'], prefix='Category')
        movies_data = movies_data.groupby(['MovieID']).sum().reset_index()
        
        # We save the preprocessed movies data.
        if not os.path.exists(f"{self.dataset_path}/preprocessed"):
            os.makedirs(f"{self.dataset_path}/preprocessed")
        movies_data.to_csv(f"{self.dataset_path}/preprocessed/movies_data.csv", index=False)
        self.movies_data = movies_data
        
 
    def _split_train_val_test(self, data, test_proportion=0.3, val_proportion=0.5):
        # We group data by user id.
        data_grouped_by_user = data.groupby('UserID')
        
        tr_list, val_list, te_list = list(), list(), list()
        for _, group in data_grouped_by_user:
            # We sort the data by Timestamp first and MovieID second. 
            # This is done because sometimes a user rates more movies in the
            # same timestamp. And with multiple rows for each movie (for different
            # categories) we may have data that is not sorted sequentially for
            # the same movie.
            group = group.sort_values(by=['Timestamp', 'MovieID'])
            
            # We consider the timestamps removing duplicates.
            timestamps = list(set(group['Timestamp'].values))
            timestamps.sort()
            
            # We split the timestamps in train, validation and test sets.
            train_split_index = int(len(timestamps) * (1 - test_proportion))
            timestamps_tr = timestamps[:train_split_index]
            timestamps_rest = timestamps[train_split_index:]
            val_split_index = int(len(timestamps_rest) * (1 - val_proportion))
            timestamps_val = timestamps_rest[:val_split_index]
            timestamps_te = timestamps_rest[val_split_index:]
            
            # We split the data of the specific user with the timestamps splits.
            tr_group = group[group['Timestamp'].isin(timestamps_tr)]
            val_group = group[group['Timestamp'].isin(timestamps_val)]
            te_group = group[group['Timestamp'].isin(timestamps_te)]
            
            # We add the split data to the corresponding lists.
            tr_list.append(tr_group)
            val_list.append(val_group)
            te_list.append(te_group) 
            
        data_tr = pd.concat(tr_list)
        data_val = pd.concat(val_list)
        data_te = pd.concat(te_list)
        
        # We balance the ratings of the train set.
        data_tr_bal = self._balance_train_set(data_tr)
        
        # We drop the duplicates with same MovieID but different genre.
        data_tr = data_tr.drop_duplicates(subset=['UserID', 'MovieID', 'Timestamp'])
        data_tr_bal = data_tr_bal.drop_duplicates(subset=['UserID', 'MovieID', 'Timestamp'])
        data_val = data_val.drop_duplicates(subset=['UserID', 'MovieID', 'Timestamp'])
        data_te = data_te.drop_duplicates(subset=['UserID', 'MovieID', 'Timestamp'])
        data_tr = data_tr.sort_values(by="Timestamp")
        data_tr_bal = data_tr_bal.sort_values(by="Timestamp")
        data_val = data_val.sort_values(by="Timestamp")
        data_te = data_te.sort_values(by="Timestamp")
        
        # We save the train, validation and test sets in CSV files.
        data_tr.to_csv(f"{self.dataset_path}/preprocessed/data_tr.csv", index=False)
        data_val.to_csv(f"{self.dataset_path}/preprocessed/data_val.csv", index=False)
        data_te.to_csv(f"{self.dataset_path}/preprocessed/data_te.csv", index=False)
        data_tr_bal.to_csv(f"{self.dataset_path}/preprocessed/data_tr_bal.csv", index=False)

        return data_tr, data_tr_bal, data_val, data_te
    
    
    def _balance_train_set(self, data_tr):
        # We drop the duplicates.
        new_data_tr = data_tr.drop_duplicates(subset=['UserID', 'MovieID', 'Timestamp'])
        
        # We consider the rating as the label. The rest are the features.
        X = new_data_tr[['UserID', 'MovieID', 'CategoryID', 'Timestamp']]
        y = new_data_tr['Rating']

        # We resample all the classes to the number of instances of the class
        # with less ratings. A random undersampler is used.
        pipeline = Pipeline([
            ('undersample', RandomUnderSampler(sampling_strategy='not minority'))
        ])
        X_resampled, y_resampled = pipeline.fit_resample(X, y)
        
        # We connect back the features and the labels
        sampled_data_tr = pd.concat([pd.DataFrame(X_resampled, columns=['UserID', 'MovieID', 'CategoryID', 'Timestamp']), 
                                     pd.Series(y_resampled, name='Rating')], axis=1)
        
        return sampled_data_tr
    
    
    def load_dataset_as_lists(self):
        # We drop the duplicates and keep only the data regarding user interactions
        # with movies at each timestep.
        data_tr = self._drop_irrelevant_data_for_dataset_lists(self.data_tr)
        data_tr_bal = self._drop_irrelevant_data_for_dataset_lists(self.data_tr_bal)
        data_val = self._drop_irrelevant_data_for_dataset_lists(self.data_val)
        data_te = self._drop_irrelevant_data_for_dataset_lists(self.data_te)
        
        # For each user we create the list of interacted movies in the train set.
        lists_tr = self._get_interacted_items_list(data_tr)
        lists_tr_bal = self._get_interacted_items_list(data_tr_bal)
        
        # For each user we create the list of interacted movies in the validation set.
        lists_val = self._get_interacted_items_list(data_val)
        
        # For each user we create the list of interacted movies in the test set.
        lists_te = self._get_interacted_items_list(data_te)
        
        # For each user we create the list of interacted movies in the whole dataset.
        lists_full = []
        for i in range(len(lists_tr)):
            user_list = lists_tr[i] + lists_val[i] + lists_te[i]
            lists_full.append(user_list)
        lists_full_bal = []
        for i in range(len(lists_tr_bal)):
            user_list = lists_tr_bal[i] + lists_val[i] + lists_te[i]
            lists_full_bal.append(user_list)
            
        return lists_full, lists_full_bal, lists_tr, lists_tr_bal, lists_val, lists_te
    
    
    def _drop_irrelevant_data_for_dataset_lists(self, data):
        new_data = data.drop_duplicates(subset=['UserID', 'MovieID', 'Timestamp'])
        new_data = new_data.drop('CategoryID', axis=1)
        new_data = new_data.drop('Rating', axis=1)
        return new_data
    
    
    def _get_interacted_items_list(self, data):
        # For each user we create the list of interacted movies in the set. 
        data_grouped_by_user = data.groupby('UserID')
        lists = list()
        for _, group in data_grouped_by_user:
            group = group.sort_values(by=['Timestamp', 'MovieID'])
            movie_list = group['MovieID'].tolist()
            lists.append(movie_list)
            
        # If we don't have a specific user in the set, we add an empty
        # list of items.
        users = data['UserID'].values 
        for i in range(self.n_users):
            if i not in users:
                lists.insert(i, [])  
        
        return lists
        
    
    def _save_sequential_data(self):
        # We create:
        #   - sequential base data: items interacted by users with their categories as one-hot encodings
        #   - sequential targe data: ratings given by users
        train_seq_base_data, train_seq_target_data = self._preprocess_sequential_data(self.data_tr)
        train_bal_seq_base_data, train_bal_seq_target_data = self._preprocess_sequential_data(self.data_tr_bal)
        val_seq_base_data, val_seq_target_data = self._preprocess_sequential_data(self.data_val)
        test_seq_base_data, test_seq_target_data = self._preprocess_sequential_data(self.data_te)
        
        # We save the sequential dataset.
        train_seq_base_data.to_csv(f"{self.dataset_path}/preprocessed/train_seq_base_data.csv", index=False)
        train_bal_seq_base_data.to_csv(f"{self.dataset_path}/preprocessed/train_bal_seq_base_data.csv", index=False)
        val_seq_base_data.to_csv(f"{self.dataset_path}/preprocessed/val_seq_base_data.csv", index=False)
        test_seq_base_data.to_csv(f"{self.dataset_path}/preprocessed/test_seq_base_data.csv", index=False)
        train_seq_target_data.to_csv(f"{self.dataset_path}/preprocessed/train_seq_target_data.csv", index=False)
        train_bal_seq_target_data.to_csv(f"{self.dataset_path}/preprocessed/train_bal_seq_target_data.csv", index=False)
        val_seq_target_data.to_csv(f"{self.dataset_path}/preprocessed/val_seq_target_data.csv", index=False)
        test_seq_target_data.to_csv(f"{self.dataset_path}/preprocessed/test_seq_target_data.csv", index=False)
        
        
    def _preprocess_sequential_data(self, data):
        # We drop the duplicates and we sort the values.
        data = data.drop_duplicates(subset=['UserID', 'MovieID'])
        data = data.sort_values(by=['Timestamp', 'MovieID'])
        
        # We create the features that contain (i) the movie id and (ii) a one-hot
        # encoding of its categories.
        attributes = ['MovieID']
        attributes.extend([f'Category_{i}' for i in range(self.n_categories)])
        seq_base_data = data.merge(self.movies_data[attributes], on='MovieID', how='left')
        
        # The target is the rating given by the user.
        seq_target_data = seq_base_data['Rating']
        
        # We drop irrelevant information.
        seq_base_data = seq_base_data.drop(['Rating', 'CategoryID', 'Timestamp'], axis=1)
        
        return seq_base_data, seq_target_data
        
        
        
    #####################  HRL DATASET PREPROCESSING  ##########################
    def get_train_instances(self):
        num_batches = int(len(self.lists_tr) / self.batch_size)
        user_idx_list, len_user_history_list, user_history_list, next_item_list, next_next_item_list = [], [], [], [], []   
        
        # We create batches that contain:
        #   - user_idx: the user id;
        #   - len_user_history: the number of interacted items in the history;
        #   - user_history: the history of interacted items for each user;
        #   - next_item: the next item the user interacts with;
        #   - next_next_item: the item following the next item.
        for batch in range(num_batches):
            user_idx, len_user_history, user_history, next_item, next_next_item = self._get_train_batch(batch)
            user_idx_list.append(user_idx)
            len_user_history_list.append(len_user_history)
            user_history_list.append(user_history)
            next_item_list.append(next_item)
            next_next_item_list.append(next_next_item)

        # With self.fast_testing = True we consider only the first batch.
        if self.fast_testing:
            user_idx_list = [user_idx_list[0]]
            len_user_history_list = [len_user_history_list[0]]
            user_history_list = [user_history_list[0]]
            next_item_list = [next_item_list[0]]
            next_next_item_list = [next_next_item_list[0]]
            num_batches = 1
        
        return [user_idx_list, len_user_history_list, user_history_list, next_item_list, next_next_item_list, num_batches]


    def _get_train_batch(self, batch):
        batch_user_idx, batch_len_user_history, batch_user_history, batch_next_item, batch_next_next_item = [], [], [], [], []
        begin = batch * self.batch_size
        num_users = len(self.lists_tr)
        
        # We find the maximum length of user history. We will use it to add padding.
        # We use the same padding for all the batches.
        max_len_user_history = max(len(user_list) for user_list in self.lists_tr)

        for idx in range(begin, begin + self.batch_size):
            if idx < num_users:
                user_idx = idx
                user_history = copy.deepcopy(self.lists_tr[user_idx])
                # We first pop the last item in history as the next next item. 
                next_next_item = user_history[-1]
                user_history.pop()
                # Then we pop the new last item in history as the next item.
                next_item = user_history[-1]
                user_history.pop()
                len_user_history = len(user_history)
                batch_user_idx.append(user_idx)
                batch_len_user_history.append(len_user_history)
                batch_user_history.append(user_history)
                batch_next_item.append(next_item)
                batch_next_next_item.append(next_next_item)

        # We add a padding number to fill the interactions of a user.
        # In this way we have user data of the same length.
        batch_user_history = self._add_padding(batch_user_history, max_len_user_history)
        
        batch_user_idx = torch.tensor(batch_user_idx)
        batch_len_user_history = torch.tensor(batch_len_user_history)
        batch_user_history = torch.tensor(batch_user_history)
        batch_next_item = torch.tensor(batch_next_item)
        batch_next_next_item = torch.tensor(batch_next_next_item)
    
        return batch_user_idx, batch_len_user_history, batch_user_history, batch_next_item, batch_next_next_item


    def _add_padding(self, user_history, max_len_user_history):
        for i in range(len(user_history)):
            user_history[i] = user_history[i] + [self.padding_number] * (max_len_user_history + 1 - len(user_history[i]))
        return user_history
    
    
    def get_val_instances(self):
        # The number of batches is the integer division of the instances and the
        # batch size, plus 1 for the remaining instances.
        num_batches = int(len(self.data_val) / self.batch_size) + 1
        user_idx_list, len_user_history_list, user_history_list, next_item_list, next_next_item_list = [], [], [], [], []
        
        # We don't use the original lists and data in order to be able to modify them.
        lists_tr = copy.deepcopy(self.lists_tr)
        lists_val = copy.deepcopy(self.lists_val)
        data_val = self.data_val
        data_val = data_val.drop_duplicates(subset=['UserID', 'MovieID'])
        
        # We find the maximum length of user history to add padding.
        # We use the same padding for all the batches.
        merged_lists = [tr_user_history + val_user_history 
                        for tr_user_history, val_user_history 
                        in zip(lists_tr, lists_val)]
        max_len_user_history = max(len(user_list) for user_list in merged_lists)
        
        # If we are fast testing, we use only a single batch.
        if self.fast_testing == True:
            num_val_users = self.batch_size #256
        else:
            # We limit the val users to the same users of the train set.
            num_train_batches = int(len(self.lists_tr) / self.batch_size)
            num_val_users = num_train_batches * self.batch_size
            
        # We use a list to keep track of the items of lists_val that are
        # used for the next items of each user.
        used_user_val_items_idx = [0] * num_val_users
        
        idx = 0
        considered_idx = 0
        val_instance_idx_list = []
        end_while = False
        while not end_while:
            user_idx = data_val.iloc[idx]['UserID']
            
            # This is used for fast testing. We limit the number of users to the
            # batched users used in the train set.
            if user_idx >= num_val_users:
                end_while = True
                break
            
            user_history_tr = copy.deepcopy(lists_tr[user_idx])
            user_history_val = copy.deepcopy(lists_val[user_idx])
            
            used_idx = used_user_val_items_idx[user_idx]
            
            # We get next_item and next_next_item only if there are at least
            # 2 not used items for the user in lists_val.
            if used_idx < len(user_history_val) - 1:
                next_item = user_history_val[used_idx]
                next_next_item = user_history_val[used_idx + 1]
                used_user_val_items_idx[user_idx] += 1

                # If we already used some items as next items, we can
                # add them to the user history for the val set. Otherwise
                # we add an empty list.
                if used_idx > 0:
                    user_history_val_to_add = user_history_val[:used_idx]
                else:
                    user_history_val_to_add = []
                user_history = user_history_tr + user_history_val_to_add
   
                user_idx_list.append(user_idx)
                len_user_history_list.append(len(user_history))
                user_history_list.append(user_history)
                next_item_list.append(next_item)
                next_next_item_list.append(next_next_item)
                val_instance_idx_list.append(considered_idx)
                
                # If we add an instance we increase the count of identifiers.
                considered_idx += 1
            
            idx += 1

        user_idx_list = self._create_test_batch(user_idx_list, self.batch_size)
        len_user_history_list = self._create_test_batch(len_user_history_list, self.batch_size)
        user_history_list = self._create_user_history_test_batch(user_history_list, self.batch_size, max_len_user_history)
        next_item_list = self._create_test_batch(next_item_list, self.batch_size)
        next_next_item_list = self._create_test_batch(next_next_item_list, self.batch_size)
        val_instance_idx_list = self._create_test_batch(val_instance_idx_list, self.batch_size)
        
        return [user_idx_list, len_user_history_list, user_history_list, next_item_list, next_next_item_list, num_batches, val_instance_idx_list]
    

    def get_test_instances(self):
        # The number of batches is the integer division of the instances and the
        # batch size, plus 1 for the remaining instances.
        num_batches = int(len(self.data_te) / self.batch_size) + 1
        user_idx_list, len_user_history_list, user_history_list, next_item_list, next_next_item_list = [], [], [], [], []
        
        # We don't use the original lists and data in order to be able to modify them.
        lists_tr = copy.deepcopy(self.lists_tr)
        lists_val = copy.deepcopy(self.lists_val)
        lists_te = copy.deepcopy(self.lists_te)
        data_te = self.data_te
        data_te = data_te.drop_duplicates(subset=['UserID', 'MovieID'])
        
        # We find the maximum length of user history to add padding.
        # We use the same padding for all the batches.
        merged_lists = [tr_user_history + val_user_history + te_user_history 
                        for tr_user_history, val_user_history, te_user_history 
                        in zip(lists_tr, lists_val, lists_te)]
        max_len_user_history = max(len(user_list) for user_list in merged_lists)
        
        # If we are fast testing, we use only a single batch.
        if self.fast_testing == True:
            num_test_users = self.batch_size #256
        else:
            # We limit the test users to the same users of the train set.
            num_train_batches = int(len(self.lists_tr) / self.batch_size)
            num_test_users = num_train_batches * self.batch_size
            
        # We use a list to keep track of the items of lists_te that are
        # used for the next items of each user.
        used_user_test_items_idx = [0] * num_test_users
        
        idx = 0
        considered_idx = 0
        test_instance_idx_list = []
        end_while = False
        while not end_while:
            user_idx = data_te.iloc[idx]['UserID']
            
            # This is used for fast testing. We limit the number of users to the
            # batched users used in the train set.
            if user_idx >= num_test_users:
                end_while = True
                break
            
            user_history_tr = copy.deepcopy(lists_tr[user_idx])
            user_history_val = copy.deepcopy(lists_val[user_idx])
            user_history_te = copy.deepcopy(lists_te[user_idx])
            
            used_idx = used_user_test_items_idx[user_idx]
            
            # We get next_item and next_next_item only if there are at least
            # 2 not used items for the user in lists_te.
            if used_idx < len(user_history_te) - 1:
                next_item = user_history_te[used_idx]
                next_next_item = user_history_te[used_idx + 1]
                used_user_test_items_idx[user_idx] += 1

                # If we already used some test items as next items, we can
                # add them to the user history for the test set. Otherwise
                # we add an empty list.
                if used_idx > 0:
                    user_history_te_to_add = user_history_te[:used_idx]
                else:
                    user_history_te_to_add = []
                user_history = user_history_tr + user_history_val + user_history_te_to_add
   
                user_idx_list.append(user_idx)
                len_user_history_list.append(len(user_history))
                user_history_list.append(user_history)
                next_item_list.append(next_item)
                next_next_item_list.append(next_next_item)
                test_instance_idx_list.append(considered_idx)
                
                # If we add an instance we increase the count of identifiers.
                considered_idx += 1
            
            idx += 1

        user_idx_list = self._create_test_batch(user_idx_list, self.batch_size)
        len_user_history_list = self._create_test_batch(len_user_history_list, self.batch_size)
        user_history_list = self._create_user_history_test_batch(user_history_list, self.batch_size, max_len_user_history)
        next_item_list = self._create_test_batch(next_item_list, self.batch_size)
        next_next_item_list = self._create_test_batch(next_next_item_list, self.batch_size)
        test_instance_idx_list = self._create_test_batch(test_instance_idx_list, self.batch_size)
        
        return [user_idx_list, len_user_history_list, user_history_list, next_item_list, next_next_item_list, num_batches, test_instance_idx_list]
    
    
    def _create_test_batch(self, input_list, batch_size):
        batches = [torch.tensor(input_list[i:i + batch_size]) for i in range(0, len(input_list), batch_size)]
        return batches
    
    
    def _create_user_history_test_batch(self, input_list, batch_size, max_len_user_history):
        batches = []
        
        # We create batches with batch size.
        num_batches = int(len(input_list) / batch_size)
        batched_input = num_batches * batch_size
        for i in range(0, batched_input, batch_size):
            batch_user_history = []
            batch_len_user_history = []
            for j in range(batch_size):
                batch_user_history.append(input_list[i + j])
                batch_len_user_history.append(len(input_list[i + j]))
            batch_user_history = self._add_padding(batch_user_history, max_len_user_history)
            batch_user_history = torch.tensor(batch_user_history)    
            batches.append(batch_user_history)
            
        # We add the rest of the instances that exceeded the batch size.
        i = batched_input
        batch_user_history = []
        batch_len_user_history = []
        for j in range(len(input_list) - batched_input):
            batch_user_history.append(input_list[i + j])
            batch_len_user_history.append(len(input_list[i + j]))
        batch_user_history = self._add_padding(batch_user_history, max_len_user_history)
        batches.append(batch_user_history)   
            
        return batches
    


    #################  PROFILE DATASET PREPROCESSING  ##########################  
    def create_user_sparse_array(self, user_id, history, n_categories, padding_number):
        history_copy = copy.deepcopy(history)
        data_full = copy.deepcopy(self.data_full)
        
        # We remove paddings from user history.
        while padding_number in history_copy:
            history_copy.remove(padding_number)
        
        # We get the relevant data for user and items.
        # We assume that each user interacts with each item only one time.
        user_data = data_full[data_full['UserID'] == user_id]
        selected_rows_history = user_data[user_data['MovieID'].isin(history_copy)]
        profile_movies = self.movies_data[self.movies_data['MovieID'].isin(history_copy)]
        
        # We merge the relevant items with their categories. Then we drop useless attributes.
        merged_df = pd.merge(selected_rows_history, profile_movies, on='MovieID', how='inner')
        merged_df  = merged_df.drop('Timestamp', axis=1) 
        merged_df  = merged_df.drop('UserID', axis=1)
        merged_df  = merged_df.drop('CategoryID', axis=1)
        
        # We assign the rating of the item to each of its categories. 
        for i in range(n_categories):
            category_col = f'Category_{i}'
            rating_col = 'Rating'
            merged_df[category_col] *= merged_df[rating_col]
        merged_df = merged_df.drop('Rating', axis=1)
        merged_df = merged_df.drop_duplicates()
        
        # We compute the mean value for each category. This is the user profile.
        categories = [f'Category_{i}' for i in range(n_categories)]
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
            
        # We convert the user profile to a sparse array.
        #array = sparse.csr_array(category_values, dtype='float32', shape=(1, n_categories))
        array = sparse.csr_matrix(category_values, dtype='float32', shape=(1, n_categories))

        return array
    
    
    def create_train_data(self):
        # We use the train data dropping duplicates.
        data_tr = self.data_tr
        data_tr = data_tr.drop_duplicates(subset=['UserID', 'MovieID'])
        data_tr = data_tr.sort_values(by=['Timestamp', 'MovieID'])
        data_tr = torch.tensor(data_tr.to_numpy())
        lists_tr = copy.deepcopy(self.lists_tr)
        
        # We identify train users. 
        train_users = [tensor[0].item() for tensor in data_tr]
        
        # We count the number of instances of the train set.
        n_instances = len(data_tr)
        
        train_base_data = None
        train_target_data = None
        for i in tqdm(range(n_instances)):
            user_id = data_tr[i][0].item()
            item_id = data_tr[i][1].item()
            # We find the index of the item in the train history of the user.
            train_item_index = lists_tr[user_id].index(item_id)
            # We consider the past train history of the user up to the considered item.
            user_past_history = lists_tr[user_id][:train_item_index]
            # We consider the current train history of the user that considers also the current item.
            user_current_history = lists_tr[user_id][:train_item_index + 1]
            # We build user profiles with past and current history.
            user_past_array = self.create_user_sparse_array(user_id, user_past_history, self.n_categories, self.padding_number)
            user_current_array = self.create_user_sparse_array(user_id, user_current_history, self.n_categories, self.padding_number)
            
            # If we haven't built any profile yet, we assign the profiles to
            # the train data. Otherwise we just add the profiles to the previous
            # ones.
            if train_base_data is None:
                train_base_data = user_past_array
                train_target_data = user_current_array
            else:
                train_base_data = sparse.vstack([train_base_data, user_past_array])
                train_target_data = sparse.vstack([train_target_data, user_current_array])
                
        # We save the train data and the users for each profile.
        train_base_data = train_base_data.todense()
        train_target_data = train_target_data.todense()
        dataframe_train_base_data = pd.DataFrame(train_base_data)
        dataframe_train_base_data.to_csv(f'{self.dataset_path}/preprocessed/train_base_data.csv', index=False)
        dataframe_train_target_data = pd.DataFrame(train_target_data)
        dataframe_train_target_data.to_csv(f'{self.dataset_path}/preprocessed/train_target_data.csv', index=False)
        dataframe_train_users = pd.DataFrame(train_users)
        dataframe_train_users.to_csv(f'{self.dataset_path}/preprocessed/train_users.csv', index=False)
    
    
    def create_validation_data(self):
        # We use the validation data dropping duplicates.
        data_val = self.data_val
        data_val = data_val.drop_duplicates(subset=['UserID', 'MovieID'])
        data_val = data_val.sort_values(by=['Timestamp', 'MovieID'])
        data_val = torch.tensor(data_val.to_numpy())
        lists_tr = copy.deepcopy(self.lists_tr)
        lists_val = copy.deepcopy(self.lists_val)
  
        # We identify validation users. 
        val_users = [tensor[0].item() for tensor in data_val]
        
        # We count the number of instances of the validation set.
        n_instances = len(data_val)
        
        val_base_data = None
        val_target_data = None
        for i in tqdm(range(n_instances)):
            user_id = data_val[i][0].item()
            item_id = data_val[i][1].item()
            # We find the index of the item in the validation history of the user.
            val_item_index = lists_val[user_id].index(item_id)
            # We consider the past history of the user up to the considered item.
            # This history includes also the train set.
            user_past_history = lists_tr[user_id] + lists_val[user_id][:val_item_index]
            # We consider the current history of the user that considers also the current item.
            user_current_history = lists_tr[user_id] + lists_val[user_id][:val_item_index + 1]
            # We build user profiles with past and current history.
            user_past_array = self.create_user_sparse_array(user_id, user_past_history, self.n_categories, self.padding_number)
            user_current_array = self.create_user_sparse_array(user_id, user_current_history, self.n_categories, self.padding_number)
            
            # If we haven't built any profile yet, we assign the profiles to
            # the validation data. Otherwise we just add the profiles to the 
            # previous ones.
            if val_base_data is None:
                val_base_data = user_past_array
                val_target_data = user_current_array
            else:
                val_base_data = sparse.vstack([val_base_data, user_past_array])
                val_target_data = sparse.vstack([val_target_data, user_current_array])
                
        # We save the validation data and the users for each profile.
        val_base_data = val_base_data.todense()
        val_target_data = val_target_data.todense()
        dataframe_val_base_data = pd.DataFrame(val_base_data)
        dataframe_val_base_data.to_csv(f'{self.dataset_path}/preprocessed/val_base_data.csv', index=False)
        dataframe_val_target_data = pd.DataFrame(val_target_data)
        dataframe_val_target_data.to_csv(f'{self.dataset_path}/preprocessed/val_target_data.csv', index=False)
        dataframe_val_users = pd.DataFrame(val_users)
        dataframe_val_users.to_csv(f'{self.dataset_path}/preprocessed/val_users.csv', index=False)
    
       
    def create_test_data(self):
        # We use the test data dropping duplicates.
        data_te = self.data_te
        data_te = data_te.drop_duplicates(subset=['UserID', 'MovieID'])
        data_te = data_te.sort_values(by=['Timestamp', 'MovieID'])
        data_te = torch.tensor(data_te.to_numpy())
        lists_tr = copy.deepcopy(self.lists_tr)
        lists_val = copy.deepcopy(self.lists_val)
        lists_te = copy.deepcopy(self.lists_te)
        
        # We identify test users. 
        test_users = [tensor[0].item() for tensor in data_te]
        
        # We count the number of instances of the test set.
        n_instances = len(data_te)
        
        test_base_data = None
        test_target_data = None
        for i in tqdm(range(n_instances)):
            user_id = data_te[i][0].item()
            item_id = data_te[i][1].item()
            # We find the index of the item in the test history of the user.
            test_item_index = lists_te[user_id].index(item_id)
            # We consider the past history of the user up to the considered item.
            # This history includes both the train and the validation sets.
            user_past_history = lists_tr[user_id] + lists_val[user_id] + lists_te[user_id][:test_item_index]
            # We consider the current history of the user that considers also the current item.
            user_current_history = lists_tr[user_id] + lists_val[user_id] + lists_te[user_id][:test_item_index + 1]
            # We build user profiles with past and current history.
            user_past_array = self.create_user_sparse_array(user_id, user_past_history, self.n_categories, self.padding_number)
            user_current_array = self.create_user_sparse_array(user_id, user_current_history, self.n_categories, self.padding_number)
            
            # If we haven't built any profile yet, we assign the profiles to
            # the test data. Otherwise we just add the profiles to the previous
            # ones.
            if test_base_data is None:
                test_base_data = user_past_array
                test_target_data = user_current_array
            else:
                test_base_data = sparse.vstack([test_base_data, user_past_array])
                test_target_data = sparse.vstack([test_target_data, user_current_array])
                
        # We save the test data and the users for each profile.
        test_base_data = test_base_data.todense()
        test_target_data = test_target_data.todense()
        dataframe_test_base_data = pd.DataFrame(test_base_data)
        dataframe_test_base_data.to_csv(f'{self.dataset_path}/preprocessed/test_base_data.csv', index=False)
        dataframe_test_target_data = pd.DataFrame(test_target_data)
        dataframe_test_target_data.to_csv(f'{self.dataset_path}/preprocessed/test_target_data.csv', index=False)
        dataframe_test_users = pd.DataFrame(test_users)
        dataframe_test_users.to_csv(f'{self.dataset_path}/preprocessed/test_users.csv', index=False)
        
    
    def create_train_balanced_data(self):
        # We use the train data dropping duplicates.
        data_tr_bal = self.data_tr_bal
        data_tr_bal = data_tr_bal.drop_duplicates(subset=['UserID', 'MovieID'])
        data_tr_bal = data_tr_bal.sort_values(by=['Timestamp', 'MovieID'])
        data_tr_bal = torch.tensor(data_tr_bal.to_numpy())
        lists_tr_bal = copy.deepcopy(self.lists_tr_bal)
        
        # We identify train users. 
        train_bal_users = [tensor[0].item() for tensor in data_tr_bal]
        
        # We count the number of instances of the train set.
        n_instances = len(data_tr_bal)
        
        train_bal_base_data = None
        train_bal_target_data = None
        for i in tqdm(range(n_instances)):
            user_id = data_tr_bal[i][0].item()
            item_id = data_tr_bal[i][1].item()
            # We find the index of the item in the train history of the user.
            train_item_index = lists_tr_bal[user_id].index(item_id)
            # We consider the past train history of the user up to the considered item.
            user_past_history = lists_tr_bal[user_id][:train_item_index]
            # We consider the current train history of the user that considers also the current item.
            user_current_history = lists_tr_bal[user_id][:train_item_index + 1]
            # We build user profiles with past and current history.
            user_past_array = self.create_user_sparse_array(user_id, user_past_history, self.n_categories, self.padding_number)
            user_current_array = self.create_user_sparse_array(user_id, user_current_history, self.n_categories, self.padding_number)
            
            # If we haven't built any profile yet, we assign the profiles to
            # the train data. Otherwise we just add the profiles to the previous
            # ones.
            if train_bal_base_data is None:
                train_bal_base_data = user_past_array
                train_bal_target_data = user_current_array
            else:
                train_bal_base_data = sparse.vstack([train_bal_base_data, user_past_array])
                train_bal_target_data = sparse.vstack([train_bal_target_data, user_current_array])
                
        # We save the train data and the users for each profile.
        train_bal_base_data = train_bal_base_data.todense()
        train_bal_target_data = train_bal_target_data.todense()
        dataframe_train_bal_base_data = pd.DataFrame(train_bal_base_data)
        dataframe_train_bal_base_data.to_csv(f'{self.dataset_path}/preprocessed/train_bal_base_data.csv', index=False)
        dataframe_train_bal_target_data = pd.DataFrame(train_bal_target_data)
        dataframe_train_bal_target_data.to_csv(f'{self.dataset_path}/preprocessed/train_bal_target_data.csv', index=False)
        dataframe_train_bal_users = pd.DataFrame(train_bal_users)
        dataframe_train_bal_users.to_csv(f'{self.dataset_path}/preprocessed/train_bal_users.csv', index=False)
        
    
    def create_validation_balanced_data(self):
        # We use the validation data dropping duplicates.
        data_val = self.data_val
        data_val = data_val.drop_duplicates(subset=['UserID', 'MovieID'])
        data_val = data_val.sort_values(by=['Timestamp', 'MovieID'])
        data_val = torch.tensor(data_val.to_numpy())
        lists_tr_bal = copy.deepcopy(self.lists_tr_bal)
        lists_val = copy.deepcopy(self.lists_val)
  
        # We identify validation users. 
        val_bal_users = [tensor[0].item() for tensor in data_val]
        
        # We count the number of instances of the validation set.
        n_instances = len(data_val)
        
        val_bal_base_data = None
        val_bal_target_data = None
        for i in tqdm(range(n_instances)):
            user_id = data_val[i][0].item()
            item_id = data_val[i][1].item()
            # We find the index of the item in the validation history of the user.
            val_item_index = lists_val[user_id].index(item_id)
            # We consider the past history of the user up to the considered item.
            # This history includes also the train set.
            user_past_history = lists_tr_bal[user_id] + lists_val[user_id][:val_item_index]
            # We consider the current history of the user that considers also the current item.
            user_current_history = lists_tr_bal[user_id] + lists_val[user_id][:val_item_index + 1]
            # We build user profiles with past and current history.
            user_past_array = self.create_user_sparse_array(user_id, user_past_history, self.n_categories, self.padding_number)
            user_current_array = self.create_user_sparse_array(user_id, user_current_history, self.n_categories, self.padding_number)
            
            # If we haven't built any profile yet, we assign the profiles to
            # the validation data. Otherwise we just add the profiles to the 
            # previous ones.
            if val_bal_base_data is None:
                val_bal_base_data = user_past_array
                val_bal_target_data = user_current_array
            else:
                val_bal_base_data = sparse.vstack([val_bal_base_data, user_past_array])
                val_bal_target_data = sparse.vstack([val_bal_target_data, user_current_array])
                
        # We save the validation data and the users for each profile.
        val_bal_base_data = val_bal_base_data.todense()
        val_bal_target_data = val_bal_target_data.todense()
        dataframe_val_bal_base_data = pd.DataFrame(val_bal_base_data)
        dataframe_val_bal_base_data.to_csv(f'{self.dataset_path}/preprocessed/val_bal_base_data.csv', index=False)
        dataframe_val_bal_target_data = pd.DataFrame(val_bal_target_data)
        dataframe_val_bal_target_data.to_csv(f'{self.dataset_path}/preprocessed/val_bal_target_data.csv', index=False)
        dataframe_val_bal_users = pd.DataFrame(val_bal_users)
        dataframe_val_bal_users.to_csv(f'{self.dataset_path}/preprocessed/val_bal_users.csv', index=False) 
        
        
    def create_test_balanced_data(self):
        # We use the test data dropping duplicates.
        data_te = self.data_te
        data_te = data_te.drop_duplicates(subset=['UserID', 'MovieID'])
        data_te = data_te.sort_values(by=['Timestamp', 'MovieID'])
        data_te = torch.tensor(data_te.to_numpy())
        lists_tr_bal = copy.deepcopy(self.lists_tr_bal)
        lists_val = copy.deepcopy(self.lists_val)
        lists_te = copy.deepcopy(self.lists_te)
        
        # We identify test users. 
        test_bal_users = [tensor[0].item() for tensor in data_te]
        
        # We count the number of instances of the test set.
        n_instances = len(data_te)
        
        test_bal_base_data = None
        test_bal_target_data = None
        for i in tqdm(range(n_instances)):
            user_id = data_te[i][0].item()
            item_id = data_te[i][1].item()
            # We find the index of the item in the test history of the user.
            test_item_index = lists_te[user_id].index(item_id)
            # We consider the past history of the user up to the considered item.
            # This history includes both the train and the validation sets.
            user_past_history = lists_tr_bal[user_id] + lists_val[user_id] + lists_te[user_id][:test_item_index]
            # We consider the current history of the user that considers also the current item.
            user_current_history = lists_tr_bal[user_id] + lists_val[user_id] + lists_te[user_id][:test_item_index + 1]
            # We build user profiles with past and current history.
            user_past_array = self.create_user_sparse_array(user_id, user_past_history, self.n_categories, self.padding_number)
            user_current_array = self.create_user_sparse_array(user_id, user_current_history, self.n_categories, self.padding_number)
            
            # If we haven't built any profile yet, we assign the profiles to
            # the test data. Otherwise we just add the profiles to the previous
            # ones.
            if test_bal_base_data is None:
                test_bal_base_data = user_past_array
                test_bal_target_data = user_current_array
            else:
                test_bal_base_data = sparse.vstack([test_bal_base_data, user_past_array])
                test_bal_target_data = sparse.vstack([test_bal_target_data, user_current_array])
                
        # We save the test data and the users for each profile.
        test_bal_base_data = test_bal_base_data.todense()
        test_bal_target_data = test_bal_target_data.todense()
        dataframe_test_bal_base_data = pd.DataFrame(test_bal_base_data)
        dataframe_test_bal_base_data.to_csv(f'{self.dataset_path}/preprocessed/test_bal_base_data.csv', index=False)
        dataframe_test_bal_target_data = pd.DataFrame(test_bal_target_data)
        dataframe_test_bal_target_data.to_csv(f'{self.dataset_path}/preprocessed/test_bal_target_data.csv', index=False)
        dataframe_test_bal_users = pd.DataFrame(test_bal_users)
        dataframe_test_bal_users.to_csv(f'{self.dataset_path}/preprocessed/test_bal_users.csv', index=False)



    #################  SEPARATED USERS DATASET PREPROCESSING  ##################
    def load_separated_dataset_as_dataframe(self):
        # We split the dataset in train, validation and test sets.
        data_tr_sep, data_val_sep, data_te_sep = self._split_train_val_test_separated(self.data_full)
        data_full_sep = pd.concat([data_tr_sep, data_val_sep, data_te_sep], axis=0)
        data_full_sep.to_csv(f"{self.dataset_path}/preprocessed/data_full_sep.csv", index=False)
    
        return data_full_sep, data_tr_sep, data_val_sep, data_te_sep 
    
    
    def _split_train_val_test_separated(self, data, test_proportion=0.3, val_proportion=0.5):
        # We find all the users of the dataset.
        unique_users = data['UserID'].unique()
        
        # We split users in train, validation and test sets.
        train_users, remaining_users = train_test_split(unique_users, test_size=test_proportion, random_state=42)
        validation_users, test_users = train_test_split(remaining_users, test_size=val_proportion, random_state=42)
        data_tr_sep = data[data['UserID'].isin(train_users)]
        data_val_sep = data[data['UserID'].isin(validation_users)]
        data_te_sep = data[data['UserID'].isin(test_users)]
        
        # We save the train, validation and test sets in CSV files.
        data_tr_sep.to_csv(f"{self.dataset_path}/preprocessed/data_tr_sep.csv", index=False)
        data_val_sep.to_csv(f"{self.dataset_path}/preprocessed/data_val_sep.csv", index=False)
        data_te_sep.to_csv(f"{self.dataset_path}/preprocessed/data_te_sep.csv", index=False)
        
        return data_tr_sep, data_val_sep, data_te_sep
        
  
    def load_separated_dataset_as_lists(self):
        # We drop the duplicates and keep only the data regarding user interactions
        # with movies at each timestep.
        data_tr_sep = self._drop_irrelevant_data_for_dataset_lists(self.data_tr_sep)
        data_val_sep = self._drop_irrelevant_data_for_dataset_lists(self.data_val_sep)
        data_te_sep = self._drop_irrelevant_data_for_dataset_lists(self.data_te_sep)
        
        # For each user we create the list of interacted movies in the train set.
        lists_tr_sep = self._get_interacted_items_list(data_tr_sep)
        
        # For each user we create the list of interacted movies in the validation set.
        lists_val_sep = self._get_interacted_items_list(data_val_sep)
        
        # For each user we create the list of interacted movies in the test set.
        lists_te_sep = self._get_interacted_items_list(data_te_sep)
        
        # For each user we create the list of interacted movies in the whole dataset.
        lists_full_sep = []
        for i in range(len(lists_tr_sep)):
            user_list = lists_tr_sep[i] + lists_val_sep[i] + lists_te_sep[i]
            lists_full_sep.append(user_list)

        return lists_full_sep, lists_tr_sep, lists_val_sep, lists_te_sep
    
    
    def _save_sequential_separated_data(self):
        # We create:
        #   - sequential base data: items interacted by users with their categories as one-hot encodings
        #   - sequential targe data: ratings given by users
        train_sep_seq_base_data, train_sep_seq_target_data = self._preprocess_sequential_data(self.data_tr_sep)
        val_sep_seq_base_data, val_sep_seq_target_data = self._preprocess_sequential_data(self.data_val_sep)
        test_sep_seq_base_data, test_sep_seq_target_data = self._preprocess_sequential_data(self.data_te_sep)
        
        # We save the sequential dataset.
        train_sep_seq_base_data.to_csv(f"{self.dataset_path}/preprocessed/train_sep_seq_base_data.csv", index=False)
        val_sep_seq_base_data.to_csv(f"{self.dataset_path}/preprocessed/val_sep_seq_base_data.csv", index=False)
        test_sep_seq_base_data.to_csv(f"{self.dataset_path}/preprocessed/test_sep_seq_base_data.csv", index=False)
        train_sep_seq_target_data.to_csv(f"{self.dataset_path}/preprocessed/train_sep_seq_target_data.csv", index=False)
        val_sep_seq_target_data.to_csv(f"{self.dataset_path}/preprocessed/val_sep_seq_target_data.csv", index=False)
        test_sep_seq_target_data.to_csv(f"{self.dataset_path}/preprocessed/test_sep_seq_target_data.csv", index=False)
    
    
    def create_separated_train_data(self):
        # We use the train data dropping duplicates.
        data_tr_sep = self.data_tr_sep
        data_tr_sep = data_tr_sep.drop_duplicates(subset=['UserID', 'MovieID'])
        data_tr_sep = data_tr_sep.sort_values(by=['Timestamp', 'MovieID'])
        data_tr_sep = torch.tensor(data_tr_sep.to_numpy())
        lists_tr_sep = copy.deepcopy(self.lists_tr_sep)
        
        # We identify train users. 
        train_sep_users = [tensor[0].item() for tensor in data_tr_sep]
        
        # We count the number of instances of the train set.
        n_instances = len(data_tr_sep)
        
        train_sep_base_data = None
        train_sep_target_data = None
        for i in tqdm(range(n_instances)):
            user_id = data_tr_sep[i][0].item()
            item_id = data_tr_sep[i][1].item()
            # We find the index of the item in the train history of the user.
            train_item_index = lists_tr_sep[user_id].index(item_id)
            # We consider the past train history of the user up to the considered item.
            user_past_history = lists_tr_sep[user_id][:train_item_index]
            # We consider the current train history of the user that considers also the current item.
            user_current_history = lists_tr_sep[user_id][:train_item_index + 1]
            # We build user profiles with past and current history.
            user_past_array = self.create_user_sparse_array(user_id, user_past_history, self.n_categories, self.padding_number)
            user_current_array = self.create_user_sparse_array(user_id, user_current_history, self.n_categories, self.padding_number)
            
            # If we haven't built any profile yet, we assign the profiles to
            # the train data. Otherwise we just add the profiles to the previous
            # ones.
            if train_sep_base_data is None:
                train_sep_base_data = user_past_array
                train_sep_target_data = user_current_array
            else:
                train_sep_base_data = sparse.vstack([train_sep_base_data, user_past_array])
                train_sep_target_data = sparse.vstack([train_sep_target_data, user_current_array])
                
        # We save the train data and the users for each profile.
        train_sep_base_data = train_sep_base_data.todense()
        train_sep_target_data = train_sep_target_data.todense()
        dataframe_train_sep_base_data = pd.DataFrame(train_sep_base_data)
        dataframe_train_sep_base_data.to_csv(f'{self.dataset_path}/preprocessed/train_sep_base_data.csv', index=False)
        dataframe_train_sep_target_data = pd.DataFrame(train_sep_target_data)
        dataframe_train_sep_target_data.to_csv(f'{self.dataset_path}/preprocessed/train_sep_target_data.csv', index=False)
        dataframe_train_sep_users = pd.DataFrame(train_sep_users)
        dataframe_train_sep_users.to_csv(f'{self.dataset_path}/preprocessed/train_sep_users.csv', index=False)
    
    
    def create_separated_validation_data(self):
        # We use the validation data dropping duplicates.
        data_val_sep = self.data_val_sep
        data_val_sep = data_val_sep.drop_duplicates(subset=['UserID', 'MovieID'])
        data_val_sep = data_val_sep.sort_values(by=['Timestamp', 'MovieID'])
        data_val_sep = torch.tensor(data_val_sep.to_numpy())
        lists_tr_sep = copy.deepcopy(self.lists_tr_sep)
        lists_val_sep = copy.deepcopy(self.lists_val_sep)
  
        # We identify validation users. 
        val_sep_users = [tensor[0].item() for tensor in data_val_sep]
        
        # We count the number of instances of the validation set.
        n_instances = len(data_val_sep)
        
        val_sep_base_data = None
        val_sep_target_data = None
        for i in tqdm(range(n_instances)):
            user_id = data_val_sep[i][0].item()
            item_id = data_val_sep[i][1].item()
            # We find the index of the item in the validation history of the user.
            val_item_index = lists_val_sep[user_id].index(item_id)
            # We consider the past history of the user up to the considered item.
            # This history includes also the train set.
            user_past_history = lists_tr_sep[user_id] + lists_val_sep[user_id][:val_item_index]
            # We consider the current history of the user that considers also the current item.
            user_current_history = lists_tr_sep[user_id] + lists_val_sep[user_id][:val_item_index + 1]
            # We build user profiles with past and current history.
            user_past_array = self.create_user_sparse_array(user_id, user_past_history, self.n_categories, self.padding_number)
            user_current_array = self.create_user_sparse_array(user_id, user_current_history, self.n_categories, self.padding_number)
            
            # If we haven't built any profile yet, we assign the profiles to
            # the validation data. Otherwise we just add the profiles to the 
            # previous ones.
            if val_sep_base_data is None:
                val_sep_base_data = user_past_array
                val_sep_target_data = user_current_array
            else:
                val_sep_base_data = sparse.vstack([val_sep_base_data, user_past_array])
                val_sep_target_data = sparse.vstack([val_sep_target_data, user_current_array])
                
        # We save the validation data and the users for each profile.
        val_sep_base_data = val_sep_base_data.todense()
        val_sep_target_data = val_sep_target_data.todense()
        dataframe_val_sep_base_data = pd.DataFrame(val_sep_base_data)
        dataframe_val_sep_base_data.to_csv(f'{self.dataset_path}/preprocessed/val_sep_base_data.csv', index=False)
        dataframe_val_sep_target_data = pd.DataFrame(val_sep_target_data)
        dataframe_val_sep_target_data.to_csv(f'{self.dataset_path}/preprocessed/val_sep_target_data.csv', index=False)
        dataframe_val_sep_users = pd.DataFrame(val_sep_users)
        dataframe_val_sep_users.to_csv(f'{self.dataset_path}/preprocessed/val_sep_users.csv', index=False)
    
    
    def create_separated_test_data(self):
        # We use the test data dropping duplicates.
        data_te_sep = self.data_te_sep
        data_te_sep = data_te_sep.drop_duplicates(subset=['UserID', 'MovieID'])
        data_te_sep = data_te_sep.sort_values(by=['Timestamp', 'MovieID'])
        data_te_sep = torch.tensor(data_te_sep.to_numpy())
        lists_tr_sep = copy.deepcopy(self.lists_tr_sep)
        lists_val_sep = copy.deepcopy(self.lists_val_sep)
        lists_te_sep = copy.deepcopy(self.lists_te_sep)
        
        # We identify test users. 
        test_sep_users = [tensor[0].item() for tensor in data_te_sep]
        
        # We count the number of instances of the test set.
        n_instances = len(data_te_sep)
        
        test_sep_base_data = None
        test_sep_target_data = None
        for i in tqdm(range(n_instances)):
            user_id = data_te_sep[i][0].item()
            item_id = data_te_sep[i][1].item()
            # We find the index of the item in the test history of the user.
            test_item_index = lists_te_sep[user_id].index(item_id)
            # We consider the past history of the user up to the considered item.
            # This history includes both the train and the validation sets.
            user_past_history = lists_tr_sep[user_id] + lists_val_sep[user_id] + lists_te_sep[user_id][:test_item_index]
            # We consider the current history of the user that considers also the current item.
            user_current_history = lists_tr_sep[user_id] + lists_val_sep[user_id] + lists_te_sep[user_id][:test_item_index + 1]
            # We build user profiles with past and current history.
            user_past_array = self.create_user_sparse_array(user_id, user_past_history, self.n_categories, self.padding_number)
            user_current_array = self.create_user_sparse_array(user_id, user_current_history, self.n_categories, self.padding_number)
            
            # If we haven't built any profile yet, we assign the profiles to
            # the test data. Otherwise we just add the profiles to the previous
            # ones.
            if test_sep_base_data is None:
                test_sep_base_data = user_past_array
                test_sep_target_data = user_current_array
            else:
                test_sep_base_data = sparse.vstack([test_sep_base_data, user_past_array])
                test_sep_target_data = sparse.vstack([test_sep_target_data, user_current_array])
                
        # We save the test data and the users for each profile.
        test_sep_base_data = test_sep_base_data.todense()
        test_sep_target_data = test_sep_target_data.todense()
        dataframe_test_sep_base_data = pd.DataFrame(test_sep_base_data)
        dataframe_test_sep_base_data.to_csv(f'{self.dataset_path}/preprocessed/test_sep_base_data.csv', index=False)
        dataframe_test_sep_target_data = pd.DataFrame(test_sep_target_data)
        dataframe_test_sep_target_data.to_csv(f'{self.dataset_path}/preprocessed/test_sep_target_data.csv', index=False)
        dataframe_test_sep_users = pd.DataFrame(test_sep_users)
        dataframe_test_sep_users.to_csv(f'{self.dataset_path}/preprocessed/test_sep_users.csv', index=False)
        
        
        
    #################  WINDOWED USERS DATASET PREPROCESSING  ###############
    def create_user_windowed_sparse_array(self, user_id, history, n_categories, padding_number):
        history_copy = copy.deepcopy(history)
        data_full = copy.deepcopy(self.data_full)
        
        # We limit user interactions to the last 68 items.
        history_copy = history_copy[-68:]
        
        # We remove paddings from user history.
        while padding_number in history_copy:
            history_copy.remove(padding_number)
        
        # We get the relevant data for user and items.
        # We assume that each user interacts with each item only one time.
        user_data = data_full[data_full['UserID'] == user_id]
        selected_rows_history = user_data[user_data['MovieID'].isin(history_copy)]
        profile_movies = self.movies_data[self.movies_data['MovieID'].isin(history_copy)]
        
        # We merge the relevant items with their categories. Then we drop useless attributes.
        merged_df = pd.merge(selected_rows_history, profile_movies, on='MovieID', how='inner')
        merged_df  = merged_df.drop('Timestamp', axis=1) 
        merged_df  = merged_df.drop('UserID', axis=1)
        merged_df  = merged_df.drop('CategoryID', axis=1)
        
        # We assign the rating of the item to each of its categories. 
        for i in range(n_categories):
            category_col = f'Category_{i}'
            rating_col = 'Rating'
            merged_df[category_col] *= merged_df[rating_col]
        merged_df = merged_df.drop('Rating', axis=1)
        merged_df = merged_df.drop_duplicates()
        
        # We compute the mean value for each category. This is the user profile.
        categories = [f'Category_{i}' for i in range(n_categories)]
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
            
        # We convert the user profile to a sparse array.
        #array = sparse.csr_array(category_values, dtype='float32', shape=(1, n_categories))
        array = sparse.csr_matrix(category_values, dtype='float32', shape=(1, n_categories))

        return array
    
    
    def create_windowed_train_data(self):
        # We use the train data dropping duplicates.
        data_tr = self.data_tr
        data_tr = data_tr.drop_duplicates(subset=['UserID', 'MovieID'])
        data_tr = data_tr.sort_values(by=['Timestamp', 'MovieID'])
        data_tr = torch.tensor(data_tr.to_numpy())
        lists_tr = copy.deepcopy(self.lists_tr)
        
        # We identify train users. 
        train_users = [tensor[0].item() for tensor in data_tr]
        
        # We count the number of instances of the train set.
        n_instances = len(data_tr)
        
        train_base_data = None
        train_target_data = None
        for i in tqdm(range(n_instances)):
            user_id = data_tr[i][0].item()
            item_id = data_tr[i][1].item()
            # We find the index of the item in the train history of the user.
            train_item_index = lists_tr[user_id].index(item_id)
            # We consider the past train history of the user up to the considered item.
            user_past_history = lists_tr[user_id][:train_item_index]
            # We consider the current train history of the user that considers also the current item.
            user_current_history = lists_tr[user_id][:train_item_index + 1]
            # We build user profiles with past and current history.
            user_past_array = self.create_user_windowed_sparse_array(user_id, user_past_history, self.n_categories, self.padding_number)
            user_current_array = self.create_user_windowed_sparse_array(user_id, user_current_history, self.n_categories, self.padding_number)
            
            # If we haven't built any profile yet, we assign the profiles to
            # the train data. Otherwise we just add the profiles to the previous
            # ones.
            if train_base_data is None:
                train_base_data = user_past_array
                train_target_data = user_current_array
            else:
                train_base_data = sparse.vstack([train_base_data, user_past_array])
                train_target_data = sparse.vstack([train_target_data, user_current_array])
                
        # We save the train data and the users for each profile.
        train_base_data = train_base_data.todense()
        train_target_data = train_target_data.todense()
        dataframe_train_base_data = pd.DataFrame(train_base_data)
        dataframe_train_base_data.to_csv(f'{self.dataset_path}/preprocessed/train_win_base_data.csv', index=False)
        dataframe_train_target_data = pd.DataFrame(train_target_data)
        dataframe_train_target_data.to_csv(f'{self.dataset_path}/preprocessed/train_win_target_data.csv', index=False)
        dataframe_train_users = pd.DataFrame(train_users)
        dataframe_train_users.to_csv(f'{self.dataset_path}/preprocessed/train_win_users.csv', index=False)
    
    
    def create_windowed_validation_data(self):
        # We use the validation data dropping duplicates.
        data_val = self.data_val
        data_val = data_val.drop_duplicates(subset=['UserID', 'MovieID'])
        data_val = data_val.sort_values(by=['Timestamp', 'MovieID'])
        data_val = torch.tensor(data_val.to_numpy())
        lists_tr = copy.deepcopy(self.lists_tr)
        lists_val = copy.deepcopy(self.lists_val)
  
        # We identify validation users. 
        val_users = [tensor[0].item() for tensor in data_val]
        
        # We count the number of instances of the validation set.
        n_instances = len(data_val)
        
        val_base_data = None
        val_target_data = None
        for i in tqdm(range(n_instances)):
            user_id = data_val[i][0].item()
            item_id = data_val[i][1].item()
            # We find the index of the item in the validation history of the user.
            val_item_index = lists_val[user_id].index(item_id)
            # We consider the past history of the user up to the considered item.
            # This history includes also the train set.
            user_past_history = lists_tr[user_id] + lists_val[user_id][:val_item_index]
            # We consider the current history of the user that considers also the current item.
            user_current_history = lists_tr[user_id] + lists_val[user_id][:val_item_index + 1]
            # We build user profiles with past and current history.
            user_past_array = self.create_user_windowed_sparse_array(user_id, user_past_history, self.n_categories, self.padding_number)
            user_current_array = self.create_user_windowed_sparse_array(user_id, user_current_history, self.n_categories, self.padding_number)
            
            # If we haven't built any profile yet, we assign the profiles to
            # the validation data. Otherwise we just add the profiles to the 
            # previous ones.
            if val_base_data is None:
                val_base_data = user_past_array
                val_target_data = user_current_array
            else:
                val_base_data = sparse.vstack([val_base_data, user_past_array])
                val_target_data = sparse.vstack([val_target_data, user_current_array])
                
        # We save the validation data and the users for each profile.
        val_base_data = val_base_data.todense()
        val_target_data = val_target_data.todense()
        dataframe_val_base_data = pd.DataFrame(val_base_data)
        dataframe_val_base_data.to_csv(f'{self.dataset_path}/preprocessed/val_win_base_data.csv', index=False)
        dataframe_val_target_data = pd.DataFrame(val_target_data)
        dataframe_val_target_data.to_csv(f'{self.dataset_path}/preprocessed/val_win_target_data.csv', index=False)
        dataframe_val_users = pd.DataFrame(val_users)
        dataframe_val_users.to_csv(f'{self.dataset_path}/preprocessed/val_win_users.csv', index=False)
    
       
    def create_windowed_test_data(self):
        # We use the test data dropping duplicates.
        data_te = self.data_te
        data_te = data_te.drop_duplicates(subset=['UserID', 'MovieID'])
        data_te = data_te.sort_values(by=['Timestamp', 'MovieID'])
        data_te = torch.tensor(data_te.to_numpy())
        lists_tr = copy.deepcopy(self.lists_tr)
        lists_val = copy.deepcopy(self.lists_val)
        lists_te = copy.deepcopy(self.lists_te)
        
        # We identify test users. 
        test_users = [tensor[0].item() for tensor in data_te]
        
        # We count the number of instances of the test set.
        n_instances = len(data_te)
        
        test_base_data = None
        test_target_data = None
        for i in tqdm(range(n_instances)):
            user_id = data_te[i][0].item()
            item_id = data_te[i][1].item()
            # We find the index of the item in the test history of the user.
            test_item_index = lists_te[user_id].index(item_id)
            # We consider the past history of the user up to the considered item.
            # This history includes both the train and the validation sets.
            user_past_history = lists_tr[user_id] + lists_val[user_id] + lists_te[user_id][:test_item_index]
            # We consider the current history of the user that considers also the current item.
            user_current_history = lists_tr[user_id] + lists_val[user_id] + lists_te[user_id][:test_item_index + 1]
            # We build user profiles with past and current history.
            user_past_array = self.create_user_windowed_sparse_array(user_id, user_past_history, self.n_categories, self.padding_number)
            user_current_array = self.create_user_windowed_sparse_array(user_id, user_current_history, self.n_categories, self.padding_number)
            
            # If we haven't built any profile yet, we assign the profiles to
            # the test data. Otherwise we just add the profiles to the previous
            # ones.
            if test_base_data is None:
                test_base_data = user_past_array
                test_target_data = user_current_array
            else:
                test_base_data = sparse.vstack([test_base_data, user_past_array])
                test_target_data = sparse.vstack([test_target_data, user_current_array])
                
        # We save the test data and the users for each profile.
        test_base_data = test_base_data.todense()
        test_target_data = test_target_data.todense()
        dataframe_test_base_data = pd.DataFrame(test_base_data)
        dataframe_test_base_data.to_csv(f'{self.dataset_path}/preprocessed/test_win_base_data.csv', index=False)
        dataframe_test_target_data = pd.DataFrame(test_target_data)
        dataframe_test_target_data.to_csv(f'{self.dataset_path}/preprocessed/test_win_target_data.csv', index=False)
        dataframe_test_users = pd.DataFrame(test_users)
        dataframe_test_users.to_csv(f'{self.dataset_path}/preprocessed/test_win_users.csv', index=False)
        
        
    #################  WINDOWED SEPARATED USERS DATASET PREPROCESSING  #########
    def create_windowed_separated_train_data(self):
        # We use the train data dropping duplicates.
        data_tr_sep = self.data_tr_sep
        data_tr_sep = data_tr_sep.drop_duplicates(subset=['UserID', 'MovieID'])
        data_tr_sep = data_tr_sep.sort_values(by=['Timestamp', 'MovieID'])
        data_tr_sep = torch.tensor(data_tr_sep.to_numpy())
        lists_tr_sep = copy.deepcopy(self.lists_tr_sep)
        
        # We identify train users. 
        train_sep_users = [tensor[0].item() for tensor in data_tr_sep]
        
        # We count the number of instances of the train set.
        n_instances = len(data_tr_sep)
        
        train_sep_base_data = None
        train_sep_target_data = None
        for i in tqdm(range(n_instances)):
            user_id = data_tr_sep[i][0].item()
            item_id = data_tr_sep[i][1].item()
            # We find the index of the item in the train history of the user.
            train_item_index = lists_tr_sep[user_id].index(item_id)
            # We consider the past train history of the user up to the considered item.
            user_past_history = lists_tr_sep[user_id][:train_item_index]
            # We consider the current train history of the user that considers also the current item.
            user_current_history = lists_tr_sep[user_id][:train_item_index + 1]
            # We build user profiles with past and current history.
            user_past_array = self.create_user_windowed_sparse_array(user_id, user_past_history, self.n_categories, self.padding_number)
            user_current_array = self.create_user_windowed_sparse_array(user_id, user_current_history, self.n_categories, self.padding_number)
            
            # If we haven't built any profile yet, we assign the profiles to
            # the train data. Otherwise we just add the profiles to the previous
            # ones.
            if train_sep_base_data is None:
                train_sep_base_data = user_past_array
                train_sep_target_data = user_current_array
            else:
                train_sep_base_data = sparse.vstack([train_sep_base_data, user_past_array])
                train_sep_target_data = sparse.vstack([train_sep_target_data, user_current_array])
                
        # We save the train data and the users for each profile.
        train_sep_base_data = train_sep_base_data.todense()
        train_sep_target_data = train_sep_target_data.todense()
        dataframe_train_sep_base_data = pd.DataFrame(train_sep_base_data)
        dataframe_train_sep_base_data.to_csv(f'{self.dataset_path}/preprocessed/train_win_sep_base_data.csv', index=False)
        dataframe_train_sep_target_data = pd.DataFrame(train_sep_target_data)
        dataframe_train_sep_target_data.to_csv(f'{self.dataset_path}/preprocessed/train_win_sep_target_data.csv', index=False)
        dataframe_train_sep_users = pd.DataFrame(train_sep_users)
        dataframe_train_sep_users.to_csv(f'{self.dataset_path}/preprocessed/train_win_sep_users.csv', index=False)
    
    
    def create_windowed_separated_validation_data(self):
        # We use the validation data dropping duplicates.
        data_val_sep = self.data_val_sep
        data_val_sep = data_val_sep.drop_duplicates(subset=['UserID', 'MovieID'])
        data_val_sep = data_val_sep.sort_values(by=['Timestamp', 'MovieID'])
        data_val_sep = torch.tensor(data_val_sep.to_numpy())
        lists_tr_sep = copy.deepcopy(self.lists_tr_sep)
        lists_val_sep = copy.deepcopy(self.lists_val_sep)
  
        # We identify validation users. 
        val_sep_users = [tensor[0].item() for tensor in data_val_sep]
        
        # We count the number of instances of the validation set.
        n_instances = len(data_val_sep)
        
        val_sep_base_data = None
        val_sep_target_data = None
        for i in tqdm(range(n_instances)):
            user_id = data_val_sep[i][0].item()
            item_id = data_val_sep[i][1].item()
            # We find the index of the item in the validation history of the user.
            val_item_index = lists_val_sep[user_id].index(item_id)
            # We consider the past history of the user up to the considered item.
            # This history includes also the train set.
            user_past_history = lists_tr_sep[user_id] + lists_val_sep[user_id][:val_item_index]
            # We consider the current history of the user that considers also the current item.
            user_current_history = lists_tr_sep[user_id] + lists_val_sep[user_id][:val_item_index + 1]
            # We build user profiles with past and current history.
            user_past_array = self.create_user_windowed_sparse_array(user_id, user_past_history, self.n_categories, self.padding_number)
            user_current_array = self.create_user_windowed_sparse_array(user_id, user_current_history, self.n_categories, self.padding_number)
            
            # If we haven't built any profile yet, we assign the profiles to
            # the validation data. Otherwise we just add the profiles to the 
            # previous ones.
            if val_sep_base_data is None:
                val_sep_base_data = user_past_array
                val_sep_target_data = user_current_array
            else:
                val_sep_base_data = sparse.vstack([val_sep_base_data, user_past_array])
                val_sep_target_data = sparse.vstack([val_sep_target_data, user_current_array])
                
        # We save the validation data and the users for each profile.
        val_sep_base_data = val_sep_base_data.todense()
        val_sep_target_data = val_sep_target_data.todense()
        dataframe_val_sep_base_data = pd.DataFrame(val_sep_base_data)
        dataframe_val_sep_base_data.to_csv(f'{self.dataset_path}/preprocessed/val_win_sep_base_data.csv', index=False)
        dataframe_val_sep_target_data = pd.DataFrame(val_sep_target_data)
        dataframe_val_sep_target_data.to_csv(f'{self.dataset_path}/preprocessed/val_win_sep_target_data.csv', index=False)
        dataframe_val_sep_users = pd.DataFrame(val_sep_users)
        dataframe_val_sep_users.to_csv(f'{self.dataset_path}/preprocessed/val_win_sep_users.csv', index=False)
    
    
    def create_windowed_separated_test_data(self):
        # We use the test data dropping duplicates.
        data_te_sep = self.data_te_sep
        data_te_sep = data_te_sep.drop_duplicates(subset=['UserID', 'MovieID'])
        data_te_sep = data_te_sep.sort_values(by=['Timestamp', 'MovieID'])
        data_te_sep = torch.tensor(data_te_sep.to_numpy())
        lists_tr_sep = copy.deepcopy(self.lists_tr_sep)
        lists_val_sep = copy.deepcopy(self.lists_val_sep)
        lists_te_sep = copy.deepcopy(self.lists_te_sep)
        
        # We identify test users. 
        test_sep_users = [tensor[0].item() for tensor in data_te_sep]
        
        # We count the number of instances of the test set.
        n_instances = len(data_te_sep)
        
        test_sep_base_data = None
        test_sep_target_data = None
        for i in tqdm(range(n_instances)):
            user_id = data_te_sep[i][0].item()
            item_id = data_te_sep[i][1].item()
            # We find the index of the item in the test history of the user.
            test_item_index = lists_te_sep[user_id].index(item_id)
            # We consider the past history of the user up to the considered item.
            # This history includes both the train and the validation sets.
            user_past_history = lists_tr_sep[user_id] + lists_val_sep[user_id] + lists_te_sep[user_id][:test_item_index]
            # We consider the current history of the user that considers also the current item.
            user_current_history = lists_tr_sep[user_id] + lists_val_sep[user_id] + lists_te_sep[user_id][:test_item_index + 1]
            # We build user profiles with past and current history.
            user_past_array = self.create_user_windowed_sparse_array(user_id, user_past_history, self.n_categories, self.padding_number)
            user_current_array = self.create_user_windowed_sparse_array(user_id, user_current_history, self.n_categories, self.padding_number)
            
            # If we haven't built any profile yet, we assign the profiles to
            # the test data. Otherwise we just add the profiles to the previous
            # ones.
            if test_sep_base_data is None:
                test_sep_base_data = user_past_array
                test_sep_target_data = user_current_array
            else:
                test_sep_base_data = sparse.vstack([test_sep_base_data, user_past_array])
                test_sep_target_data = sparse.vstack([test_sep_target_data, user_current_array])
                
        # We save the test data and the users for each profile.
        test_sep_base_data = test_sep_base_data.todense()
        test_sep_target_data = test_sep_target_data.todense()
        dataframe_test_sep_base_data = pd.DataFrame(test_sep_base_data)
        dataframe_test_sep_base_data.to_csv(f'{self.dataset_path}/preprocessed/test_win_sep_base_data.csv', index=False)
        dataframe_test_sep_target_data = pd.DataFrame(test_sep_target_data)
        dataframe_test_sep_target_data.to_csv(f'{self.dataset_path}/preprocessed/test_win_sep_target_data.csv', index=False)
        dataframe_test_sep_users = pd.DataFrame(test_sep_users)
        dataframe_test_sep_users.to_csv(f'{self.dataset_path}/preprocessed/test_win_sep_users.csv', index=False)
        
        
        
        
