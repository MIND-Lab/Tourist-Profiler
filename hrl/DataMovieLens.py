import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

'''
Code adapted from: https://github.com/jerryhao66/HRL
'''


class Dataset(object):
    def __init__(self, data_path, num_negatives, batch_size, fast_running=False):
        self.data_path = data_path
        self.batch_size = int(batch_size)
        self.num_negatives = int(num_negatives)
        self.data_full, self.data_tr, self.data_te, self.num_users, self.num_items_dataset, self.num_items, self.padding_number = self.load_dataset_as_dataframe()
        self.lists_full, self.lists_tr, self.lists_te = self.load_dataset_as_lists()
        self.trainMatrix = self.load_training_file_as_matrix()
        self.trainList = self.load_training_file_as_list()
        self.testRatings = self.load_rating_file_as_list()
        
        negative_file_path = f"{self.data_path}/preprocessed/test_negatives.csv"
        if os.path.exists(negative_file_path):
            self.testNegatives = self.load_negative_file()
        else:
            self.save_negative_file()
            self.testNegatives = self.load_negative_file()
        
        if fast_running:
            self.testRatings = self.testRatings[:2000]
            self.testNegatives = self.testNegatives[:2000]
        

    ########################  load data from the file #########################
    def load_dataset_as_dataframe(self):
        raw_data = self._preprocess_dataset()
        
        self.n_categories = len(np.unique(raw_data['CategoryID'].values))
        num_users = len(np.unique(raw_data['UserID'].values))
        num_items_dataset = len(np.unique(raw_data['MovieID'].values))
        
        unique_movie_id = pd.unique(raw_data['MovieID'])
        unique_user_id = pd.unique(raw_data['UserID'])
        movie2id = dict((movie_id, i) for (i, movie_id) in enumerate(unique_movie_id))
        user2id = dict((uid, i) for (i, uid) in enumerate(unique_user_id))
        
        data_full = self._create_dataframe_with_mapping(raw_data, user2id, movie2id)
        
        data_full.to_csv(f"{self.data_path}/preprocessed/data_full.csv", index=False)
        
        data_tr, data_te = self._split_train_test(data_full)

        # We also consider the padding number.
        num_items = num_items_dataset
        padding_number = num_items
        
        return data_full, data_tr, data_te, num_users, num_items_dataset, num_items, padding_number
    
    
    def _preprocess_dataset(self):
        # We load the ratings dataset.
        ratings = pd.read_csv(
            f'{self.data_path}/u.data', 
            sep='\t', 
            engine='python', 
            encoding='latin-1', 
            names=['UserID', 'MovieID', 'Rating', 'Timestamp']
        )
        
        # We load the movies dataset.
        movies = pd.read_csv(
            f'{self.data_path}/u.item', 
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
        
        # We find the number of users for the whole dataset.
        self.n_users = len(np.unique(merged_df['UserID'].values))
        
        return merged_df
        
        
    def _create_dataframe_with_mapping(self, in_dataframe, user2id, movie2id):
        in_dataframe["UserID"] = in_dataframe["UserID"].map(user2id)
        in_dataframe["MovieID"] = in_dataframe["MovieID"].map(movie2id)
        out_dataframe = in_dataframe
        return out_dataframe
    
    
    def _split_train_test(self, data, test_proportion=0.3, val_proportion=0.5):
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
        data_tr.to_csv(f"{self.data_path}/preprocessed/data_tr.csv", index=False)
        data_val.to_csv(f"{self.data_path}/preprocessed/data_val.csv", index=False)
        data_te.to_csv(f"{self.data_path}/preprocessed/data_te.csv", index=False)
        data_tr_bal.to_csv(f"{self.data_path}/preprocessed/data_tr_bal.csv", index=False)
        
        self.data_tr_bal = data_tr_bal
        self.data_val = data_val

        return data_tr, data_te
    
    
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
            
        return lists_full, lists_tr, lists_te
    
    
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
    
    
    def load_training_file_as_matrix(self):
        data_tr = self.data_tr
        data_tr['BinaryRating'] = (data_tr['Rating'] > 0).astype(int)
        mat = sp.dok_matrix((self.num_users, self.num_items_dataset), dtype=np.float32)
        for _, row in data_tr.iterrows():
            user_id = row['UserID']
            movie_id = row['MovieID']
            rating = row['Rating']
            if rating > 0:
                mat[user_id, movie_id] = 1
        return mat
    
    
    def load_training_file_as_list(self):
        return self.lists_tr
    
    
    def load_rating_file_as_list(self):
        data_te = self.data_te
        testRatings = []
        for _, row in data_te.iterrows():
            user_id = row['UserID']
            movie_id = row['MovieID']
            testRatings.append([user_id, movie_id])
        return testRatings
    
    
    def save_negative_file(self):
        data_full = self.data_full
        
        # Dictionary for the movies rated by users.
        user_movie_dict = {}
        for _, row in data_full.iterrows():
            user_id = row['UserID']
            movie_id = row['MovieID']

            if user_id not in user_movie_dict:
                user_movie_dict[user_id] = set()
            user_movie_dict[user_id].add(movie_id)

        testNegatives = []

        # We generate 99 random movies that users haven't rated.
        for idx in tqdm(range(len(self.testRatings))):
            user_id = self.testRatings[idx][0]
            voted_movies = user_movie_dict[user_id]
            all_movies = set(data_full['MovieID'].unique())
            non_voted_movies = list(all_movies - voted_movies)
            random_non_voted_movies = np.random.choice(non_voted_movies, 99, replace=False)
            testNegatives.append(list(random_non_voted_movies))
            
        testNegativesDataframe = pd.DataFrame(testNegatives)
        testNegativesDataframe.to_csv(f"{self.data_path}/preprocessed/test_negatives.csv", index=False)    


    def load_negative_file(self):
        testNegativesDataframe = pd.read_csv(f"{self.data_path}/preprocessed/test_negatives.csv")
        testNegatives = testNegativesDataframe.values.tolist()
        return testNegatives
                
        
    ##################### generate positive instances
    def get_positive_instances(self):
        p_user_input_list, p_num_idx_list, p_item_input_list, p_labels_list, p_user_idx = [], [], [], [], []
        p_batch_num = int(len(self.trainList) / self.batch_size)
        for batch in range(p_batch_num):
            u, n, i, l, u_idx = self._get_positive_batch(batch)
            p_user_input_list.append(u)
            p_num_idx_list.append(n)
            p_item_input_list.append(i)
            p_labels_list.append(l)
            p_user_idx.append(u_idx)
        p_user_input_list = np.array(p_user_input_list)
        p_num_idx_list = np.array(p_num_idx_list)
        p_item_input_list = np.array(p_item_input_list)
        p_labels_list = np.array(p_labels_list)
        p_user_idx = np.array(p_user_idx)

        return [p_user_input_list, p_num_idx_list, p_item_input_list, p_labels_list, p_batch_num, p_user_idx]

    def _get_positive_batch(self,i):
        user, number, item, label, user_idx = [], [], [], [], []
        padding_number = self.trainMatrix.shape[1]
        begin = i * self.batch_size
        for idx in range(begin, begin + self.batch_size):
            sample = self.trainList[idx]
            i_i = sample[-1]
            sample.pop()
            u_i = sample
            user.append(u_i)
            number.append(len(u_i))
            item.append(i_i)
            label.append(1)
            user_idx.append(idx)
        user_input = self._add_mask(padding_number, user, max(number))
        return user_input, number, item, label, user_idx

    
    ################# generate positive/negative instances for training
    def get_dataset_with_neg(self):  # negative sampling and shuffle the data
        self._get_train_data_fixed()
        iterations = len(self.user_input_with_neg)
        self.index_with_neg = np.arange(iterations)
        self.num_batch_with_neg = iterations / self.batch_size
        return self._preprocess(self._get_train_batch_fixed)


    def _preprocess(self, get_train_batch):  # generate the masked batch list
        user_input_list, num_idx_list, item_input_list, labels_list, user_idx_list = [], [], [], [], []

        for i in range(int(self.num_batch_with_neg)):
            ui, ni, ii, l, u_idx = get_train_batch(i)
            user_input_list.append(ui)
            num_idx_list.append(ni)
            item_input_list.append(ii)
            labels_list.append(l)
            user_idx_list.append(u_idx)

        return [user_input_list, num_idx_list, item_input_list, labels_list, self.num_batch_with_neg, user_idx_list]


    def _get_train_data_fixed(self):
        self.user_input_with_neg, self.item_input_with_neg, self.labels_with_neg = [], [], []
        for u in range(len(self.trainList)):
            i = self.trainList[u][-1]
            self.user_input_with_neg.append(u)
            self.item_input_with_neg.append(i)
            self.labels_with_neg.append(1)
            # negative instances
            for t in range(self.num_negatives):
                j = np.random.randint(self.num_items)
                while j in self.trainList[u]:
                    j = np.random.randint(self.num_items)
                self.user_input_with_neg.append(u)
                self.item_input_with_neg.append(j)
                self.labels_with_neg.append(0)


    def _get_train_batch_fixed(self, i):
        user_list, num_list, item_list, labels_list, user_idx_list = [], [], [], [], []
        trainList = self.trainList
        begin = i * self.batch_size
        for idx in range(begin, begin + self.batch_size):
            user_idx = self.user_input_with_neg[self.index_with_neg[idx]]
            item_idx = self.item_input_with_neg[self.index_with_neg[idx]]
            nonzero_row = []
            nonzero_row += self.trainList[user_idx]
            num_list.append(self._remove_item(self.num_items, nonzero_row, nonzero_row[-1]))
            user_list.append(nonzero_row)
            item_list.append(item_idx)
            labels_list.append(self.labels_with_neg[self.index_with_neg[idx]])
            user_idx_list.append(user_idx)
        user_input = self._add_mask(self.num_items, user_list, max(num_list))
        num_idx = num_list
        item_input = item_list
        labels = labels_list
        user_idx = user_idx_list
        return (user_input, num_idx, item_input, labels, user_idx)


    def _remove_item(self,feature_mask, users, item):
        flag = 0
        for i in range(len(users)):
            if users[i] == item:
                users[i] = users[-1]
                users[-1] = feature_mask
                flag = 1
                break
        return len(users) - flag


    def _add_mask(self, feature_mask, features, num_max):
        for i in range(len(features)):
            features[i] = features[i] + [feature_mask] * (num_max + 1 - len(features[i]))
        return features


 ################# generate positive/negative instances for test
    def get_test_instances(self):
        test_user_input, test_num_idx, test_item_input, test_labels, test_user_idx, test_instance_idx = [], [], [], [], [], []

        for idx in range(len(self.testRatings)):
            test_user = self.testRatings[idx][0]
            rating = self.testRatings[idx][:] 
            items = self.testNegatives[idx][:]
            user = self.trainList[test_user][:]

            items.append(rating[1]) # add positive instance at the end of the negative instances 
            num_idx = np.full(len(items), len(user), dtype=np.int32) # the length of historical items are all the same, equaling to the length of the historical items of the positive instance
            user_input = np.tile(user, (len(items), 1)) # historical items are the same for the positive and negative instances
            item_input = np.array(items)
            labels = np.zeros(len(items)) 
            labels[-1] = 1 # the last label for the positive instance is 1

            test_user_input.append(user_input)
            test_num_idx.append(num_idx)
            test_item_input.append(item_input)
            test_labels.append(labels)
            test_user_idx.append(test_user)
            test_instance_idx.append(idx)
            
        test_num_batches = len(self.testRatings)
        
        return [test_user_input, test_num_idx, test_item_input, test_labels, test_num_batches, test_user_idx, test_instance_idx]
    
    
    def _create_test_batch(self, input_list, batch_size):
        batches = [np.array(input_list[i:i + batch_size]) for i in range(0, len(input_list), batch_size)]
        return batches
    
    
    def _create_user_history_test_batch(self, input_list, batch_size):
        batches = []
        num_batches = int(len(input_list) / batch_size)
        batched_input = num_batches * batch_size
        
        # We create batches that have a number of instances equalling the batch size.
        for i in range(0, batched_input, batch_size):
            batch_user_history = []
            batch_len_user_history = []
            for j in range(batch_size):
                batch_user_history.append(input_list[i + j].tolist())
                batch_len_user_history.append(len(input_list[i + j][0]))
            max_len_user_history_batch = max(batch_len_user_history)
            new_batch_user_history = []
            for user_history in batch_user_history:
                new_user_history = self._add_padding(user_history, max_len_user_history_batch)
                new_batch_user_history.append(new_user_history)
            batches.append(new_batch_user_history)
            
        # We add the rest of the instances that exceed the batch size.  
        i = batched_input
        batch_user_history = []
        batch_len_user_history = []
        for j in range(len(input_list) - batched_input):
            batch_user_history.append(input_list[i + j].tolist())
            batch_len_user_history.append(len(input_list[i + j]))
        max_len_user_history_batch = max(batch_len_user_history)
        new_batch_user_history = []
        for user_history in batch_user_history:
            new_user_history = self._add_padding(user_history, max_len_user_history_batch)
            new_batch_user_history.append(new_user_history)
        batches.append(new_batch_user_history)
        batches = np.array(batches)    
        
        return batches
    
    
    def _add_padding(self, user_history, max_len_user_history):
        for i in range(len(user_history)):
            user_history[i] = user_history[i] + [self.padding_number] * (max_len_user_history + 1 - len(user_history[i]))
        return user_history
        
    

