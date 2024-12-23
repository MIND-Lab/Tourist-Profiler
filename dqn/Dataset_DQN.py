import os
import numpy as np
import pandas as pd
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
        # To reproduce the experiments you need to modify the code below.    
        #############################  MODIFY HERE  #############################  
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
        #############################  MODIFY HERE  #############################
        
        
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
    

