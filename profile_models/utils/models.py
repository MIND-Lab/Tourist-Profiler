import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional
from collections import Counter


class MultiVAE(nn.Module):
    def __init__(self, p_dims, q_dims=None, lam=0.01, lr=1e-3, dropout=0.5):
        """
        Liang, D., Krishnan, R. G., Hoffman, M. D., and Jebara,
        T. Variational autoencoders for collaborative filtering. In Proceedings
        of the 2018 World Wide Web Conference (2018), WWW ’18, International
        World Wide Web Conferences Steering Committee.
        Code adapted from: https://github.com/dawenl/vae_cf
        """
        super().__init__()

        self.p_dims = p_dims
        if q_dims is None:
            self.q_dims = p_dims[::-1]
        else:
            assert q_dims[0] == p_dims[-1], "Input and output dimension must equal each other for autoencoders."
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q-network mismatches."
            self.q_dims = q_dims

        self.lam = lam
        self.lr = lr
        self.q_layers = self.construct_q_network()
        self.p_layers = self.construct_p_network()
        self.dropout = nn.Dropout(dropout)


    def compute_category_weights(self):
        total_samples = self.category_count_train.sum(dim=0)

        category_weights = []
        for count in self.category_count_train:
            # len(self.category_count_train) = 18 for MovieLens
            weight = 1 / (len(self.category_count_train) * count / total_samples)
            category_weights.append(weight)
            
        for i in range(len(category_weights)):
            if np.isinf(category_weights[i]):
                category_weights[i] = 0
        category_weights = torch.Tensor(category_weights)

        return category_weights
    
    
    def get_category_weights(self):
        return self.category_weights
    
    
    def set_category_weights(self, category_weights):
        self.category_weights = category_weights
        

    def construct_q_network(self):
        # q is the encoder
        encoder = []
        for i, (d_in, d_out) in enumerate(zip(self.q_dims[:-1], self.q_dims[1:])):
            if i == len(self.q_dims[:-1]) - 1:
                # We need 2 sets of parameters for mean and variance.
                d_out *= 2
            layer = nn.Linear(d_in, d_out)
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.normal_(layer.bias, mean=0, std=0.001)
            encoder.append(layer)
        return nn.ModuleList(encoder)
    

    def construct_p_network(self):
        # p is the decoder
        decoder = []
        for i, (d_in, d_out) in enumerate(zip(self.p_dims[:-1], self.p_dims[1:])):
            layer = nn.Linear(d_in, d_out)
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.normal_(layer.bias, mean=0, std=0.001)
            decoder.append(layer)
        return nn.ModuleList(decoder)


    def forward(self, input):
        input = input.type(torch.FloatTensor)
        mu, logvar = self.encode(input)
        z = self.sample(mu, logvar)
        logits = self.decode(z)
        
        return logits, mu, logvar
    

    def encode(self, input):
        # l2 normalization.
        h = functional.normalize(input, p=2, dim=1)
        h = self.dropout(h)
 
        for i, layer in enumerate(self.q_layers):
            # In this case we don't need torch.matmul(h, w) + b because
            # we have linear layers that already incorporate this multiplication.   
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = torch.relu(h)
            else:
                # mu is the mean of the Gaussian distribution.
                mu = h[:, :self.q_dims[-1]]
                # logvar is the logarithm of the variance of the Gaussian distribution.
                # The logarithm of the variance is better than the variance since
                # it cannot have negative values.
                logvar = h[:, self.q_dims[-1]:]
        return mu, logvar


    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        # Reparametrization trick.
        z = mu + epsilon * std
        return z


    def decode(self, z):
        # z is the latent representation.
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = torch.relu(h)
        return h
    

def logistic_function(x):
        return torch.div(1, 1 + torch.exp(-x))   


def multivae_loss_function(recon_x, x, mu, logvar, anneal=1.0, type=None, category_weights=None):
    if type == None or type == 'none' or type == 'original_loss':
        # Original loss

        # Kullback-Leibler divergence.
        KL = torch.mean(torch.sum(0.5 * (-logvar + torch.exp(logvar) + mu ** 2 - 1), dim=1))

        # Log softmax.
        log_softmax_var = functional.log_softmax(recon_x, dim=1)

        # Negative log-likelihood.
        neg_ll = - torch.mean(torch.sum(log_softmax_var * x, dim=-1))

        # In PyTorch the L2 regularization is implemented as the weight decay for
        # the Adam optimizer. So we don't need to specify this regularization here.
        neg_ELBO = neg_ll + anneal * KL 

        return neg_ELBO
    
    elif type == 'cross-entropy_loss':
        # Cross-entropy loss
        
        loss = torch.nn.functional.cross_entropy(input=recon_x, target=x)

        return loss
    
    elif type == 'cross-entropy_plus_kl_loss':
        # Cross entropy-loss + KL

        KL = torch.mean(torch.sum(0.5 * (-logvar + torch.exp(logvar) + mu ** 2 - 1), dim=1))

        loss = torch.nn.functional.cross_entropy(input=recon_x, target=x)

        neg_ELBO = loss + anneal * KL 

        return neg_ELBO
    
    elif type == 'mse_loss':
        # MSE loss

        loss = torch.nn.functional.mse_loss(input=recon_x, target=x)

        return loss
    
    elif type == 'mse_plus_kl_loss':
        # MSE loss + KL

        KL = torch.mean(torch.sum(0.5 * (-logvar + torch.exp(logvar) + mu ** 2 - 1), dim=1))

        loss = torch.nn.functional.mse_loss(input=recon_x, target=x)

        neg_ELBO = loss + anneal * KL 

        return neg_ELBO
        
    elif type == 'l1_loss':
        # L1 loss

        loss = torch.nn.functional.l1_loss(input=recon_x, target=x)

        return loss
        
    elif type == 'l1_plus_kl_loss':
        # L1 loss + KL

        KL = torch.mean(torch.sum(0.5 * (-logvar + torch.exp(logvar) + mu ** 2 - 1), dim=1))

        loss = torch.nn.functional.l1_loss(input=recon_x, target=x)

        neg_ELBO = loss + anneal * KL 

        return neg_ELBO
    
    elif type == 'gaussian_loss':
        # Gaussian likelihood

        c = 1 + 40 * x
        diff = x - recon_x
        square_diff = diff ** 2
        loss = torch.mean(torch.sum(torch.div(torch.sum(square_diff), 2) * c))

        return loss
    
    elif type == 'gaussian_plus_kl_loss':
        # Gaussian likelihood + KL

        KL = torch.mean(torch.sum(0.5 * (-logvar + torch.exp(logvar) + mu ** 2 - 1), dim=1))

        c = 1 + 40 * x
        diff = x - recon_x
        square_diff = diff ** 2
        neg_ll = torch.mean(torch.sum(torch.div(torch.sum(square_diff), 2) * c))

        neg_ELBO = neg_ll + anneal * KL 

        return neg_ELBO
        
    elif type == 'logistic_loss':
        # Logistic likelihood

        log_sigma = torch.log(logistic_function(recon_x))
        log_comp_sigma = torch.log(logistic_function(1 - recon_x))
        log_term = log_sigma + (1 - x) * log_comp_sigma
        loss = - torch.mean(torch.sum(log_term * x, dim=-1))

        return loss
    
    elif type == 'logistic_plus_kl_loss':
        # Logistic likelihood + KL

        KL = torch.mean(torch.sum(0.5 * (-logvar + torch.exp(logvar) + mu ** 2 - 1), dim=1))

        log_sigma = torch.log(logistic_function(recon_x))
        log_comp_sigma = torch.log(logistic_function(1 - recon_x))
        log_term = log_sigma + (1 - x) * log_comp_sigma
        neg_ll = - torch.mean(torch.sum(log_term * x, dim=-1))
        neg_ELBO = neg_ll + anneal * KL 

        return neg_ELBO

    elif type == 'weighted_cross-entropy_loss' and category_weights != None:
        # Weighted cross-entropy loss
        
        loss = torch.nn.functional.cross_entropy(input=recon_x, target=x, weight=category_weights, reduction='none')

        return torch.mean(loss)
    
    elif type == 'weighted_cross-entropy_loss2' and category_weights != None:
        # Weighted cross-entropy loss 2
        
        batch_size = recon_x.shape[0]
        
        category_weights = category_weights.repeat(batch_size,1)
        
        # Log softmax.
        log_softmax_var = functional.log_softmax(recon_x, dim=1)

        # Negative log-likelihood.
        neg_ll = - torch.mean(torch.sum((category_weights * log_softmax_var) * x), dim=-1)
        
        return neg_ll

    elif type == 'focal_cross-entropy_loss' and category_weights != None:
        # Focal cross-entropy loss
        
        # Softmax.
        softmax_var = functional.softmax(recon_x, dim=1)

        ce = torch.nn.functional.cross_entropy(input=recon_x, target=x, reduction='none')
        
        # Get 'top true class' from each row.
        top_categories_ground_truth = torch.argmax(x, dim=1)
        
        pt = []
        category_weight = []
        for i in range(len(top_categories_ground_truth)):
            pt.append(softmax_var[i][top_categories_ground_truth[i]])
            category_weight.append(category_weights[top_categories_ground_truth[i]])
        
        pt = torch.Tensor(pt)
        focal_term = (1 - pt)**2.0
        loss = focal_term * ce
        loss *= torch.Tensor(category_weight)
        
        return torch.mean(loss)
    
    elif type == 'focal_cross-entropy_loss2' and category_weights != None:
        # Focal cross-entropy loss
        
        # Log softmax.
        log_softmax_var = functional.log_softmax(recon_x, dim=1)

        ce = torch.nn.functional.cross_entropy(input=log_softmax_var, target=x, weight=category_weights, reduction='none')
    
        # Get 'top true class' from each row.
        top_categories_ground_truth = torch.argmax(x, dim=1)
        
        log_pt = []
        for i in range(len(top_categories_ground_truth)):
            log_pt.append(log_softmax_var[i][top_categories_ground_truth[i]])
        
        # Focal term (1 - pt)^gamma, with gamma = 2.0.
        pt = torch.exp(torch.Tensor(log_pt))
        focal_term = (1 - pt)**2.0
        loss = focal_term * ce

        return torch.mean(loss)
    
    else:
        return 0
    
    
class CustomMSE:
    def __init__(self, type, reduction=torch.mean):
        super().__init__()
        self.type = type
        self.reduction = reduction
        
    def __call__(self, output, target, mu=None, logvar=None, anneal=1.0):
        # We compute the loss between the predictions (output) and the ground truth (target).
        loss = torch.nn.functional.mse_loss(input=output, target=target, reduction="none")
        # We keep the computed loss only for the values in the target that are not zero.
        # We consider the zero values as unknown values.
        loss[target == 0] = 0
        loss = self.reduction(loss) 
        # If specified, we add KL divergence as a regularization term.
        if self.type == "mse_plus_kl_loss":
            KL = torch.mean(torch.sum(0.5 * (-logvar + torch.exp(logvar) + mu ** 2 - 1), dim=1))
            loss = loss + anneal * KL
        return loss   
    
    
    
class EASE(nn.Module):
    def __init__(self, lambda_p):
        """
        Steck, H. Embarrassingly shallow autoencoders for sparse data. In
        The World Wide Web Conference (2019), WWW ’19, Association for
        Computing Machinery.
        """

        super().__init__()
        self.lambda_p = lambda_p
        
    
    def get_category_weights(self):
        return None


    def fit(self, input):
        X = input
        
        # We compute the Gram matrix.
        G = torch.matmul(X.T, X) 
        G += self.lambda_p * torch.eye(G.shape[0], dtype=G.dtype)
        # We compute the inverse of the Gram matrix.
        P = torch.linalg.inv(G)
        B = -P / torch.diag(P) 
        # We fill the diagonal of the weight matrix B with zeros.
        B.fill_diagonal_(0.0)
        self.B = B

        return


    def predict(self, input, user_id):
        preds = []

        num_rows, num_cols = input.size()
        for i in range(num_rows):
            user_tensor = input[i]
            user_tensor = np.array(user_tensor)
            # We multiply the user profile with the weight matrix B.
            preds_tensor = user_tensor.dot(self.B)
            preds.append(preds_tensor)

        preds = np.array(preds)

        return torch.tensor(preds)
    


class TopPopularityModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    
    def get_category_weights(self):
        return None


    def fit(self, input):
        dense_matrix = input
        
        # We compute the most popular rating for each category.
        self.category_ratings_dict = {}
        self.category_most_pop = []
        for i in range(dense_matrix.shape[1]):
            category_ratings = dense_matrix[:, i].tolist()
            # We keep only the non-zero values.
            category_ratings = [value for value in category_ratings if value != 0.0]
            # If we don't have non-zero values, we consider a 0.0 value that will
            # be the category rating.
            if len(category_ratings) == 0:
                category_ratings = [0.0]
            self.category_ratings_dict[i] = category_ratings
            count = Counter(self.category_ratings_dict[i])
            # We use the most common value for each category.
            value = count.most_common(1)[0][0]
            self.category_most_pop.append(value)
            
        return


    def predict(self, input):
        preds = []
        # For each category we pick the most common value in the train set.
        for i in range(len(input)):
            preds.append(self.category_most_pop)

        return torch.tensor(preds)
    
    
       
class UserKNNModel(nn.Module):
    def __init__(self, k, similarity):
        '''
        Resnick, P., Iacovou, N., Suchak, M., Bergstrom, P., and
        Riedl, J. Grouplens: an open architecture for collaborative filtering
        of netnews. In Proceedings of the 1994 ACM Conference on Computer
        Supported Cooperative Work (1994), CSCW ’94, Association for Computing
        Machinery.
        '''
        super().__init__()

        self.k = k
        self.similarity = similarity 
        
        
    def get_category_weights(self):
        return None


    def fit(self, input):
        self.pop_items = input.sum(axis=0).tolist()[0]
        dense_matrix = input
        
        self.category_ratings_dict = {}
        self.category_most_pop = []
        for i in range(dense_matrix.shape[1]):
            category_ratings = dense_matrix[:, i].tolist()
            self.category_ratings_dict[i] = category_ratings
            count = Counter(self.category_ratings_dict[i])
            # We use the most common value for each category.
            value = count.most_common(1)[0][0]
            self.category_most_pop.append(value)
        self.users_matrix = input

        return


    def predict(self, input, user_id):
        preds = []
        
        n_users = input.shape[0]

        self.users_matrix = np.array(self.users_matrix)
        input = np.array(input)
 
        similar_users_matrix, similarity_weights = find_k_similar_users(self.users_matrix, input, self.k, self.similarity, user_id)
        
        similar_users_matrix = np.array(similar_users_matrix)
        similarity_weights = np.array(similarity_weights)

        for i in range(n_users):
            values_similar_users = []
            user_vector = np.array(input[i])
            zero_vector = all(value == 0 for value in user_vector)
            # If the user hasn't rated any category, we can't find similar users.
            # Therefore we recommend the most popular categories.
            if zero_vector:
                preds = []
                for i in range(len(input)):
                    preds.append(self.category_most_pop)
            else:
                for j in range(self.k):
                    similar_user = similar_users_matrix[i][j]
                    mean_similar_user = np.mean(similar_user)
                    value_similar_user = (similar_user - mean_similar_user)
                    value_similar_user = similarity_weights[i][j] * value_similar_user
                    values_similar_users.append(value_similar_user)
                    
                normalization_term = 1 / np.sum(similarity_weights[i])
                
                mean_user = np.mean(user_vector)
                sum_similar_users = np.sum(values_similar_users, axis=0)
                normalized_sum = normalization_term * sum_similar_users
                pred = mean_user + normalized_sum
                pred = np.clip(pred, 0, 5)
                preds.append(pred)
                
        # Creating a tensor from a list of numpy.ndarrays is extremely slow. 
        # Thus we convert the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
        preds = torch.tensor(np.array(preds))

        return preds


def find_k_similar_users(matrix, user_vectors, k, similarity, user_id):
    most_similar_vectors_list = []
    similarity_weights_list = []
    matrix = np.array(matrix)
    user_vectors = np.array(user_vectors)
    
    for i in range(user_vectors.shape[0]):
        user_vector = user_vectors[i]
        
        # We find the indices for which the current user and each row of the matrix
        # have non-zero values.
        matching_indices = [
            [j for j in range(len(user_vector)) if user_vector[j] != 0 and row[j] != 0]
            for row in matrix
        ]
        
        # We keep only the non-zero values for which there are matching indices
        # between the current user and the users' matrix. 
        selected_matrix = []
        selected_user = []
        for j in range(len(matrix)):
            selected_matrix_row = matrix[j]
            selected_matrix_row = selected_matrix_row[matching_indices[j]]
            selected_user_row = user_vector[matching_indices[j]]
            selected_matrix.append(selected_matrix_row)
            selected_user.append(selected_user_row)
        selected_matrix = np.array(selected_matrix, dtype=np.object0)
        selected_user = np.array(selected_user, dtype=np.object0)
        
        similarities = []
        for j in range(len(selected_matrix)):
            # If both the current user and the compared user have no
            # non-zero values, then we assign a low similarity instead of 0.
            # If we have no non-zero common values for all the other users, 
            # having the same similarity -1000 will allow picking random 
            # similar users.
            if len(selected_user[j]) == 0:
                similarity_value = -1000
            else:
                # If there is only one non-zero value we need to create a list
                # n order to compute the euclidean distance.
                if len(selected_matrix[j].shape) == 0:
                    euclidean_distance = np.linalg.norm(selected_user[j] - [selected_matrix[j]])
                else:
                    euclidean_distance = np.linalg.norm(selected_user[j] - selected_matrix[j])
                similarity_value = 1 / (1 + euclidean_distance)
            
            similarities.append(similarity_value)
            
        # We use a low value for the similarity with the current user 
        # because we want to find the profiles of the most similar 
        # users, excluding the current user.
        # We use -3000 instead of -1000 to prevent selecting the same user
        # in any case.
        similarities[user_id] = -3000
        
        similarities = torch.tensor(similarities)
        
        _, most_similar_indices = similarities.sort(dim=0, descending=True)
        most_similar_indices = most_similar_indices[:k]
        most_similar_vectors = matrix[most_similar_indices]
        
        similarity_weights = []
        for j in most_similar_indices:
            similarity_weights.append(similarities[j].item())
        most_similar_vectors_list.append(most_similar_vectors)
        similarity_weights_list.append(similarity_weights)
        
    return most_similar_vectors_list, similarity_weights_list
    
    
class MatrixFactorization(nn.Module):
    def __init__(self, n_users=943, n_topics=18, emb_dim=20, init=True, bias=True, sigmoid=True):
        super().__init__()
        self.bias = bias  # Enable/disable bias
        self.sigmoid = sigmoid  # Enable/disable sigmoid function to constrain predictions between 0 and 5
        self.user_emb = nn.Embedding(n_users, emb_dim).requires_grad_(True)  # Embedding for users
        self.topic_emb = nn.Embedding(n_topics, emb_dim).requires_grad_(True)  # Embedding for topics
        self.n_users = n_users
        self.n_topics = n_topics

        # If bias is enabled, initialize biases for users and topics and a global offset
        if bias:
            self.user_bias = nn.Parameter(torch.zeros(n_users))  # User biases
            self.topic_bias = nn.Parameter(torch.zeros(n_topics))  # Topic biases
            self.offset = nn.Parameter(torch.zeros(1))  # Global offset

        # If init is True, initialize embeddings with random values in a narrow range
        if init:
            self.user_emb.weight.data.uniform_(0., 0.05)
            self.topic_emb.weight.data.uniform_(0., 0.05)

    # Forward function to make prediction given user and topic embeddings
    def forward(self, data):
        user, topic = data[:, 0], data[:, 1]  # Extract user and topic IDs from data batch
        
        user = torch.clamp(user, 0, self.n_users - 1)  # Limit user IDs to a valid range
        topic = torch.clamp(topic, 0, self.n_topics - 1)

        # Extract user and topic embeddings
        user_emb = self.user_emb(user)
        topic_emb = self.topic_emb(topic)
        
        # Compute the element-wise product of embeddings and then sum
        element_product = (user_emb * topic_emb).sum(1)

        # If bias is enabled, add user and topic biases and the global offset
        if self.bias:
            user_b = self.user_bias[user]
            topic_b = self.topic_bias[topic]
            element_product += user_b + topic_b + self.offset

        # If sigmoid is enabled, apply it to constrain predictions between 0 and 5
        if self.sigmoid:
            return torch.sigmoid(element_product) * 5.0
        
        return element_product
        

    # Function to make predictions given user and topic IDs
    def predict(self, user_id, topic_id):
        user_tensor = torch.tensor([user_id])
        topic_tensor = torch.tensor([topic_id])
   
        # Use forward function to compute prediction
        prediction = self.forward(torch.stack((user_tensor, topic_tensor), dim=1))

        return prediction  # Return prediction value
    
    
    
        