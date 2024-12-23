import torch

class Setting():
    def __init__(self):
        ## Hyper-parameters for the Agent.
        self.agent_learning_rate = 0.001
        self.agent_tau = 0.1
        self.agent_gamma = 0.9
        self.agent_epsilon = 0.9
        self.agent_epsilon_decay = 0.999
        self.agent_initial_epsilon = 0.9
        self.agent_final_epsilon = 0.1
        self.agent_rnn_input_size = 7 # *21
        self.agent_rnn_hidden_size = 128
        self.agent_rnn_output_size = 20
        self.agent_rnn_num_layers = 2
        self.agent_state_size = 26 # 6 category + 20 state size created by RNN
        self.agent_hidden_size_0 = 256
        self.agent_hidden_size_1 = 128
        self.agent_output_size = 5
        self.agent_replay_buffer_size = 1024
        self.agent_action_rating_profile = True
        self.agent_path = "./checkpoints/agent/"

        ## Hyper-parameters about the dataset.
        self.dataset_path = '../dataset/movielens-100k'
        self.train_limit = 10000
        self.test_limit = 1000
        self.batch_size = 32
        self.num_epochs = 5
        self.window = 10
        self.sep_test_users = True
        self.windowed_dataset = True
        self.fast_testing = False
        self.device = torch.device('cpu')
