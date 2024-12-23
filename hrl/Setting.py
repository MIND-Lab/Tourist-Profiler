'''
Code adapted from: https://github.com/jerryhao66/HRL
'''

class setting():
    def __init__(self):
        ############ hyper-parameters for recommender ###########
        self.recommender_epochs = 10
        self.recommender_lr = 0.02
        self.recommender_tau = 0.0005
        self.algorithm = 0
        self.recommender_weight_size = 16
        self.recommender_embedding_size = 16
        self.regs = [1e-6, 1e-6, 1e-6]
        self.alpha = 0
        self.beta = 0.5


        ############# hyper-parameters for agent ################
        self.agent_epochs = 10
        self.agent_pretrain_lr = 0.001
        self.agent_pretrain_tau = 1
        self.agent_lr = 0.0005
        self.agent_tau = 0.0005
        self.high_state_size = 18
        self.low_state_size = 34
        self.agent_weight_size = 8
        self.sample_cnt = 3


        ############# hyper-parameres about the dataset ##############
        self.num_neg = 4
        self.datapath = './Data'
        self.batch_size = 235
        self.agent_pretrain = True
        self.recommender_pretrain = True
        self.pre_agent = "./Checkpoint/pre-agent/"
        self.pre_recommender = "./Checkpoint/pre-recommender/"
        self.agent = "./Checkpoint/agent/"
        self.recommender = "./Checkpoint/recommender/"
        self.sep_test_users = False
        self.windowed_dataset = True
        self.fast_running = False
        self.null_values_model = "multivae"
        self.agent_verbose = 1
        self.recommender_verbose = 3

