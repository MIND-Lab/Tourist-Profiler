import math
import random
import torch
import torch.nn as nn
import torch.optim as optim

# Note: The replay buffer is implemented for experiments with the DQN algorithm.
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.num_elements = 0
        self.states = torch.tensor([])
        self.actions = torch.tensor([])
        self.rewards = torch.tensor([])
        self.next_states = torch.tensor([])
        self.stored_objects = [self.states, self.actions, self.rewards, self.next_states]

    
    def push(self, state, action, reward, next_state):
        pushed_objects = [state, action, reward, next_state]

        if self.num_elements == 0:
            for j in range(len(self.stored_objects)):
                self.stored_objects[j] = pushed_objects[j]
            self.num_elements += pushed_objects[0].shape[0]

        # If the replay buffer isn't full we append the new data. Otherwise
        # we replace the oldest stored elements with new ones. The new elements
        # are added at the end.
        elif self.num_elements >= 1 and self.num_elements < self.capacity:
            for j in range(len(self.stored_objects)):
                self.stored_objects[j] = torch.cat((self.stored_objects[j], pushed_objects[j]), dim=0)
            self.num_elements += pushed_objects[0].shape[0]
        else:
            for j in range(len(self.stored_objects)):
                self.stored_objects[j] = self.stored_objects[j][pushed_objects[j].shape[0]:]
                self.stored_objects[j] = torch.cat((self.stored_objects[j], pushed_objects[j]), dim=0)
                    
            
    def sample(self, batch_size):
        random_indices = random.sample(range(self.num_elements), batch_size)
        
        self.states = self.stored_objects[0]
        self.actions = self.stored_objects[1]
        self.rewards = self.stored_objects[2]
        self.next_states = self.stored_objects[3]

        sampled_states = self.states[random_indices]
        sampled_actions = torch.squeeze(self.actions[random_indices], 1)
        sampled_rewards = torch.squeeze(self.rewards[random_indices], 1)
        sampled_next_states = self.next_states[random_indices]
        
        return sampled_states, sampled_actions, sampled_rewards, sampled_next_states



class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, args):
        super(RNN, self).__init__()
        self.device = args.device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.gru(x, h0) 
        # The output is the sequence at the last timestep. 
        out = self.fc(out[:, -1, :])
        return out.to(self.device)



class DQN(nn.Module):
    def __init__(self, state_size, hidden_size_0, hidden_size_1, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size_0)
        nn.init.uniform_(
            self.fc1.weight, 
            a= - (6.0 / (state_size + hidden_size_0)) ** 0.5, 
            b= + (6.0 / (state_size + hidden_size_0)) ** 0.5
        )
        self.fc2 = nn.Linear(hidden_size_0, hidden_size_1)
        nn.init.uniform_(
            self.fc2.weight, 
            a= - (6.0 / (hidden_size_0 + hidden_size_1)) ** 0.5, 
            b= + (6.0 / (hidden_size_0 + hidden_size_1)) ** 0.5
        )
        self.fc3 = nn.Linear(hidden_size_1, output_size)
        nn.init.uniform_(
            self.fc3.weight, 
            a= - (6.0 / (hidden_size_1 + output_size)) ** 0.5, 
            b= + (6.0 / (hidden_size_1 + output_size)) ** 0.5
        )
        
    
    def forward(self, state):
        hidden = torch.relu(self.fc1(state))
        hidden = torch.relu(self.fc2(hidden))
        return self.fc3(hidden)



class Agent(nn.Module):
    def __init__(self, args):
        super(Agent, self).__init__()
        self.learning_rate = args.agent_learning_rate
        self.tau = args.agent_tau
        self.gamma = args.agent_gamma
        self.epsilon = args.agent_epsilon
        self.epsilon_decay = args.agent_epsilon_decay
        self.initial_epsilon = args.agent_initial_epsilon
        self.final_epsilon = args.agent_final_epsilon
        self.rnn_input_size = args.agent_rnn_input_size
        self.rnn_hidden_size = args.agent_rnn_hidden_size
        self.rnn_output_size = args.agent_rnn_output_size
        self.rnn_num_layers = args.agent_rnn_num_layers
        self.dqn_state_size = args.agent_state_size
        self.dqn_hidden_size_0 = args.agent_hidden_size_0
        self.dqn_hidden_size_1 = args.agent_hidden_size_1
        self.dqn_output_size = args.agent_output_size
        self.replay_buffer_size = args.agent_replay_buffer_size
        
        self.rnn = RNN(self.rnn_input_size, self.rnn_hidden_size, self.rnn_output_size, self.rnn_num_layers, args).to(args.device)
        self.active_network = DQN(self.dqn_state_size, self.dqn_hidden_size_0, self.dqn_hidden_size_1, self.dqn_output_size).to(args.device)
        self.target_network = DQN(self.dqn_state_size, self.dqn_hidden_size_0, self.dqn_hidden_size_1, self.dqn_output_size).to(args.device)
 
        self.rnn_params = [param for param in self.rnn.parameters()]
        self.active_network_params = [param for param in self.active_network.parameters()]
        self.target_network_params = [param for param in self.target_network.parameters()]
        self.model_params = self.rnn_params + self.active_network_params

        self.model_optimizer = torch.optim.Adam(self.model_params, lr=self.learning_rate)
 
        self.replay_buffer = ReplayBuffer(capacity=self.replay_buffer_size)
        
        self.softmax = nn.Softmax(dim=0)
        self.mse_loss = nn.MSELoss()
        
        self.hard_update_target_network()

     
    def update_tau(self, tau):
        self.tau = tau


    def update_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.active_network_params, lr=self.learning_rate)


    def forward_active_network(self, state):
        # The forward function of the DQN network is called with the state as input.
        action_values = self.active_network(state)
        return action_values
    
    
    def forward_target_network(self, state):
        # The forward function of the DQN network is called with the state as input.
        action_values = self.target_network(state)
        return action_values
    
    # Returns the action with the highest Q-value.
    def get_active_action(self, state, test=False):
        batch_size = state.shape[0]
        
        actions = []
        for i in range(batch_size):
            user_state = state[i]
            random_number = torch.rand(1)
            if random_number < self.epsilon and test == False:
                action_values = torch.rand(self.dqn_output_size)
            else:
                action_values = self.forward_active_network(user_state)
            action = torch.argmax(action_values)
            actions.append(action)
            
        actions = torch.vstack(actions)
        return actions
        
    
    def get_target_action(self, state, test=False):
        batch_size = state.shape[0]
        
        actions = []
        for i in range(batch_size):
            user_state = state[i]
            random_number = torch.rand(1)
            if random_number < self.epsilon and test == False:
                action_values = torch.rand(self.dqn_output_size)
            else:
                action_values = self.forward_target_network(user_state)
            action = torch.argmax(action_values)
            actions.append(action)
        
        actions = torch.vstack(actions)
            
        return actions    
        
     
    def train_active_network(self, state, action, reward, next_state):
        # Here action must specify the indices of the selected actions. In this
        # way we consider only the Q-values for the selected actions.
        # Another way to implement this is using a one-hot encoding.
        
        # The Q-values for the selected actions are extracted from the active network. 
        q_active = self.forward_active_network(state).gather(1, action.unsqueeze(1))
        
        # The Q-values for the next state are extracted from the target network.
        q_next = self.forward_target_network(next_state)
        max_q_next = torch.max(q_next, dim=-1).values
        max_q_next = torch.reshape(max_q_next, (-1, 1))

        reward = torch.reshape(reward, (-1, 1))
        q_target = reward + (self.gamma * max_q_next)
        
        loss = self.mse_loss(q_active, q_target)
        self.model_optimizer.zero_grad()
        loss.backward(retain_graph=True) 
        self.model_optimizer.step()

        return loss.item()
        
 
    def get_loss(self, state, action, reward, next_state):
        action = action.squeeze()
        q_active = self.forward_active_network(state).gather(1, action.unsqueeze(1))
        q_next = self.forward_target_network(next_state)
        
        max_q_next = torch.max(q_next, dim=-1).values
        max_q_next = torch.reshape(max_q_next, (-1, 1))
        reward = torch.reshape(reward, (-1, 1))
        q_target = reward + (self.gamma * max_q_next)
        loss = self.mse_loss(q_active, q_target)

        return loss.item()
    

    def soft_update_target_network(self):
        # The target network is updated with the active network parameters.
        for i in range(len(self.target_network_params)):
            self.target_network_params[i].data = self.active_network_params[i].data * self.tau +  self.target_network_params[i].data * (1 - self.tau)

 
    def hard_update_target_network(self):
        for i in range(len(self.target_network_params)):
            self.target_network_params[i].data = self.active_network_params[i].data

 
    def update_epsilon(self, epoch, num_epochs):
        self.epsilon = self.final_epsilon + (self.initial_epsilon - self.final_epsilon) * math.exp(-1. * epoch / self.epsilon_decay)
        
    
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
        
        