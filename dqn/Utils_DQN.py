import os
import sys
import tqdm
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append('.')
sys.path.append('../../')
sys.path.append('../profile_models/')

notebook_path = os.getcwd()
project_path = os.path.abspath(os.path.join(notebook_path, '..'))
sys.path.append(project_path)

from profile_models.utils import metrics

def train(env, agent, args):
    device = args.device
    if args.sep_test_users:
        train_base_data = env.train_sep_base_data
        train_target_data = env.train_sep_target_data
        train_seq_base_data = env.train_sep_seq_base_data
        train_seq_target_data = env.train_sep_seq_target_data
    else:
        train_base_data = env.train_base_data
        train_target_data = env.train_target_data
        train_seq_base_data = env.train_seq_base_data
        train_seq_target_data = env.train_seq_target_data
    train_users = [int(tensor[0].item()) for tensor in train_seq_base_data]
    train_items = [int(tensor[1].item()) for tensor in train_seq_base_data]
    train_users = torch.tensor(train_users, dtype=torch.int32).to(device)
    train_items = torch.tensor(train_items, dtype=torch.int32).to(device)
    N = len(train_seq_base_data)
    idxs = list(range(N))
    batch_size = args.batch_size

    train_losses, train_rewards = [], []
    true_0, true_1, true_2, true_3, true_4, true_5 = [], [], [], [], [], []
    pred_0, pred_1, pred_2, pred_3, pred_4, pred_5 = [], [], [], [], [], []
    for batch_idx, start_idx in tqdm(enumerate(range(0, N, batch_size))):
        end_idx = min(start_idx + batch_size, N)
        batch_train_users = train_users[idxs[start_idx:end_idx]]
        batch_train_items = train_items[idxs[start_idx:end_idx]]
        batch_train_base_data = train_base_data[idxs[start_idx:end_idx]]
        batch_train_target_data = train_target_data[idxs[start_idx:end_idx]]
        batch_train_seq_target_data = train_seq_target_data[idxs[start_idx:end_idx]]

        # We get the states from the environment.
        batch_state, batch_next_state, batch_categories, _, _ = env.get_states(batch_train_users, batch_train_items)

        # We get the action values from the agent. # ! The action of the model infers the value of a topic.
        batch_action = agent.get_active_action(batch_state, test=False).to(device)
        
        # We get the reward from the environment.
        if args.agent_action_rating_profile == True:
            batch_target_profile_value = batch_train_target_data * batch_categories
            batch_target_profile_value = torch.sum(batch_target_profile_value, dim=1)
            batch_target_profile_value = torch.reshape(batch_target_profile_value, (-1, 1)).to(device) 
            batch_reward = env.get_reward(batch_target_profile_value, batch_action).to(device)
        else:
            batch_reward = env.get_reward(batch_train_seq_target_data, batch_action).to(device)
        
        batch_action = torch.squeeze(batch_action, 1)
        
        # We compute the loss.
        loss = agent.train_active_network(batch_state, batch_action, batch_reward, batch_next_state)
        train_losses.append(loss)
        batch_reward = torch.squeeze(batch_reward, 1)
        batch_reward = torch.mean(batch_reward, dtype=torch.float32).item()
        train_rewards.append(batch_reward)
        
        batch_action = batch_action + 1
        pred_0.append(torch.sum(torch.eq(batch_action, 0)).item())
        pred_1.append(torch.sum(torch.eq(batch_action, 1)).item())
        pred_2.append(torch.sum(torch.eq(batch_action, 2)).item())
        pred_3.append(torch.sum(torch.eq(batch_action, 3)).item())
        pred_4.append(torch.sum(torch.eq(batch_action, 4)).item())
        pred_5.append(torch.sum(torch.eq(batch_action, 5)).item())
        
        mask = batch_categories.bool()
        batch_train_target_data = batch_train_target_data[mask]
        
        batch_train_target_data = torch.reshape(batch_train_target_data, (-1, 1)).to(device)
        true_0.append(torch.sum(torch.eq(batch_train_target_data, 0)).item())
        true_1.append(torch.sum(torch.eq(batch_train_target_data, 1)).item())
        true_2.append(torch.sum(torch.eq(batch_train_target_data, 2)).item())
        true_3.append(torch.sum(torch.eq(batch_train_target_data, 3)).item())
        true_4.append(torch.sum(torch.eq(batch_train_target_data, 4)).item())
        true_5.append(torch.sum(torch.eq(batch_train_target_data, 5)).item())

    agent.soft_update_target_network() # We update the target network.
    
    print("train true_0: ", np.sum(true_0))
    print("train true_1: ", np.sum(true_1))
    print("train true_2: ", np.sum(true_2))
    print("train true_3: ", np.sum(true_3))
    print("train true_4: ", np.sum(true_4))
    print("train true_5: ", np.sum(true_5)) 
    print("train pred_0: ", np.sum(pred_0))
    print("train pred_1: ", np.sum(pred_1))
    print("train pred_2: ", np.sum(pred_2))
    print("train pred_3: ", np.sum(pred_3))
    print("train pred_4: ", np.sum(pred_4))
    print("train pred_5: ", np.sum(pred_5))
           
    return train_losses, train_rewards
    
    
def test(env, agent, args):
    device = args.device
    if args.sep_test_users:
        test_base_data = env.test_sep_base_data
        test_target_data = env.test_sep_target_data
        test_seq_base_data = env.test_sep_seq_base_data
        test_seq_target_data = env.test_sep_seq_target_data
    else: 
        test_base_data = env.test_base_data
        test_target_data = env.test_target_data
        test_seq_base_data = env.test_seq_base_data
        test_seq_target_data = env.test_seq_target_data
    test_users = [int(tensor[0].item()) for tensor in test_seq_base_data]
    test_items = [int(tensor[1].item()) for tensor in test_seq_base_data]
    test_users = torch.tensor(test_users, dtype=torch.int32).to(device)
    test_items = torch.tensor(test_items, dtype=torch.int32).to(device)
    N = len(test_seq_base_data)
    idxs = list(range(N))
    batch_size = args.batch_size
    
    test_base_data, masked_test_ref = env.mask_data(test_base_data, test_target_data)
    
    recon_data = []
    test_losses, test_rewards = [], []
    true_0, true_1, true_2, true_3, true_4, true_5 = [], [], [], [], [], []
    pred_0, pred_1, pred_2, pred_3, pred_4, pred_5 = [], [], [], [], [], []
    print(N)
    for batch_idx, start_idx in tqdm(enumerate(range(0, N, batch_size))):
        end_idx = min(start_idx + batch_size, N)
        
        if start_idx + 1 != end_idx:
            batch_test_users = test_users[idxs[start_idx:end_idx]]
            batch_test_items = test_items[idxs[start_idx:end_idx]]        
            batch_test_base_data = test_base_data[idxs[start_idx:end_idx]]
            batch_test_target_data = test_target_data[idxs[start_idx:end_idx]]
            batch_test_seq_target_data = test_seq_target_data[idxs[start_idx:end_idx]]

            # We get the states from the environment.
            batch_state, batch_next_state, batch_categories, batch_past_rnn_output, _ = env.get_states(batch_test_users, batch_test_items)
            
            # We get the action.
            batch_action = agent.get_active_action(batch_state, test=True).to(device)  
            
            # We get the reward from the environment.
            if args.agent_action_rating_profile == True:
                batch_target_profile_value = batch_test_target_data * batch_categories
                batch_target_profile_value = torch.sum(batch_target_profile_value, dim=1)
                batch_target_profile_value = torch.reshape(batch_target_profile_value, (-1, 1)).to(device) 
                batch_reward = env.get_reward(batch_target_profile_value, batch_action).to(device)
            else:
                batch_reward = env.get_reward(batch_test_seq_target_data, batch_action).to(device)
            
            # We compute the test loss.
            loss = agent.get_loss(batch_state, batch_action, batch_reward, batch_next_state)
            test_losses.append(loss)
            
            # We save the reward.
            reward = torch.mean(batch_reward, dtype=torch.float32).item()
            test_rewards.append(reward)

            # We repeat the state for each category.
            batch_state, batch_n_elements = env.get_repeated_states(batch_past_rnn_output)
            
            # We get the action values from the agent. Then we add 1 in order to
            # obtain a rating since the action values are between 0 and 4, whereas
            # the ratings are from 1 to 5.
            batch_action = agent.get_active_action(batch_state, test=True).to(device)
            batch_action = batch_action + 1
            
            pred_0.append(torch.sum(torch.eq(batch_action, 0)).item())
            pred_1.append(torch.sum(torch.eq(batch_action, 1)).item())
            pred_2.append(torch.sum(torch.eq(batch_action, 2)).item())
            pred_3.append(torch.sum(torch.eq(batch_action, 3)).item())
            pred_4.append(torch.sum(torch.eq(batch_action, 4)).item())
            pred_5.append(torch.sum(torch.eq(batch_action, 5)).item())
            
            batch_test_target_data = torch.reshape(batch_test_target_data, (-1, 1)).to(device)
            true_0.append(torch.sum(torch.eq(batch_test_target_data, 0)).item())
            true_1.append(torch.sum(torch.eq(batch_test_target_data, 1)).item())
            true_2.append(torch.sum(torch.eq(batch_test_target_data, 2)).item())
            true_3.append(torch.sum(torch.eq(batch_test_target_data, 3)).item())
            true_4.append(torch.sum(torch.eq(batch_test_target_data, 4)).item())
            true_5.append(torch.sum(torch.eq(batch_test_target_data, 5)).item())

            # We reshape the ratings in order to obtain a profile for each test instance.
            batch_recon_data = batch_action.view(batch_n_elements, env.n_categories)
            recon_data.append(batch_recon_data)
    recon_data = torch.vstack(recon_data).to(device)

    # We compute the errors.
    rmse_total, rmse_explicit, rmse_implicit = metrics.rmse_with_mask(test_base_data, test_base_data, recon_data, test_target_data, masked_test_ref)
    
    print("test true_0: ", np.sum(true_0))
    print("test true_1: ", np.sum(true_1))
    print("test true_2: ", np.sum(true_2))
    print("test true_3: ", np.sum(true_3))
    print("test true_4: ", np.sum(true_4))
    print("test true_5: ", np.sum(true_5)) 
    print("test pred_0: ", np.sum(pred_0))
    print("test pred_1: ", np.sum(pred_1))
    print("test pred_2: ", np.sum(pred_2))
    print("test pred_3: ", np.sum(pred_3))
    print("test pred_4: ", np.sum(pred_4))
    print("test pred_5: ", np.sum(pred_5))

    return test_losses, test_rewards, rmse_total, rmse_explicit, rmse_implicit
    

def print_epoch_results(epoch, rmse_total, rmse_explicit, rmse_implicit, train_reward, test_reward, train_time, test_time):
    print("="*100)
    print("| Epoch {:d} | Average train reward: {:5.4f} ± {:5.4f} | Average test reward: {:5.4f} ± {:5.4f} | ".
          format(epoch, np.mean(train_reward), np.std(train_reward), 
                 np.mean(test_reward), np.std(test_reward)))
    print("| Total RMSE: {:5.4f} ± {:5.4f} | Explicit RMSE: {:5.4f} ± {:5.4f} | Implicit RMSE: {:5.4f} ± {:5.4f} | ".
          format(torch.mean(rmse_total), torch.std(rmse_total), 
                 torch.mean(rmse_explicit), torch.std(rmse_explicit),
                 torch.mean(rmse_implicit), torch.std(rmse_implicit)))
    print("| Train time: {:5.4f} s. | Test time: {:5.4f} s. |".
          format(train_time, test_time))
    print("="*100)    
    
    
def plot_results(
    train_loss_list, train_rewards_list,
    test_loss_list, test_rewards_list,
    rmse_total_list, rmse_explicit_list, rmse_implicit_list,
    num_epochs
):
    steps_list = np.arange(len(train_loss_list))
    epochs_list = np.arange(num_epochs)
    
    train_loss_array = np.array(train_loss_list)
    train_loss_steps = np.reshape((train_loss_array), train_loss_array.shape[0] * train_loss_array.shape[1])
    train_rewards_array = np.array(train_rewards_list)
    train_rewards_steps = np.reshape((train_rewards_array), train_rewards_array.shape[0] * train_rewards_array.shape[1])
    test_loss_array = np.array(test_loss_list)
    test_loss_steps = np.reshape((test_loss_array), test_loss_array.shape[0] * test_loss_array.shape[1])
    test_rewards_array = np.array(test_rewards_list)
    test_rewards_steps = np.reshape((test_rewards_array), test_rewards_array.shape[0] * test_rewards_array.shape[1])
    
    fig, axs = plt.subplots(1, 1, figsize=(12, 4))
    axs.plot(train_loss_steps, c='r', label='Train loss', linewidth=1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("DQN train loss")
    plt.show()
    
    fig, axs = plt.subplots(1, 1, figsize=(12, 4))
    axs.plot(train_rewards_steps, c='r', label='Train reward', linewidth=1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("DQN train rewards")
    plt.show()
    
    fig, axs = plt.subplots(1, 1, figsize=(12, 4))
    axs.plot(test_loss_steps, c='b', label='Test loss', linewidth=1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("DQN test loss")
    plt.show()
    
    fig, axs = plt.subplots(1, 1, figsize=(12, 4))
    axs.plot(test_rewards_steps, c='b', label='Test reward', linewidth=1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("DQN test rewards")
    plt.show()
    
    mean_train_loss = np.mean(train_loss_list, axis=1)
    std_train_loss = np.std(train_loss_list, axis=1)
    std_low_train_loss = mean_train_loss - std_train_loss
    std_high_train_loss = mean_train_loss + std_train_loss

    mean_test_loss = np.mean(test_loss_list, axis=1)
    std_test_loss = np.std(test_loss_list, axis=1)
    std_low_test_loss = mean_test_loss - std_test_loss
    std_high_test_loss = mean_test_loss + std_test_loss

    fig, axs = plt.subplots(1, 1, figsize=(12, 4))
    axs.plot(mean_train_loss, c='r', label='Train loss', linewidth=1.0)
    print(epochs_list.shape)
    print(mean_train_loss.shape)
    print(std_low_train_loss.shape)
    axs.fill_between(epochs_list, mean_train_loss, std_low_train_loss, color='r', alpha=.1)
    axs.fill_between(epochs_list, mean_train_loss, std_high_train_loss, color='r', alpha=.1)
    plt.plot(mean_test_loss, c='b', label='Test loss', linewidth=1.0)
    axs.fill_between(epochs_list, mean_test_loss, std_low_test_loss, color='b', alpha=.1)
    axs.fill_between(epochs_list, mean_test_loss, std_high_test_loss, color='b', alpha=.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DQN loss")
    plt.show()
    
    mean_train_rewards = np.mean(train_rewards_list, axis=1)
    std_train_rewards = np.std(train_rewards_list, axis=1)
    std_low_train_rewards = mean_train_rewards - std_train_rewards
    std_high_train_rewards = mean_train_rewards + std_train_rewards

    mean_test_rewards = np.mean(test_rewards_list, axis=1)
    std_test_rewards = np.std(test_rewards_list, axis=1)
    std_low_test_rewards = mean_test_rewards - std_test_rewards
    std_high_test_rewards = mean_test_rewards + std_test_rewards
    
    fig, axs = plt.subplots(1, 1, figsize=(12, 4))
    axs.plot(mean_train_rewards, c='r', label='Train rewards', linewidth=1.0)
    axs.fill_between(epochs_list, mean_train_rewards, std_low_train_rewards, color='r', alpha=.1)
    axs.fill_between(epochs_list, mean_train_rewards, std_high_train_rewards, color='r', alpha=.1)
    plt.plot(mean_test_rewards, c='b', label='Test rewards', linewidth=1.0)
    axs.fill_between(epochs_list, mean_test_rewards, std_low_test_rewards, color='b', alpha=.1)
    axs.fill_between(epochs_list, mean_test_rewards, std_high_test_rewards, color='b', alpha=.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.title("DQN rewards")
    plt.show()
    

    rmse_total_list = torch.vstack(rmse_total_list)
    mean_rmse_total = torch.mean(rmse_total_list, dim=1)
    std_rmse_total = torch.std(rmse_total_list, dim=1)
    std_low_rmse_total = mean_rmse_total - std_rmse_total
    std_high_rmse_total = mean_rmse_total + std_rmse_total

    rmse_explicit_list = torch.vstack(rmse_explicit_list)
    mean_rmse_explicit = torch.mean(rmse_explicit_list, dim=1)
    std_rmse_explicit = torch.std(rmse_explicit_list, dim=1)
    std_low_rmse_explicit = mean_rmse_explicit - std_rmse_explicit
    std_high_rmse_explicit = mean_rmse_explicit + std_rmse_explicit

    rmse_implicit_list = torch.vstack(rmse_implicit_list)
    mean_rmse_implicit = torch.mean(rmse_implicit_list, dim=1)
    std_rmse_implicit = torch.std(rmse_implicit_list, dim=1)
    std_low_rmse_implicit = mean_rmse_implicit - std_rmse_implicit
    std_high_rmse_implicit = mean_rmse_implicit + std_rmse_implicit

    fig, axs = plt.subplots(1, 1, figsize=(12, 4))
    axs.plot(mean_rmse_total, c='r', label='Total RMSE', linewidth=1.0)
    axs.fill_between(epochs_list, mean_rmse_total, std_low_rmse_total, color='r', alpha=.1)
    axs.fill_between(epochs_list, mean_rmse_total, std_high_rmse_total, color='r', alpha=.1)
    plt.plot(mean_rmse_explicit, c='b', label='Explicit RMSE', linewidth=1.0)
    axs.fill_between(epochs_list, mean_rmse_explicit, std_low_rmse_explicit, color='b', alpha=.1)
    axs.fill_between(epochs_list, mean_rmse_explicit, std_high_rmse_explicit, color='b', alpha=.1)
    plt.plot(mean_rmse_implicit, c='g', label='Implicit RMSE', linewidth=1.0)
    axs.fill_between(epochs_list, mean_rmse_implicit, std_low_rmse_implicit, color='g', alpha=.1)
    axs.fill_between(epochs_list, mean_rmse_implicit, std_high_rmse_implicit, color='g', alpha=.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("RMSE with DQN")
    plt.show()
    

def save_results(
    train_losses_list, train_rewards_list,
    test_losses_list, test_rewards_list,
    rmse_total_list, rmse_explicit_list, rmse_implicit_list,
    args
):
    results_path = "./results"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
     
    df_train_losses_list = pd.DataFrame(train_losses_list)
    df_train_rewards_list = pd.DataFrame(train_rewards_list)
    df_test_losses_list = pd.DataFrame(test_losses_list)
    df_test_rewards_list = pd.DataFrame(test_rewards_list)
    rmse_total_list = torch.vstack(rmse_total_list).numpy()
    rmse_explicit_list = torch.vstack(rmse_explicit_list).numpy()
    rmse_implicit_list = torch.vstack(rmse_implicit_list).numpy()
    df_rmse_total_list = pd.DataFrame(rmse_total_list)
    df_rmse_explicit_list = pd.DataFrame(rmse_explicit_list)
    df_rmse_implicit_list = pd.DataFrame(rmse_implicit_list)

    if args.sep_test_users:
        df_train_losses_list.to_csv(f'{results_path}/train_sep_losses_const_item_id.csv', index=False) 
        df_train_rewards_list.to_csv(f'{results_path}/train_sep_rewards_const_item_id.csv', index=False) 
        df_test_losses_list.to_csv(f'{results_path}/test_sep_losses_const_item_id.csv', index=False)  
        df_test_rewards_list.to_csv(f'{results_path}/test_sep_rewards_const_item_id.csv', index=False)  
        df_rmse_total_list.to_csv(f'{results_path}/rmse_total_sep_const_item_id.csv', index=False)  
        df_rmse_explicit_list.to_csv(f'{results_path}/rmse_explicit_sep_const_item_id.csv', index=False) 
        df_rmse_implicit_list.to_csv(f'{results_path}/rmse_implicit_sep_const_item_id.csv', index=False)  
    else:
        df_train_losses_list.to_csv(f'{results_path}/train_losses_const_item_id.csv', index=False) 
        df_train_rewards_list.to_csv(f'{results_path}/train_rewards_const_item_id.csv', index=False) 
        df_test_losses_list.to_csv(f'{results_path}/test_losses_const_item_id.csv', index=False)  
        df_test_rewards_list.to_csv(f'{results_path}/test_rewards_const_item_id.csv', index=False)  
        df_rmse_total_list.to_csv(f'{results_path}/rmse_total_const_item_id.csv', index=False)  
        df_rmse_explicit_list.to_csv(f'{results_path}/rmse_explicit_const_item_id.csv', index=False) 
        df_rmse_implicit_list.to_csv(f'{results_path}/rmse_implicit_const_item_id.csv', index=False)   
    

