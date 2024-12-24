import torch
import numpy as np
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from Evaluation import eval_rating

'''
Code adapted from: https://github.com/jerryhao66/HRL
'''


def print_train_rewards(mean_train_rewards, std_train_rewards):
    mean_train_rewards = np.array(mean_train_rewards)
    std_train_rewards = np.array(std_train_rewards)
    std_low_train_rewards = mean_train_rewards - std_train_rewards
    std_high_train_rewards = mean_train_rewards + std_train_rewards
    epochs_list = np.arange(len(mean_train_rewards))
    
    fig, axs = plt.subplots(1, 1, figsize=(12, 4))
    axs.plot(mean_train_rewards, c='r', label='Train rewards', linewidth=1.0)
    axs.fill_between(epochs_list, mean_train_rewards, std_low_train_rewards, color='r', alpha=.05)
    axs.fill_between(epochs_list, mean_train_rewards, std_high_train_rewards, color='r', alpha=.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.title("HRL train rewards")
    plt.show()


def print_recommender_message(unit, index, hr5, ndcg5, hr10, ndcg10, map, mrr, 
                              test_loss, test_time, train_loss, train_time):
    print("%s %d : HR5 = %.4f, NDCG5 = %.4f, HR10 = %.4f, NDCG10 = %.4f, "
          "MAP = %.4f, MRR = %.4f, test loss = %.4f [%.1fs] train_loss = %.4f "
          "[%.1fs]" % (unit, index, hr5, ndcg5, hr10, ndcg10, map, mrr, 
                       test_loss, test_time, train_loss, train_time))
    

def print_agent_message(epoch, avg_reward, total_selected_items, 
                        total_revised_instances, total_notrevised_instances, 
                        total_deleted_instances, total_keep_instances, 
                        test_time, train_time):
    partial_revised = total_revised_instances-total_deleted_instances-total_keep_instances
    print("Epoch %d : avg reward = %.4f, items (keep = %d), instances "
          "(revise = %d, notrevise = %d, delete = %d, keep = %d, "
          "partial revise = %d), test time = %.1fs, train_time = %.1fs" % 
          (epoch, avg_reward, total_selected_items, total_revised_instances, 
           total_notrevised_instances, total_deleted_instances, 
           total_keep_instances, partial_revised, test_time, train_time))


def _get_high_action(prob, Random):
    batch_size = prob.shape[0]
    if Random:
        random_number = np.random.rand(batch_size)
        return np.where(random_number < prob, np.ones(batch_size,dtype=np.int), 
                        np.zeros(batch_size,dtype=np.int))
    else:
        return np.where(prob >= 0.5, np.ones(batch_size,dtype=np.int), 
                        np.zeros(batch_size,dtype=np.int))


def _get_low_action(prob, user_input_column, padding_number, Random):
    batch_size = prob.shape[0]
    if Random:
        random_number = np.random.rand(batch_size)
        return np.where((random_number < prob) & (user_input_column != padding_number), 
                        np.ones(batch_size,dtype=np.int),
                        np.zeros(batch_size,dtype=np.int))
    else:
        return np.where((prob >= 0.5) & (user_input_column != padding_number), 
                        np.ones(batch_size,dtype=np.int), 
                        np.zeros(batch_size,dtype=np.int))


def sampling_RL(user_input, num_idx, item_input, labels, batch_index, agent, 
                env, padding_number, Random=True):
    batch_size = user_input.shape[0]
    max_item_num = user_input.shape[1]
    env.reset_state(user_input, num_idx, item_input, labels, batch_size, 
                    max_item_num, batch_index)
    high_state = env.get_overall_state()
    high_prob = agent.predict_high_target(high_state)
    high_action = _get_high_action(high_prob, Random)

    for i in range(max_item_num):
        low_state = env.get_state(i)
        low_prob = agent.predict_low_target(low_state)
        low_action = _get_low_action(low_prob, user_input[:, i], padding_number, Random)
        env.update_state(low_action, low_state, i)
    select_user_input, select_num_idx, notrevised_index, revised_index, \
        delete_index, keep_index = env.get_selected_items(high_action)

    return high_action, high_state, select_user_input, select_num_idx, \
        item_input, labels, notrevised_index, revised_index, delete_index, \
        keep_index


def evaluate(agent, recommender, testset, env, padding_number, noAgent=False):
    test_user_input, test_num_idx, test_item_input, \
        test_labels, test_batch_num, test_user_idx, test_instance_idx  = (testset[0], testset[1], testset[2], testset[3], testset[4], testset[5], testset[6])
    if noAgent:
        return eval_rating(recommender, test_user_input, test_num_idx, test_item_input, test_labels, test_batch_num)
    else:
        env.set_test_original_rewards()
        select_user_input_list, select_num_idx_list, select_item_input_list, select_label_list = [],[],[],[]
        positive_user_input_list, positive_select_user_input_list = [], []
        not_revised_profiles, revised_profiles, deleted_profiles = 0, 0, 0
        for i in tqdm(range(test_batch_num)):
            _, _, select_user_input, select_num_idx, select_item_input, select_label_input, _, _, _, _ = sampling_RL(test_user_input[i], test_num_idx[i], test_item_input[i], test_labels[i], i, agent, env, padding_number, Random=False)
            
            select_user_input_list.append(select_user_input)
            select_item_input_list.append(select_item_input)
            select_num_idx_list.append(select_num_idx)
            select_label_list.append(select_label_input)

            t_u = test_user_input[i]
            t_num = test_num_idx[i]
            t_i = test_item_input[i]
            t_label = test_labels[i]
            predictions, attention, loss = recommender.predict_with_attentions(t_u, 
                                                                               np.reshape(t_num,(-1,1)), 
                                                                               np.reshape(t_i,(-1,1)), 
                                                                               np.reshape(t_label,(-1,1)))
            map_item_score = {t_i[j]: predictions[j] for j in range(len(t_i))}
            max_score_item = max(map_item_score, key=lambda k: map_item_score[k])
            max_score_index = list(map_item_score.keys()).index(max_score_item)

            if np.array_equal(test_user_input[i][max_score_index], select_user_input[max_score_index]):
                not_revised_profiles += 1
            else:
                revised_profiles += 1

            revised_array = select_user_input[max_score_index]
            filtered_select_user_input = revised_array[revised_array != padding_number]
            if len(filtered_select_user_input) == 1:
                deleted_profiles += 1

            positive_user_input_list.append(test_user_input[i][max_score_index])
            positive_select_user_input_list.append(select_user_input[max_score_index])
        env.set_train_original_rewards()

        print("Not revised profiles: ", not_revised_profiles)
        print("Revised profiles: ", revised_profiles)
        print("Deleted profiles: ", deleted_profiles)

        rmse_total, rmse_explicit, rmse_implicit = env.get_rmse(positive_select_user_input_list, test_user_idx, test_flag=True)
        print("Total RMSE: {:5.4f} ± {:5.4f}".format(torch.mean(rmse_total, dtype=torch.float64), torch.std(rmse_total)))
        print("Explicit RMSE: {:5.4f} ± {:5.4f}".format(torch.mean(rmse_explicit, dtype=torch.float64), torch.std(rmse_explicit)))
        print("Implicit RMSE: {:5.4f} ± {:5.4f}".format(torch.mean(rmse_implicit, dtype=torch.float64), torch.std(rmse_implicit)))

        return eval_rating(recommender, select_user_input_list, select_num_idx_list, select_item_input_list, select_label_list, test_batch_num)


def train(sess, agent, recommender, trainset, testset, args, env, padding_number, recommender_trainable=True, agent_trainable=True):
    train_user_input, train_num_idx, train_item_input, train_labels, \
        train_batch_num, train_user_idx = (trainset[0], trainset[1], trainset[2], trainset[3], trainset[4], trainset[5])
    sample_times = args.sample_cnt
    high_state_size = args.high_state_size
    low_state_size = args.low_state_size
    avg_loss = 0

    shuffled_batch_indexes = np.random.permutation(int(train_batch_num))
    for batch_index in shuffled_batch_indexes:

        batched_user_input = np.array([u for u in train_user_input[batch_index]])
        batched_item_input = np.reshape(train_item_input[batch_index], (-1, 1))
        batched_label_input = np.reshape(train_labels[batch_index], (-1, 1))
        batched_num_idx = np.reshape(train_num_idx[batch_index], (-1,1))
        batched_user_idx = np.reshape(train_user_idx[batch_index], (-1,1))

        batch_size = batched_user_input.shape[0]
        max_item_num = batched_user_input.shape[1]

        train_loss = 0
        agent.assign_active_high_network()
        agent.assign_active_low_network()
        recommender.assign_active_network()
        if agent_trainable:

            sampled_high_states = np.zeros((sample_times, batch_size, high_state_size), dtype=np.float32)
            sampled_high_actions = np.zeros((sample_times, batch_size), dtype=np.int)

            sampled_low_states = np.zeros((sample_times, batch_size, max_item_num, low_state_size), dtype=np.float32)
            sampled_low_actions = np.zeros((sample_times, batch_size, max_item_num), dtype=np.float32)

            sampled_high_rewards = np.zeros((sample_times, batch_size), dtype=np.float32)
            sampled_low_rewards = np.zeros((sample_times, batch_size), dtype=np.float32)

            sampled_revise_index = []

            avg_high_reward = np.zeros((batch_size), dtype=np.float32)
            avg_low_reward = np.zeros((batch_size), dtype=np.float32)

            for sample_time in range(sample_times):
                high_action, high_state, select_user_input, select_num_idx, \
                    item_input, label_input, notrevised_index, revised_index, \
                    delete_index, keep_index =  sampling_RL(batched_user_input, batched_num_idx, batched_item_input, batched_label_input, batch_index, agent, env, padding_number)
                sampled_high_actions[sample_time, :] = high_action
                sampled_high_states[sample_time, :] = high_state
                sampled_revise_index.append(revised_index)

                _, _, reward = env.get_reward(recommender, batch_index, high_action, 
                                              select_user_input, select_num_idx, 
                                              batched_item_input, batched_label_input, 
                                              batched_user_idx, batched_user_input, 
                                              test_flag=False)
                
                avg_high_reward += reward
                avg_low_reward += reward
                sampled_high_rewards[sample_time, :] = reward
                sampled_low_rewards[sample_time, :] = reward
                sampled_low_actions[sample_time, :] = env.get_action_matrix()
                sampled_low_states[sample_time, :] = env.get_state_matrix()

            avg_high_reward = avg_high_reward / sample_times
            avg_low_reward = avg_low_reward / sample_times
            high_gradbuffer = agent.init_high_gradbuffer()
            low_gradbuffer = agent.init_low_gradbuffer()
            for sample_time in range(sample_times):
                high_reward = np.subtract(sampled_high_rewards[sample_time], avg_high_reward)
                high_gradient = agent.get_high_gradient(sampled_high_states[sample_time], high_reward,sampled_high_actions[sample_time] )
                agent.train_high(high_gradbuffer, high_gradient)

                revised_index = sampled_revise_index[sample_time]
                low_reward = np.subtract(sampled_low_rewards[sample_time], avg_low_reward)
                low_reward_row = np.tile(np.reshape(low_reward[revised_index], (-1, 1)), max_item_num)
                low_gradient = agent.get_low_gradient(
                    np.reshape(sampled_low_states[sample_time][revised_index], (-1, low_state_size)),
                    np.reshape(low_reward_row, (-1,)),
                    np.reshape(sampled_low_actions[sample_time][revised_index], (-1,)))
                agent.train_low(low_gradbuffer, low_gradient)

            if recommender_trainable:
                _, _, select_user_input, select_num_idx, _, _, _, _, _, _ =  sampling_RL(
                     batched_user_input, batched_num_idx, batched_item_input, batched_label_input, batch_index, agent, env, padding_number, Random=False)
                train_loss,_ = recommender.train(select_user_input, np.reshape(select_num_idx,(-1,1)), batched_item_input,  batched_label_input)
        else:
            train_loss,_ = recommender.train(batched_user_input,batched_num_idx , batched_item_input, batched_label_input)
        avg_loss += train_loss

        # Update parameters
        if agent_trainable:
            agent.update_target_high_network()
            agent.update_target_low_network()
            if recommender_trainable:
                recommender.update_target_network()
        else:
            recommender.assign_target_network()

    return avg_loss / train_batch_num


def get_avg_reward(agent, trainset, testset, recommender, env, padding_number):
    train_user_input, train_num_idx, train_item_input, train_labels, \
        train_batch_num, train_user_idx = (trainset[0], trainset[1], trainset[2], trainset[3], trainset[4], trainset[5])
    avg_reward, total_selected_items, total_revised_instances, total_notrevised_instances, total_deleted_instances, total_keep_instances = 0,0,0,0,0,0
    rewards_list = []
    total_instances = 0
    test_begin = time()
    for batch_index in range(train_batch_num):
        batched_user_input = np.array([u for u in train_user_input[batch_index]])
        batched_item_input = np.reshape(train_item_input[batch_index], (-1, 1))
        batched_label_input = np.reshape(train_labels[batch_index], (-1, 1))
        batched_num_idx = np.reshape(train_num_idx[batch_index], (-1,1))
        batched_user_idx = np.reshape(train_user_idx[batch_index], (-1,1))

        high_action, high_state, select_user_input, select_num_idx, _, _, notrevised_index, revised_index, delete_index, keep_index =  sampling_RL(batched_user_input, batched_num_idx, batched_item_input, batched_label_input, batch_index, agent, env, padding_number, Random=False)
        _, _, reward = env.get_reward(recommender, batch_index, high_action, 
                                      select_user_input, select_num_idx, 
                                      batched_item_input, batched_label_input, 
                                      batched_user_idx, batched_user_input, 
                                      test_flag=False)
        
        avg_reward += np.sum(reward)
        rewards_list.extend(reward)
        total_selected_items += np.sum(select_num_idx)
        total_revised_instances += len(revised_index)
        total_notrevised_instances += len(notrevised_index)
        total_deleted_instances += len(delete_index)
        total_keep_instances += len(keep_index)
        total_instances += batched_user_input.shape[0]
    test_time = time() - test_begin
    avg_reward = np.mean(rewards_list)
    std_reward = np.std(rewards_list)

    return avg_reward, std_reward, total_selected_items, total_revised_instances,total_notrevised_instances, total_deleted_instances,total_keep_instances, test_time



