import torch
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error


def rmse_with_mask(unmasked_input_data, masked_input_data, pred_ratings, true_ratings, masked_ref):
    rmse_total = []
    rmse_explicit = []
    rmse_mask = []
    
    # Here we compute the total RMSE for all the non-zero values in the true
    # ratings.
    for i in range(len(pred_ratings)):
        # For the total RMSE we consider only the values of the true (not masked) 
        # ratings that are not zero. We compute a mask for these values.
        mask = true_ratings[i] != 0
        # We filter the predicted and true ratings with the mask.
        filtered_pred_ratings = pred_ratings[i][mask]
        filtered_true_ratings = true_ratings[i][mask]
        filtered_pred_ratings = filtered_pred_ratings.numpy()
        filtered_true_ratings = filtered_true_ratings.numpy()
        # We compute the RMSE only if we still have values after the filtering.
        # Otherwise we return an error of 0.
        if (len(filtered_pred_ratings) > 0):    
            rmse = sqrt(mean_squared_error(filtered_pred_ratings, filtered_true_ratings))
            rmse_total.append(rmse)
        else:
            rmse_total.append(0)
        
    # Here we compute the RMSE for all the non-zero values in the masked data.
    for i in range(len(pred_ratings)):
        # We compute a mask for the non-zero values in the unmasked input data.
        mask = unmasked_input_data[i] != 0
        # We filter the predicted and true ratings with the mask.
        filtered_pred_ratings = pred_ratings[i][mask]
        filtered_true_ratings = true_ratings[i][mask]
        filtered_pred_ratings = filtered_pred_ratings.numpy()
        filtered_true_ratings = filtered_true_ratings.numpy()
        # We compute the RMSE only if we still have values after the filtering.
        if (len(filtered_pred_ratings) > 0):    
            rmse = sqrt(mean_squared_error(filtered_pred_ratings, filtered_true_ratings))
            rmse_explicit.append(rmse)

    # Here we compute the RMSE only for the masked zero values in the masked data.
    for i in range(len(pred_ratings)):
        # We compute the mask for the positions we want to consider. For this
        # we use the masked reference that specifies the positions to use.
        mask = masked_ref[i]
        mask = mask.bool()
        # We filter the predicted and true ratings with the mask.
        filtered_pred_ratings = pred_ratings[i][mask]
        filtered_true_ratings = true_ratings[i][mask]
        filtered_pred_ratings = filtered_pred_ratings.numpy()
        filtered_true_ratings = filtered_true_ratings.numpy()
        # We compute the RMSE only if we still have values after the filtering.
        if (len(filtered_pred_ratings) > 0):    
            rmse = sqrt(mean_squared_error(filtered_pred_ratings, filtered_true_ratings))
            rmse_mask.append(rmse)
            
    return torch.tensor(rmse_total), torch.tensor(rmse_explicit), torch.tensor(rmse_mask)


def mae_with_mask(unmasked_input_data, masked_input_data, pred_ratings, true_ratings, masked_ref):
    mae_total = []
    mae_explicit = []
    mae_mask = []
    
    # Here we compute the total MAE for all the non-zero values in the true
    # ratings.
    for i in range(len(pred_ratings)):
        # For the total MAE we consider only the values of the true (not masked) 
        # ratings that are not zero. We compute a mask for these values.
        mask = true_ratings[i] != 0
        # We filter the predicted and true ratings with the mask.
        filtered_pred_ratings = pred_ratings[i][mask]
        filtered_true_ratings = true_ratings[i][mask]
        filtered_pred_ratings = filtered_pred_ratings.numpy()
        filtered_true_ratings = filtered_true_ratings.numpy()
        # We compute the mean absolute error.
        mae = mean_absolute_error(filtered_pred_ratings, filtered_true_ratings)
        mae_total.append(mae)
        
    # Here we compute the MAE for all the non-zero values in the masked data.
    for i in range(len(pred_ratings)):
        # We compute a mask for the non-zero values in the unmasked input data.
        mask = unmasked_input_data[i] != 0
        # We filter the predicted and true ratings with the mask.
        filtered_pred_ratings = pred_ratings[i][mask]
        filtered_true_ratings = true_ratings[i][mask]
        filtered_pred_ratings = filtered_pred_ratings.numpy()
        filtered_true_ratings = filtered_true_ratings.numpy()
        # We compute the MAE only if we still have values after the filtering.
        if (len(filtered_pred_ratings) > 0):    
            mae = mean_absolute_error(filtered_pred_ratings, filtered_true_ratings)
            mae_explicit.append(mae)

    # Here we compute the MAE only for the masked zero values in the masked data.
    for i in range(len(pred_ratings)):
        # We compute the mask for the positions we want to consider. For this
        # we use the masked reference that specifies the positions to use.
        mask = masked_ref[i]
        mask = mask.bool()
        # We filter the predicted and true ratings with the mask.
        filtered_pred_ratings = pred_ratings[i][mask]
        filtered_true_ratings = true_ratings[i][mask]
        filtered_pred_ratings = filtered_pred_ratings.numpy()
        filtered_true_ratings = filtered_true_ratings.numpy()
        # We compute the RMSE only if we still have values after the filtering.
        if (len(filtered_pred_ratings) > 0): 
            mae = mean_absolute_error(filtered_pred_ratings, filtered_true_ratings)
            mae_mask.append(mae)
    
    return torch.tensor(mae_total), torch.tensor(mae_explicit), torch.tensor(mae_mask)
