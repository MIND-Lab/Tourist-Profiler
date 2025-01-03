# Tourist Profiler

This work was supported by the research project RASTA: Realtá Aumentata e Story-Telling Automatizzato 
per la valorizzazione di Beni Culturali ed Itinerari; funded by the Italian Ministry of University 
and Research (MUR) under the PON Project ARS01\_00540.

## Dataset

The `dataset` folder contains the methods used for creating the dataset utilized
in training and testing the models. Additionally, it includes a notebook for
analyzing the dataset, allowing for a detailed exploration of its characteristics
and insights.

## DQN

The `dqn` folder contains the implementation of the DQN (Deep Q-Network) model.
It includes the following files:

- **`Agent_DQN.py`**: Contains the implementation of the DQN agent.
- **`Environment_DQN.py`**: Contains the implementation of the environment used
for the model.
- **`Dataset_DQN.py`**: Implements methods for managing the dataset.
- **`Settings_DQN.py`**: Includes the configuration settings for the DQN model.

To run the experiment, use the notebook **`dqn.ipynb`**, which provides a complete
process for training and testing the DQN model.

### Visualization

Below is an image representation of the DQN model:  
![DQN](./img/RASTA%20-%20DQN.png)

## HRL
The `hrl` folder contains the implementation of the HRL (Hierarchical Reinforcement learning) model, combined with a Matrix Factorization model designed to infer the missing values in the Profile. It includes the following files:

- **`Agent.py`**: Contains the implementation of the HRL agent.
- **`Environment.py`**: Contains the implementation of the environment used
for the model.
- **`Recommender.py`**: Contains the implementation of the NAIS recommender model used combined with HRL agent.
- **`DataMovielens.py`**: Implements methods for managing the dataset.
- **`Settings.py`**: Includes the configuration settings for the HRL model
- **`Evaluation.py`**: Implements the calculation of some evaluation metrics for the model
- **`Utils.py`**: Implements some functions for training and evaluating the model

### Visualization

Below is an image representation of the HRL model:  
![HRL](./img/RASTA%20-%20HRL.png)

## Profile Models

The `profile_models` folder contains the implementation of the baseline profile
models used for comparison against our model. This folder includes all the necessary
files to run the experiments, along with a notebook that provides a complete
workflow for training and testing these models.
