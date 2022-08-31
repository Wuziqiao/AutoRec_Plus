# AutoRec++
This is the implementions for the paper work of "AutoRec++: Incorporating Debias Methods Into Autoencoder-Based Recommender System"

## Brief Introduction
In this paper, we aim at comprehensively addressing the various biases existed in user behavior data for DNN-based RSs. To this end, we incorporate various combinations of preprocessing bias (PB) and training bias (TB) into the Autoencoder to propose our AutoRec++ model. By conducting extensive experiments on five benchmark datasets, we demonstrate that: 1) the Autoencoder’s prediction accuracy and computational efficiency can be significantly boosted by incorporating the optimal combination of PB and TB into it without structural change, and 2) our AutoRec++ achieves significantly better prediction accuracy than both DNN-based and non-DNN-based state-of-the-art models. 

## Enviroment Requirement
- python 3.7
- numpy
- tensorflow (below 2.0 otherwise need to call disable_v2_behavior())

## Dataset
In the example, we offer a "douban" dataset, and the corresponding hyperparameters for "douban" are offered below.

## Parameters setting
- hidden_neuron = 500
- train_epoch = 500
- lambda_value_1 = 1
- lambda_value_2 = 1
- batch_size = 1500
- base_lr = 0.001