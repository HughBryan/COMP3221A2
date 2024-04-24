import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

import os
import json
import copy
import random
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import time

def get_data(client_id):

    np.random.seed(client_id)
    n_samples = np.random.randint(800, 1000)
    X_train = [[1,2,3,4,5,6,7],[7,6,5,4,3,2,1]]
    y_train = [10,9,8,7,6,5,4]


    X_test = [[1,2,3,4,5,6,7],[7,6,5,4,3,2,1]]
    y_test = [10,9,8,7,6,5,4]



    # Cast variables to torch type
    X_train = torch.Tensor(X_train).view(-1,1).type(torch.float32)
    y_train = torch.Tensor(y_train).type(torch.float32)
    X_test = torch.Tensor(X_test).view(-1,1).type(torch.float32)
    y_test = torch.Tensor(y_test).type(torch.float32)
    # Training Size, Test Size
    train_samples, test_samples = len(y_train), len(y_test)

    print(X_train)

    return X_train, y_train, X_test, y_test, train_samples, test_samples

class LinearRegressionModel(nn.Module):
    def __init__(self, input_size = 1):
        super(LinearRegressionModel, self).__init__()
        # Create a linear transformation to the incoming data
        self.linear = nn.Linear(input_size, 1)

    # Define how the model is going to be run, from input to output
    def forward(self, x):
        # Apply linear transformation
        output = self.linear(x)
        return output.reshape(-1)
    

class UserAVG():
    def __init__(self, client_id, model, learning_rate, batch_size):

        self.X_train, self.y_train, self.X_test, self.y_test, self.train_samples, self.test_samples = get_data(client_id)

        self.train_data = [(x, y) for x, y in zip(self.X_train, self.y_train)]
        self.test_data = [(x, y) for x, y in zip(self.X_test, self.y_test)]

        # Define dataloader for iterable sample over a dataset
        self.trainloader = DataLoader(self.train_data, batch_size = batch_size)
        self.testloader = DataLoader(self.test_data, batch_size = self.test_samples)

        # Define the Mean Square Error Loss
        self.loss = nn.MSELoss()

        self.model = copy.deepcopy(model)

        self.id = client_id

        # Define the Gradient Descent optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

    def set_parameters(self, model):
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()

    def train(self, epochs):
        LOSS = 0
        self.model.train()
        for epoch in range(1, epochs + 1):
            self.model.train()
            for batch_idx, (X, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
        return loss.data

    def test(self):
        self.model.eval()
        mse = 0
        for x, y in self.testloader:
            y_pred = self.model(x)
            # Calculate evaluation metrics
            mse += self.loss(y_pred, y)
            print(str(self.id) + ", MSE of client ",self.id, " is: ", mse)
        return mse
    


def send_parameters(server_model, users):
    for user in users:
        user.set_parameters(server_model)

def aggregate_parameters(server_model, users, total_train_samples):
    # Clear global model before aggregation
    for param in server_model.parameters():
        param.data = torch.zeros_like(param.data)

    for user in users:
        for server_param, user_param in zip(server_model.parameters(), user.model.parameters()):
            server_param.data = server_param.data + user_param.data.clone() * user.train_samples / total_train_samples
    return server_model

def evaluate(users):
    total_mse = 0
    for user in users:
        total_mse += user.test()
    return total_mse/len(users)


# Init parameters
num_user = 5
users = []
server_model = LinearRegressionModel()
batch_size = 64
learning_rate = 0.01
num_glob_iters = 1 # No. of global rounds




# TODO:  Create a federate learning network with 5 clients and append it to users list.
total_train_samples = 0
for i in range(1,num_user+1):
    user = UserAVG(i, server_model, learning_rate, batch_size)
    users.append(user)
    total_train_samples += user.train_samples
    send_parameters(server_model, users)


# Runing FedAvg
train_mse = []
test_mse = []

for glob_iter in range(num_glob_iters):

    # TODO: Broadcast global model to all clients
    send_parameters(server_model,users)

    # Evaluate the global model across all clients
    avg_mse = evaluate(users)
    test_mse.append(avg_mse.item())
    print("Global Round:", glob_iter + 1, "Average MSE across all clients : ", avg_mse)

    # Each client keeps training process to  obtain new local model from the global model
    avgLoss = 0
    for user in users:
        # Each user trains the local model for 2 epochs
        avgLoss += user.train(2)
    # Above process training all clients and all client paricipate to server, how can we just select subset of user for aggregation
    train_mse.append(avgLoss)

    # TODO:  Aggregate all clients model to obtain new global model
    aggregate_parameters(server_model, users, total_train_samples)

weights = server_model.linear.weight.data
bias = server_model.linear.bias.data

print("Weights:", weights)
print("Bias:", bias)