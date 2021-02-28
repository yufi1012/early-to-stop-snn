import numpy as np
import torch
from model import FirstToSpike
from sklearn.metrics import accuracy_score
from dvsgestures_dataloaders import *
import time
import os

# --- methods ---
def exponentialDecay(N):
    tau = 1 
    tmax = 4 
    t = np.linspace(0, tmax, N)
    y = np.exp(-t/tau)
    y = torch.FloatTensor(y)
    return y/10.

# --- hyperparameters ---
N_CLASSES = 11


def run(LEARNING_RATE, SEQ_LENGTH, N_EPOCHS, LAMBDA):
    
    exponentials = exponentialDecay(N_EPOCHS)
    
    # ---dvs dataset---
    train_loader, test_loader = create_dataloader(root='DVS/data/dvsgesture/dvs_gestures_build19.hdf5', batch_size=64, ds=1, num_workers=0)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- initialize the model and the optimizer ---
    model = FirstToSpike(nhid=N_CLASSES, nclasses=N_CLASSES, lam=LAMBDA)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # --- training ---
    training_loss = []
    for epoch in range(N_EPOCHS):

        model._REWARDS = 0
        model._r_sums = np.zeros(SEQ_LENGTH).reshape(1, -1)
        model._r_counts = np.zeros(SEQ_LENGTH).reshape(1, -1)
        model._epsilon = exponentials[epoch]#!
        loss_sum = 0
        start_time = time.time()
        
        training_predictions = []
        training_locations = []
        training_labels = []
        for i, (X, y) in enumerate(train_loader):
            idx = (y.sum(1) > 0).nonzero(as_tuple=False)
            y = idx[:,-1]
            
            # --- Forward pass ---
            rate, halting_points, _ = model(X, epoch)
            _, predictions = torch.max(rate[:,-1,:], dim=1)
            training_locations.append(halting_points)
            training_predictions.append(predictions)
            training_labels.append(y)

            # --- Compute gradients and update weights ---
            optimizer.zero_grad()
            loss = model.computeLoss(rate, y)
            loss.backward()
            loss_sum += loss.item()
            optimizer.step()

            if i+1 == 19:
                training_predictions = torch.cat(training_predictions,dim=0).numpy().reshape(-1, 1)
                training_labels = torch.cat(training_labels,dim=0).numpy().reshape(-1, 1)
                training_locations = torch.stack(training_locations).numpy().reshape(-1, 1)

                print ('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}, Acc: {}, Mean proportion used: {}%, Time: {}'.format(epoch+1, N_EPOCHS, i+1, len(train_loader), loss.item(), np.round(accuracy_score(training_labels, training_predictions), 3), np.round(100.*np.mean(training_locations), 3), time.time()-start_time))
        
        
        training_loss.append(np.round(loss_sum/len(train_loader), 3))
        scheduler.step()


    # --- testing ---
    testing_predictions = []
    testing_labels = []
    testing_locations = []
    loss_sum = 0
    for i, (X, y) in enumerate(test_loader):
        idx = (y.sum(1) > 0).nonzero(as_tuple=False)
        y = idx[:,-1]
        
        rate, halting_points = model(X, test=True)
        _, predictions = torch.max(rate, dim=1)
        testing_locations.append(halting_points)
        testing_predictions.append(predictions)
        testing_labels.append(y)

    testing_predictions = torch.cat(testing_predictions,dim=0).numpy().reshape(-1, 1)
    testing_labels = torch.cat(testing_labels,dim=0).numpy().reshape(-1, 1)
    testing_locations = torch.stack(testing_locations).numpy().reshape(-1, 1)
    print("Accuracy: {}".format(np.round(accuracy_score(testing_labels, testing_predictions), 3)))
    print("Mean proportion used: {}%".format(np.round(100.*np.mean(testing_locations), 3)))


