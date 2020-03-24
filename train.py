import pickle

import torch
import copy
import torch.nn.functional as F


def train (model, data_loader, loss_func, batch, val_cutting_point, max_epochs, max_patience):

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    current_patience = 0
    best_loss = 0
    best_state = dict()

    model.train()

    train_losses = []
    val_losses = []
    for epoch in range(max_epochs):
        
        
        avg_train_loss = 0
        avg_val_loss = 0
        train_batches = 0
        val_batches = 0
        
        #For some reason it takes SIGNIFICANLY less memory if you're using only one dataloader
        #So this is kind of a crutch, but it allows to train faster
        
        for data in data_loader:
            optimizer.zero_grad()
            pred = model(data[0])
            loss = loss_func(pred, data[0])
                
            if (train_batches+val_batches)*batch < val_cutting_point:
                
                loss.backward()
                optimizer.step()
                train_batches+= 1
                avg_train_loss += float(loss)
            else:
                val_batches+= 1
                avg_val_loss += float(loss)    
        
        
        avg_train_loss /= train_batches
        avg_val_loss /= val_batches
        
        
        if  epoch == 0 or avg_val_loss < best_loss:
            best_state = copy.deepcopy(model.state_dict())
            current_patience = 0
            best_loss = avg_val_loss
        else: 
            current_patience += 1

            if current_patience >= max_patience:
                print ("Patience is run out, halting the training")
                break
        print("Epoch: " + str(epoch) + "/" + str(max_epochs) + ", current validation loss: " + str(float(avg_val_loss)))
        train_losses.append(avg_val_loss)
        val_losses.append(avg_val_loss)

    return train_losses, val_losses, best_state