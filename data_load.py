import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import numpy as np
import time

def load_data():
    
    #print(torch.cuda.is_available())
    #device_id = 0 if torch.cuda.is_available() else 'cpu' # Equivalent to device_id = 'cuda:0'

    X_test = np.load("X_test.npy") 

    y_test = np.load("y_test.npy")-769  #the-769 is necessary here because the y_test and y_train_valid are whats storing the class labels. 
                                        #From the project document (769, 770, 771, 773) are classes (1,2,3,4) Respectively.
                                        #Subtracting -769 helps in many ways: helps in the architecture utilizing 4 classes and not confusing with like 700
                                        #Also helps to train and model. 

    person_train_valid = np.load("person_train_valid.npy")
    X_train_valid = np.load("X_train_valid.npy")
    y_train_valid = np.load("y_train_valid.npy")-769
    person_test = np.load("person_test.npy")
    return X_test, y_test, person_train_valid,X_train_valid,y_train_valid,person_test

def data_loader_setup(subject, X_test, y_test, batch_size,person_train_valid,X_train_valid,y_train_valid,person_test,verbose=False,):
    
    print('Subject:',subject)
    device_id = 'cpu'
    device = torch.device(device_id)
    classes = np.unique(y_test)

    #So this block is setting up the model to just go through one subject
    #We have to also do this for the whole dataset so I believe adding another block would be necessary
    #The challenge will be how we will run the code so that it runs through just one subject first and train, then have it run towards all subjects
    #We can try to do it automatically or we can just opt to do it manually. 


    #Extracting First subject data from train/valid dataset
    X_train_valid_sub1 = X_train_valid[person_train_valid.squeeze()==subject-1] 
    y_train_valid_sub1 = y_train_valid[person_train_valid.squeeze()==subject-1]

    #Extracting First subject data from test dataset
    X_test_sub1 = torch.FloatTensor(X_test[person_test.squeeze()==subject-1]).to(device)
    y_test_sub1 = torch.LongTensor(y_test[person_test.squeeze()==subject-1]).to(device)

    #Setting up to split train/valid dataset to train dataset and validation dataset with shuffled indices
    num_sub = X_train_valid_sub1.shape[0]
    p = np.random.permutation(num_sub)
    num_train = int(0.8*num_sub)

    #loading each separate dataset as a tensor
    Xtrain = torch.FloatTensor(X_train_valid_sub1[p[0:num_train]]).to(device)
    ytrain = torch.LongTensor(y_train_valid_sub1[p[0:num_train]]).to(device)

    Xval = torch.FloatTensor(X_train_valid_sub1[p[num_train:]]).to(device)
    yval = torch.LongTensor(y_train_valid_sub1[p[num_train:]]).to(device)

    Xtest = torch.FloatTensor(X_test_sub1).to(device)
    ytest = torch.LongTensor(y_test_sub1).to(device)

    #joining respective dataset together
    train_data = data.TensorDataset(Xtrain, ytrain)
    val_data = data.TensorDataset(Xval, yval)
    test_data = data.TensorDataset(Xtest,ytest)

    #Creating dataloader from the above dataset
    train_dataloader = data.DataLoader(train_data, batch_size=batch_size)
    val_dataloader = data.DataLoader(val_data, batch_size=batch_size)
    test_dataloader = data.DataLoader(test_data, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader
    
def train_eval(model,train_dataloader, val_dataloader,crit,optimizer,num_epoch):
    #These variables are from the graduate notebook that stores the loss throughout the training and stores them for later plotting. 
    loss_hist = []
    val_loss_hist = []
    acc_hist = []
    val_acc_hist = []

    for epoch in range(num_epoch):
        model.train()#Before training we place the model in training mode. 
        for data, label in train_dataloader: #separating dataloader to data and the label (or x_traina and y_train)

            #Set optimizer gradient
            optimizer.zero_grad() 

            #Have the input go through model (CNN)
            out = model(data)

            #Calculate loss (this igoing through crit = nn.CrossEntropyLoss())
            loss = crit(out, label)

            #Backpropagaation
            loss.backward()

            #Gradient Step (optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate, weight_decay=wd))
            optimizer.step()

            #Storing the loss to an array
            loss_hist.append(loss.item())

        model.eval() #We place the model in evaluation mode
        #Some variable initialization
        train_loss = 0 
        correct = 0
        ns = 0
        #First we evaluation against the train dataset (what we just trained it on)
        for data, label in train_dataloader: #Obtaining data from train dataloader
            #Input through model
            out = model(data)

            #Calculate loss
            loss = crit(out,label)

            #Add up the losses through each epoch
            train_loss += loss.item()

            #Get max log-prob and obtain label. This will be our predicted labels
            pred = out.data.max(1,keepdim=True)[1]

            #Calculating correct labels
            correct += pred.eq(label.data.view_as(pred)).detach().cpu().numpy().sum()

            ns += len(label)

            #This code is help make the graphs from the graduate notebook.
            #I have it cut out for now as I think we can utilize the code above to make our own graphs.  
            #nc += (out.max(1)[1] == label).detach().cpu().numpy().sum()
        #acc_hist.append(nc/ns)

        #Calculate avg train loss
        train_loss /= ns
        #Calculate percentage training accuracy
        perc_acc_train = 100*correct/ns

        #Code to print out calculate loss and accuracy after every epoch. 
        print('train_loss:',train_loss, ',num_correct:',correct, ',total:', ns, ',percent_accuracy:', perc_acc_train)

        #Follow the same code structure to obtain the validation accuracy 
        val_loss = 0
        correct = 0
        ns = 0
        for data, label in val_dataloader:
                out = model(data)
                loss = crit(out, label)

                val_loss += loss.item()
                val_loss_hist.append(loss.item())

                pred = out.data.max(1,keepdim=True)[1]

                correct += pred.eq(label.data.view_as(pred)).detach().cpu().numpy().sum()

                ns += len(label)
        val_loss /= ns
        perc_acc_val = 100*correct/ns
        print('val_loss:',val_loss, ',num_correct:',correct, ',total:', ns, ',percent_accuracy:', perc_acc_val)
        
        return
    
def test_eval(model, test_dataloader, crit):
    #This block just evaluates our trained model towards the test dataset
    classes = [1, 2, 3, 4]

    test_loss = 0
    correct = 0
    ns = 0

    correct_class = np.zeros(len(classes))
    num_class = np.zeros(len(classes))

    model.eval()
    for data, label in test_dataloader:
        out = model(data)
        loss = crit(out,label)

        test_loss += loss.item()
        pred = out.data.max(1,keepdim=True)[1]

        cor = pred.eq(label.data.view_as(pred)).long().detach().cpu().numpy()
        correct += pred.eq(label.data.view_as(pred)).detach().cpu().numpy().sum()

        ns += len(label)
            #nc += (out.max(1)[1] == label).detach().cpu().numpy().sum()
        #acc_hist.append(nc/ns)

        test_loss /= ns
        perc_acc_test = 100*correct/ns

        for i in range(ns):
            m = label.data[i]
            correct_class[m] += cor[i].item()
            num_class[m] += 1

    #What I printed out is the loss, # of correct labels, #of total labels, Percentage test accuracy
    print('test_loss:', test_loss, ',num_correct:',correct, ',total:', ns, ',percent_accuracy:', perc_acc_test)
    print('correct_class',correct_class)#This shows the number of corrected labels per class
    print('num_classes',num_class)#This shows the total number of each class in that data

    #The amount is small because the data we are looking it is only on one subject.
    return
   

def data_loader_rnn(subject, X_test, y_test, batch_size,person_train_valid,X_train_valid,y_train_valid,person_test,verbose=False,):
    
    print('Subject:',subject)
    device_id = 'cpu'
    device = torch.device(device_id)

    Xtest = torch.FloatTensor(X_test.transpose(0,2,1)).to(device)
    ytest = torch.LongTensor(y_test).to(device)
    test_data = data.TensorDataset(Xtest,ytest)


    #Extracting First subject data from train/valid dataset
    num_tv = X_train_valid.shape[0]
    p = np.random.permutation(num_tv)
    num_train = int(0.8*num_tv)

    Xtrain = torch.FloatTensor(X_train_valid[p[0:num_train]].transpose(0,2,1)).to(device)
    ytrain = torch.LongTensor(y_train_valid[p[0:num_train]]).to(device)
    train_data = data.TensorDataset(Xtrain, ytrain)

    Xval = torch.FloatTensor(X_train_valid[p[num_train:]].transpose(0,2,1)).to(device)
    yval = torch.LongTensor(y_train_valid[p[num_train:]]).to(device)
    val_data = data.TensorDataset(Xval, yval)

    batch_size = 128
    train_dataloader = data.DataLoader(train_data, batch_size=batch_size)
    val_dataloader = data.DataLoader(val_data, batch_size=batch_size)

    test_dataloader = data.DataLoader(test_data, batch_size=batch_size)


    return train_dataloader, val_dataloader, test_dataloader

def train_eval_rnn(model,train_dataloader, batch_size,val_dataloader,crit,optimizer,num_epoch):
    start = time.time()

    num_train = len(train_dataloader.sampler)
    iterations_per_epoch = max(num_train //batch_size, 1)
    num_iterations = num_epoch * iterations_per_epoch


    epoch_loss = np.zeros(num_epoch)
    avg_train_loss = []
    avg_val_loss = []
    perc_acc_train =[]
    perc_acc_val =[]

    for epoch in range(num_epoch):
        model.train()
        for data, label in train_dataloader:
            optimizer.zero_grad()
            out = model(data)
            loss = crit(out, label)
            loss.backward()
            optimizer.step()
            #loss_hist.append(loss.item())
            
        model.eval()
        train_loss = 0
        correct = 0
        ns = 0
        for data, label in train_dataloader:
            out = model(data)
            loss = crit(out,label)
            
            train_loss += loss.item()*data.size(0)
            
            pred = out.data.max(1,keepdim=True)[1]
            
            correct += pred.eq(label.data.view_as(pred)).detach().cpu().numpy().sum()
            
            ns += len(label)
            #nc += (out.max(1)[1] == label).detach().cpu().numpy().sum()
        #acc_hist.append(nc/ns)
        
        avg_train_loss.append(train_loss/len(train_dataloader.sampler))
        perc_acc_train.append(100*correct/ns)
        print('epoch:',epoch, ',avg_train_loss:',avg_train_loss[epoch], ',correct',correct, ',ns',ns, ',percent_accuracy:',perc_acc_train[epoch])
        
        val_loss = 0
        correct = 0
        ns = 0
        for data, label in val_dataloader:
            out = model(data)
            loss = crit(out, label)
                
            val_loss += loss.item()*data.size(0)
            #val_loss_hist.append(loss.item())
                
            pred = out.data.max(1,keepdim=True)[1]
                
            correct += pred.eq(label.data.view_as(pred)).detach().cpu().numpy().sum()
                
            ns += len(label)
        avg_val_loss.append(val_loss/len(val_dataloader.sampler))
        perc_acc_val.append(100*correct/ns)
        print('epoch:',epoch, ',avg_val_loss:',avg_val_loss[epoch], ',correct',correct, ',ns',ns, ',percent_accuracy:',perc_acc_val[epoch])
                #nc += (out.max(1)[1] == label).detach().cpu().numpy().sum()
        #val_acc_hist.append(nc/ns)    
        
        
        best_val = 10000
        if val_loss <= best_val:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),}
            torch.save(checkpoint, 'CNN_model.pt')
            best_val = val_loss

    end = time.time()      
    total_time = end - start 

    print(total_time)