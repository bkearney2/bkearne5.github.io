import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


import numpy as np
import pandas as pd

from neural_hw3 import dlnet

# Data input

data = pd.read_csv('wdbc.data',header=None)
data.iloc[:,1].replace('B', 0,inplace=True)
data.iloc[:,1].replace('M', 1,inplace=True)
data = data.astype(float)
data.head(3)
scaled_data=data
names = data.columns[1:13]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data.iloc[:,1:13])
scaled_data = pd.DataFrame(scaled_data, columns=names)

# Scale data
x=scaled_data.iloc[:,2:13].values.transpose()
y=data.iloc[:,1].values.transpose()
y = np.array([y])

# Create lists for output storage
y_actual = []
predictions = []
accuracy = []
total_accuracy = []
loss_list = []
incorrects = []
incorrect_counter = 0

# leave-one-out cross validation
for i in range(scaled_data.shape[0]):
    
    # Remove one sample from training set
    x_train = np.delete(x, i, axis=1)
    y_train = np.delete(y, i, axis=1)
    
    # Create neural net class
    nn = dlnet(x_train, y_train)
    final_loss = nn.gd(x_train, y_train)
    
    # Make k=1 testing set
    x_test = np.array([np.array(x[:,i])]).transpose()
    y_test = np.array([np.array(y[:,i])]).transpose()
    
    # Prediction, return prediction and accuracy
    pred_prob,pred_comp = nn.pred(x_test, y_test)
    
    # Add to lists
    predictions.append(pred_prob[0][0])
    y_actual.append(y_test[0][0])
    accuracy.append(pred_comp[0][0])
    
    # Check for incorrect predictions
    if pred_comp[0][0] != y_test[0][0]:
        incorrect_counter = incorrect_counter + 1
        incorrects.append(i)
    total_accuracy.append(100*((i+1)-incorrect_counter)/(i+1))
    loss_list.append(final_loss[0][0])
#    print("accuracy: ", accuracy)
#    print("predictions: ",predictions)
#    print("y actual: ",y_actual)
    
# Plot and output
    plt.plot(total_accuracy, 'o', color='black')
    axes = plt.gca()
    axes.set_ylim([0,100])
    plt.title("Neural net accuracy through leave-1-out cycle # "+str(i))
    plt.ylabel('% Accuracy')
    plt.xlabel('leave-one-out cycles')
    plt.show()
    print("Cycle number: ", i)
    print("average loss: ", sum(loss_list)/len(loss_list))
    print("Total accuracy: ",total_accuracy[i])
    print("Incorrect predictions, sample #: ", incorrects)
