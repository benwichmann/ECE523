from functions import *


path = "c:/datasets/acute_inflamation.csv"
dataset = pd.read_csv(path, header=None)  

#split dataset into testing and traing
xtest,xtrain,ytest,ytrain = split_dataset(dataset,.6)

#train w parameters 
wtr,k = sgd(xtrain,ytrain,1000,0.02)

#test w parameters against xtest feature data 
yhat = logistic_function(xtest,wtr)
for i in range(len(yhat)):
    if yhat[i] >=0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0


print("w parameters: \n", wtr)
print("k occurences using Cross Entropy: ",k)
dif_y = yhat[:,0] - ytest[:]
perc_correct = (1.0 - (float(np.count_nonzero(dif_y))/len(dif_y)))*100

print("Precent Accuracy with test data: ", perc_correct)

print("\nModify ytest[2] value to confirm SGD working...")
if ytest[2]==1:
    ytest[2]=0
else:
    ytest[2]=1

dif_y = yhat[:,0] - ytest[:]
perc_correct = (1.0 - (float(np.count_nonzero(dif_y))/len(dif_y)))*100

print("Precent Accuracy with altered test data: ", perc_correct)