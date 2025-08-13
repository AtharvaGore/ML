import numpy as np
final=[]
final2=[]
with open('diabetes.csv','r') as f:
    f.readline()
    
    for i in range(700):
        row = f.readline().split(',')
        final2.append([float(row[-1])])
        row = row[0:len(row)-1]
        temp=[]
        for j in row:
            temp+=[float(j)]
        final.append(temp)
X=np.array(final)
Y=np.array(final2)
theta = np.array([[1],[1],[1],[1],[1],[1],[1],[1],[1]])
X_mean = np.mean(X,axis=0)
X_std = np.std(X,axis=0)
X = (X-X_mean)/X_std
X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

lr=0.01
for i in range(10000):
    XTheta = np.dot(X,theta)
    prediction = 1/(1+np.exp(-XTheta))
    yminusprediction = np.subtract(Y,prediction)
    derivative = np.dot(np.transpose(X),yminusprediction)
    theta = theta + (lr*derivative)

with open('diabetes.csv','r') as f:
    count=0
    for i in range(700):
        f.readline()
    for i in range(69):
        row = f.readline().split(',')
        real = int(row[-1])
        row = row[0:len(row)-1]
        temp=[]
        for j in row:
            temp+=[float(j)]
        row = np.array(temp)
        row = (row-X_mean)/X_std
        row = np.concatenate([[1],row])
        predicted = 1/(1+np.exp(-np.dot(row,theta)))
        if predicted[0]>0.5 and real==1:
            count+=1
        elif predicted[0]<0.5 and real==0:
            count+=1
    print(count/69)
