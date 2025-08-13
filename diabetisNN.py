import numpy as np
import neuron
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
theta3,theta4,theta5,theta6 = 1.0,-1.0,0.0,0.0
bias3,bias4 = 0,0

X=np.array(final)
Y=np.array(final2)
X_mean = np.mean(X,axis=0)
X_std = np.std(X,axis=0)
X = (X-X_mean)/X_std
node1 = neuron.Neuron([[1],[1],[1],[1],[1],[1],[1],[1]],[[1],[1]])
node2 = neuron.Neuron([[1],[1],[1],[1],[1],[1],[1],[1]],[[1],[1]])

node1input = np.dot(X,node1.inputs) + node1.bias
node2input = np.dot(X,node2.inputs) + node2.bias

Xtheta1 = node1input
Xtheta2 = node2input

node1input = np.log1p(np.exp(node1input))
node2input = np.log1p(np.exp(node2input))

rawY = node1input*theta3 + node2input*theta4 + bias3
rawN = node1input*theta5 + node2input*theta6 + bias4

eY = np.exp(rawY)
eN = np.exp(rawN)

pY = eY/(eY + eN)
pN = eN/(eY + eN)

temp = (1/(1+np.exp(Xtheta1)))*np.exp(Xtheta1)
temp2 = []
for i in range(700):
    if Y[i,0] == 1:
        temp2.append([pN[i,0]*(theta5-theta3)])
    else:
        temp2.append([pY[i,0]*(theta3-theta5)])
temp2 = np.array(temp2)
temp2 = temp*temp2
temp2 = X*temp2



summation=0
for j in range(700):

    node1input = np.dot(X,node1.inputs) + node1.bias
    node2input = np.dot(X,node2.inputs) + node2.bias
    Xtheta1 = node1input
    Xtheta2 = node2input

    node1input = np.log1p(np.exp(node1input))
    node2input = np.log1p(np.exp(node2input))

    rawY = node1input*theta3 + node2input*theta4 + bias3
    rawN = node1input*theta5 + node2input*theta6 + bias4

    eY = np.exp(rawY)
    eN = np.exp(rawN)

    pY = eY/(eY + eN)
    pN = eN/(eY + eN)

    summation=0
    summation = np.mean(pY-Y)
    bias3 = bias3 - 0.01*summation
    summation = np.mean(pN-(1-Y))
    bias4 = bias4 - 0.01*summation
    summation = np.mean((pY-Y)*node1input)
    theta3 = theta3 - 0.01*summation
    summation = np.mean((pY-Y)*node2input)
    theta5 = theta5 - 0.01*summation
    summation = np.mean((pN-(1-Y))*node1input)
    theta4 = theta4 - 0.01*summation
    summation = np.mean((pN-(1-Y))*node2input)
    theta6 = theta6 - 0.01*summation

    temp = (1/(1+np.exp(Xtheta1)))*np.exp(Xtheta1)
    temp2 = []
    for i in range(700):
        if Y[i,0] == 1:
            temp2.append([pN[i,0]*(theta5-theta3)])
        else:
            temp2.append([pY[i,0]*(theta3-theta5)])
    temp2 = np.array(temp2)
    temp2 = temp*temp2
    temp2 = X*temp2


    for i in range(8):
        summation = temp2[:, i].sum()
        node1.inputs[i,0] = node1.inputs[i,0] - 0.05*summation
        # print(node1.inputs[i,0])
    
# print(bias3)
# print(bias4)
# print(theta3)
# print(theta4)
#print(node1.inputs)

# with open('diabetes.csv','r') as f:
#     for i in range(705):
#         f.readline()

#         row = f.readline().split(',')
#         real = int(row[-1])
#         row = row[0:len(row)-1]
#         temp=[]
#         for j in row:
#             temp+=[float(j)]
#         row = np.array(temp)
#         row = (row-X_mean)/X_std
#         row = np.concatenate([[1],row])



