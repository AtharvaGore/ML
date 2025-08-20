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
node1 = neuron.Neuron([[1],[1],[1],[1],[1],[1],[1],[1]],[1,-1])
node2 = neuron.Neuron([[1],[1],[1],[1],[1],[1],[1],[1]],[0,0])

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
for j in range(10000):

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
    bias3 = bias3 - 0.05*summation
    summation = np.mean(pN-(1-Y))
    bias4 = bias4 - 0.05*summation
    summation = np.mean((pY-Y)*node1input)
    theta3 = theta3 - 0.05*summation
    summation = np.mean((pY-Y)*node2input)
    theta5 = theta5 - 0.05*summation
    summation = np.mean((pN-(1-Y))*node1input)
    theta4 = theta4 - 0.05*summation
    summation = np.mean((pN-(1-Y))*node2input)
    theta6 = theta6 - 0.05*summation

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

    temp = (1/(1+np.exp(Xtheta2)))*np.exp(Xtheta2)
    temp2 = []
    for i in range(700):
        if Y[i,0] == 1:
            temp2.append([pN[i,0]*(theta6-theta4)])
        else:
            temp2.append([pY[i,0]*(theta4-theta6)])
    temp2 = np.array(temp2)
    temp2 = temp*temp2
    temp2 = X*temp2


    for i in range(8):
        summation = temp2[:, i].sum()
        node2.inputs[i,0] = node2.inputs[i,0] - 0.05*summation
    
# print(bias3)
# print(bias4)
# print(theta3)
# print(theta4)
#print(node1.inputs)

with open('diabetes.csv','r') as f:
    for i in range(709):
        f.readline()
    node3 = neuron.Neuron([[0.3], [-0.2], [0.1], [0.05], [-0.25], [0.15], [0.2], [-0.1]],[0.5, -0.5])
    node4 = neuron.Neuron([[-0.15], [0.25], [0.2], [-0.05], [0.3], [-0.2], [0.1], [0.05]],[-0.4, 0.4])
    node5 = neuron.Neuron([[0.321],[-0.142],[0.567],[-0.654],[0.123],[-0.298],[0.432],[-0.075]],[-0.298, 0.411])
    onehot = np.eye(2)[Y.ravel().astype(int)]
    onehot = onehot[:, ::-1]
    print(onehot)
    obj=neuron.NeuralNetwork(X,Y,2,onehot,(node5,))
    obj.train()

    count=0
    for i in range(50):
        row = f.readline().split(',')
        answer = float(row[-1])
        row = row[0:len(row)-1]
        #print(row)
        temp=[]
        for j in row:
            temp+=[float(j)]
        row = np.array(temp)
        row = (row-X_mean)/X_std

        prediction = obj.forwardPass(row)[0]
        if prediction[0]>prediction[1]:
            if answer == 1:
                count+=1
        elif prediction[1]>prediction[0]:
            if answer == 0:
                count+=1
    print(count/50)

    # node1input = np.dot(row,node1.inputs) + node1.bias
    # node2input = np.dot(row,node2.inputs) + node2.bias
    # Xtheta1 = node1input
    # Xtheta2 = node2input

    # node1input = np.log1p(np.exp(node1input))
    # node2input = np.log1p(np.exp(node2input))

    # rawY = node1input*theta3 + node2input*theta4 + bias3
    # rawN = node1input*theta5 + node2input*theta6 + bias4
    # eY = np.exp(rawY)
    # eN = np.exp(rawN)

    # pY = eY/(eY + eN)
    # pN = eN/(eY + eN)
    # print(pY)




