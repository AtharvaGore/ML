import numpy as np
import neuron
final,final2 = [],[]

with open('mnist_train.csv','r') as f:
    f.readline()
    for i in range(50000):
        row = f.readline().split(',')
        final2.append([float(row[0])])
        row = row[1:len(row)-1]
        temp=[]
        for j in row:
            temp+=[float(j)]
        final.append(temp)


def xavier_init(fan_in, fan_out):
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))

weights = xavier_init(783, 10)

# Convert to your required format:
# column vector for inputs, list for outputs
weights_list = [[w] for w in weights[:, 0]]  # 784x1 column vector
outputs_list = list(weights[0, :]) 

weights = xavier_init(783, 10)

# Convert to your required format:
# column vector for inputs, list for outputs
weights_list2 = [[w] for w in weights[:, 0]]  # 784x1 column vector
outputs_list2 = list(weights[0, :]) 

abcd = []

# Xavier init for input → hidden (783 inputs, 128 hidden neurons)
W_input_hidden = xavier_init(783, 128)   # shape (783, 128)

# Xavier init for hidden → output (128 hidden neurons, 10 outputs)
W_hidden_output = xavier_init(128, 10)   # shape (128, 10)

for j in range(128):
    # Input weights for this neuron (column j of W_input_hidden)
    weights_list = [[w] for w in W_input_hidden[:, j]]  # shape (783, 1)

    # Output weights for this neuron (row j of W_hidden_output)
    outputs_list = list(W_hidden_output[j, :])          # length 10

    abcd.append(neuron.Neuron(weights_list, outputs_list))

X=np.array(final)
Y=np.array(final2)

X_mean = np.mean(X,axis=0)
X_std = np.std(X,axis=0)
X_std[X_std == 0] = 1
X = (X-X_mean)/X_std

node3 = neuron.Neuron(weights_list,outputs_list)
node4 = neuron.Neuron(weights_list2,outputs_list2)
node5 = neuron.Neuron([[0.321],[-0.142],[0.567],[-0.654],[0.123],[-0.298],[0.432],[-0.075]],[-0.298, 0.411])

onehot = np.eye(10)[Y.ravel().astype(int)]
# onehot = onehot[:, ::-1]

accuracyList = []
numberofepochs = []
for i in range(15,51,2):
    numberofepochs.append(i)
    obj = neuron.NeuralNetwork(X,Y,10,onehot,tuple(abcd))

    obj.train(i)
    with open('mnist_test.csv','r') as f:
        f.readline()
        count=0
        for i in range(9000):
            row = f.readline().split(',')
            real = int(row[0])
            #print(row[0],end="")
            row = row[1:len(row)-1]
            temp=[]
            for j in row:
                temp+=[float(j)]
            row = np.array(temp)
            row = (row-X_mean)/X_std
            #print(" "+str(np.argmax(obj.forwardPass(np.expand_dims(row, axis=0))[0])))
            if real == int(np.argmax(obj.forwardPass(np.expand_dims(row, axis=0))[0])):
                count+=1
        print(f"Model accuracy: {count/9000} Number of epochs: {i}")
        accuracyList.append(count/9000)