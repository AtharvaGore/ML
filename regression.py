import numpy as np

with open('boston.csv','r') as f:
    f.readline()
    final=[]
    final2 = []
    for j in range(500):
        line = f.readline().split(',')
        row=[]
        for i in range(len(line)-1):
            if i ==8:
                row+=[1 if j==int(line[i] )else 0 for j in range(1,25)]
            else:
                row+=[float(line[i])]
        final.append(row)
        final2.append([float(line[len(line)-1][0:len(line[len(line)-1])-2])])
    X = np.array(final)
    Y = np.array(final2)
    #print(Y)
theta = np.array([[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]])
numeric_indices = list(range(0,8))+list(range(32,36))
X_mean = np.mean(X[:, numeric_indices], axis=0)
X_std = np.std(X[:, numeric_indices], axis=0)
X_std[X_std==0]=1
X[:, numeric_indices] = (X[:, numeric_indices] - X_mean) / X_std
# Add bias column
X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

# Initialize theta
theta = np.ones((X.shape[1], 1))
#print(theta)
for i in range(50000):
    XTheta = np.dot(X,theta)
    predictionminusy = np.subtract(XTheta,Y)
    derivative = np.dot(np.transpose(X),predictionminusy)
    theta = np.subtract(theta,0.1*(derivative/len(X)))
#print(theta)

with open('boston.csv', 'r') as f:
    for i in range(504):
        f.readline()
    line = f.readline().strip().split(',')
    print(line)
    row=[]
    for i in range(len(line)-1):
        if i ==8:
            row+=[1 if j==int(float(line[i]))else 0 for j in range(1,25)]
        else:
            row+=[float(line[i])]
    features = np.array(row)
    features[numeric_indices] = (features[numeric_indices] - X_mean) / X_std  # normalize
    inpt = np.concatenate([[1], features])
    print(np.dot(inpt, theta))  # your predicted house price
    mae = np.mean(np.abs(np.dot(X, theta) - Y))
    print("MAE:", mae)

    