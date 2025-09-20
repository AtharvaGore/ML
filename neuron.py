

import numpy as np
# initializes new neuron. each neuron has inputs, which are the weights going into it and outputs which are the weights going out of it to the output layer.
# the bias is the input bias and the output bias is not included in this class
class Neuron:
    def __init__(self,inputs,outputs):
        self.inputs = np.array(inputs)
        self.outputs=np.array(outputs)
        self.bias=0.0



class NeuralNetwork:
    #initializes everything
    def __init__(self,data,expected,outputs,test,nodes):
        self.data = data
        self.Y = expected
        self.hiddenNodes = nodes
        self.test = test
        self.outputBias = [0.0 for i in range(outputs)]
        
    # takes in a input and applies softplus on it
    def softplus(x):
        return np.where(x > 0, x + np.log1p(np.exp(-x)), np.log1p(np.exp(x)))
        # this is complicated as it manages overflow errors and stuff. u can use basic softplus but there is a bigger risk of overflow errors

    def forwardPass(self,inputs):
        # this function takes in a input matrix and passes it through the neural network. each row of the matrix is the first test case, the next row is the 
        # next test case and so on

        # this contains the input weight vectors of all the nodes. it is stored as column vectors. first bias contains all the input biases of the nodes.
        firstNodeInputs = np.column_stack([n.inputs.flatten() for n in self.hiddenNodes])
        firstBias = np.array([n.bias for n in self.hiddenNodes])

        # same thing as above but for the output nodes and the output biases
        hiddenNodeoutputs = np.vstack([n.outputs for n in self.hiddenNodes])
        hiddenNodeBiases = np.array(self.outputBias)

        """ the first layer processing. row of inputs multiplies with a column of firstnodeinputs to get an element which is added with the firstbias for that
        # node. that is ur first node number. same input row is used for all the columns, each column is the weights of the different nodes. so one row in the
        # result is the same inputs processed through all the input weights. the next row is the same but for a different input test case.
        """
        hiddenLayerInputs = (inputs @ firstNodeInputs) + firstBias


        """ this applies the softplus function to all the elements of the matrix. a new variable is created cuz we need the non softplused version inside the
        training loop """
        afterSoftplus = NeuralNetwork.softplus(hiddenLayerInputs)

        """ all the aftersoftplus values are now passed throught the output weights to get the raw output. its very similar to the first layer processing just
        that each row in aftersoftplus is still a different test case but modified by the first layer weights and softplused also"""
        rawOutput = (afterSoftplus @ hiddenNodeoutputs) + hiddenNodeBiases

        """ this line stabilizes and prevents overflow errors. how? idk lmao but it still works without this line. its just for safety im pretty sure"""
        shifted = rawOutput - rawOutput.max(axis=1, keepdims=True)
        
        expRaw = np.exp(shifted)
        predictions = [expRaw/expRaw.sum(axis=1, keepdims=True)]
        rawOutput = [rawOutput]
        afterSoftplus = [afterSoftplus]
        hiddenLayerInputs = [hiddenLayerInputs]
        # print(np.shape(np.transpose(predictions)))
        # print(np.shape(hiddenLayerInputs))
        # print(np.shape(afterSoftplus))
        #print(np.shape(expRaw))
        

        # nodeinputs = []
        # for i in range(len(self.hiddenNodes)):
        #     nodeinputs.append(np.dot(inputs,self.hiddenNodes[i].inputs) + self.hiddenNodes[i].bias)
        # beforeSoftPlus = nodeinputs
        # # print(beforeSoftPlus[0])
        # nodeinputs = [NeuralNetwork.softplus(i) for i in nodeinputs]
        # rawOutput = [self.outputBias[i] for i in range(len(self.outputBias))]
        # # print(rawOutput)
        # for i in range(len(self.outputBias)):
        #     for j in range(len(nodeinputs)):
        #        rawOutput[i] += nodeinputs[j]*self.hiddenNodes[j].outputs[i]

        # expRawOutput = [np.exp(i) for i in rawOutput]
        # sumi=0
        # for i in expRawOutput:
        #     sumi+=i
        # predictions=[]
        # for i in range(len(self.outputBias)):
        #     predictions.append(expRawOutput[i]/sumi)
        # print(np.shape(predictions))
        # print(np.shape(beforeSoftPlus))
        # print(np.shape(nodeinputs))
        # print(np.shape(rawOutput))
        

        # pY = expRawOutput[0]/(expRawOutput[0]+expRawOutput[1])
        # pN = expRawOutput[1]/(expRawOutput[0]+expRawOutput[1])
        # print(pY)
        return (np.transpose(predictions),np.transpose(afterSoftplus),np.transpose(hiddenLayerInputs),np.transpose(rawOutput),np.transpose(expRaw))
    
    
    
    def train(self,epochs):
        rate = 0.001
        for i in range(epochs):
            batchsize = 100
            for start in range(0, 50000, batchsize):
                end = start + batchsize
                batch_X = self.data[start:end]
                predictedProbabilities = self.forwardPass(batch_X)
                onehot = self.test[start:end]
                summation=0
                for i in range(len(self.outputBias)):
                    self.outputBias[i] = self.outputBias[i] - rate*np.mean(predictedProbabilities[0][i]-onehot[:, [i]])
                # self.outputBias[0] = self.outputBias[0] - rate*np.mean(predictedProbabilities[0][0]-self.Y)
                # self.outputBias[1] = self.outputBias[1] - rate*np.mean(predictedProbabilities[0][1]-(1-self.Y))
                for i in range(len(self.hiddenNodes)):
                    for j in range(len(self.outputBias)):
                        self.hiddenNodes[i].outputs[j] -= rate * np.mean((predictedProbabilities[0][j] - onehot[:, [j]]) * predictedProbabilities[1][i])
                    # self.hiddenNodes[i].outputs[1] -=rate*np.mean((predictedProbabilities[0][1]-(1-self.Y))*predictedProbabilities[1][i])
                    
                
                # for j in range(len(self.hiddenNodes)):
                #     temp = (1/(1+np.exp(-predictedProbabilities[2][j])))
                #     temp2 = []
                trueclasses = np.argmax(onehot,axis=1)
                #     weights = self.hiddenNodes[j].outputs
                #     wdiff = weights - weights[trueclasses][:, None]
                #     vectors = np.array(predictedProbabilities[0])
                #     vectors = vectors.squeeze(axis=-1).T 
                #     temp2 = np.sum(vectors * wdiff, axis=1, keepdims=True)
                #     # for i in range(700):
                #     #     trueclasses = np.argmax(self.test,axis=1)
                        
                #     #     temp5 = np.array([0.0])
                #     #     for l in range(len(self.outputBias)):
                #     #         temp5+= predictedProbabilities[0][l][i,0]*(self.hiddenNodes[j].outputs[l]-self.hiddenNodes[j].outputs[trueclasses[i]])
                #     #     temp2.append(temp5)
                        
                #     #temp2 = np.array(temp2)
                #     temp2 = temp*temp2
                #     summation = temp2.mean()
                #     #print(summation)
                #     self.hiddenNodes[j].bias = self.hiddenNodes[j].bias - rate*summation
                #     temp2 = self.data*temp2

                #     for i in range(783):
                #         summation = temp2[:, i].mean()
                #         self.hiddenNodes[j].inputs[i,0] = self.hiddenNodes[j].inputs[i,0] - rate*summation
                #     summation=0
                # raw = np.transpose(predictedProbabilities[0])
                # raw = np.column_stack(raw)
                # raw = np.transpose(raw)
                expRaw = predictedProbabilities[4]
                #print(trueclasses)
                temp = []
                for j in trueclasses:
                    temp2 = []
                    for i in self.hiddenNodes:
                        temp2.append(i.outputs - i.outputs[j])
                    temp2 = np.array(temp2)
                    temp.append(temp2)
                
                temp = np.array(temp)
                stage1 = [temp[i] @ expRaw[:, i] for i in range(len(temp))]
                #print(np.shape(expRaw))
                sums = [expRaw[:, i].sum() for i in range(batchsize)]
                stage1 = np.transpose(np.array(stage1))
                stage1 = stage1/np.array(sums)
                #print(np.shape(stage1))
                
                temp2 = np.squeeze(predictedProbabilities[2])
                temp = (1/(1+np.exp(temp2)))
                stage2 = stage1*temp
                #print(np.shape(temp))
                stage3 = np.array([stage2[:, i][:, None]*batch_X[i ,:] for i in range(batchsize)])
                # print(np.shape(np.array(stage3)))
                stage3 = stage3.sum(axis=0)
                stage3 = stage3*rate
                
                oldweights = np.row_stack(np.array([np.transpose(i.inputs) for i in self.hiddenNodes]))
                oldweights = oldweights - stage3
                for i in range(len(self.hiddenNodes)):
                    self.hiddenNodes[i].inputs = oldweights[i ,:].copy()

                oldbiases = np.array([[i.bias] for i in self.hiddenNodes])
                stage2 = stage2.sum(axis=0)
                stage2 = stage2*rate
                oldbiases = oldbiases - stage2

                for i in range(len(self.hiddenNodes)):
                    self.hiddenNodes[i].bias = oldbiases[i][0]


            
                
        
        
