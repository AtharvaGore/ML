import numpy as np
class Neuron:
    def __init__(self,inputs,outputs):
        self.inputs = np.array(inputs)
        self.outputs=np.array(outputs)
        self.bias=0.0



class NeuralNetwork:
    def __init__(self,data,expected,*nodes):
        self.data = data
        self.Y = expected
        self.hiddenNodes = nodes
        self.outputBias = [0.0,0.0]
        
    def softplus(nodeinput):
        return np.log1p(np.exp(nodeinput))

    def forwardPass(self,inputs):
        nodeinputs = []
        for i in range(len(self.hiddenNodes)):
            nodeinputs.append(np.dot(inputs,self.hiddenNodes[i].inputs) + self.hiddenNodes[i].bias)
        beforeSoftPlus = nodeinputs
        # print(beforeSoftPlus[0])
        nodeinputs = [NeuralNetwork.softplus(i) for i in nodeinputs]
        rawOutput = [self.outputBias[0],self.outputBias[1]]
        # print(rawOutput)
        for i in range(2):
            for j in range(len(nodeinputs)):
               rawOutput[i] += nodeinputs[j]*self.hiddenNodes[j].outputs[i]

        expRawOutput = [np.exp(i) for i in rawOutput]
        pY = expRawOutput[0]/(expRawOutput[0]+expRawOutput[1])
        pN = expRawOutput[1]/(expRawOutput[0]+expRawOutput[1])
        # print(pY)
        return ([pY,pN],nodeinputs,beforeSoftPlus,rawOutput)
    
    
    
    def train(self):
        rate = 0.01
        for i in range(10000):
            predictedProbabilities = self.forwardPass(self.data)
            summation=0
            self.outputBias[0] = self.outputBias[0] - rate*np.mean(predictedProbabilities[0][0]-self.Y)
            self.outputBias[1] = self.outputBias[1] - rate*np.mean(predictedProbabilities[0][1]-(1-self.Y))
            for i in range(2):
                self.hiddenNodes[i].outputs[0] -=rate*np.mean((predictedProbabilities[0][0]-self.Y)*predictedProbabilities[1][i])
                self.hiddenNodes[i].outputs[1] -=rate*np.mean((predictedProbabilities[0][1]-(1-self.Y))*predictedProbabilities[1][i])
            
            for j in range(2):
                temp = (1/(1+np.exp(-predictedProbabilities[2][j])))
                temp2 = []
                for i in range(700):
                    if self.Y[i,0] == 1:
                        temp2.append([predictedProbabilities[0][1][i,0]*(self.hiddenNodes[j].outputs[1]-self.hiddenNodes[j].outputs[0])])
                    else:
                        temp2.append([predictedProbabilities[0][0][i,0]*(self.hiddenNodes[j].outputs[0]-self.hiddenNodes[j].outputs[1])])
                temp2 = np.array(temp2)
                temp2 = temp*temp2
                summation = temp2.mean()
                #print(summation)
                self.hiddenNodes[j].bias = self.hiddenNodes[j].bias - rate*summation
                temp2 = self.data*temp2

                for i in range(8):
                    summation = temp2[:, i].mean()
                    self.hiddenNodes[j].inputs[i,0] = self.hiddenNodes[j].inputs[i,0] - rate*summation
                summation=0
                
        
        
