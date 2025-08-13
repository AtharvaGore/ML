import numpy as np
class Neuron:
    def __init__(self,inputs,outputs):
        self.inputs = np.array(inputs)
        self.outputs=np.array(outputs)
        self.bias=0.0