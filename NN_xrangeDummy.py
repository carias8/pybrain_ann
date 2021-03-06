import random

from base_neuron import Neuron

class net(object):
    """net type creates a fully connected net of neurons and can run them"""
    
    def newLayer(self,layer,size):
        """returns a new layer on top of the in put layer at a given size"""
        #normally weight and threshold are constants (or inputs) (or random)
        weight=1#random.randint(-1,2)
        threshold=1#random.randint(0,2)
        new=[]
        for newNeuron in range(size):
            #for loop to append out layer with (size) number of neurons
            new.append(neuron(layer,[weight]*len(layer),threshold))
        return new

    def __init__(self,lenInput,hiddenLayerLengths,lenOutput):

        #instantiates hidden layers
        self.hiddenLayers=[]
        for layer in range(len(hiddenLayerLengths)):
            #for loop creates layers of neurons on top of oneanother
            if layer==0:
                #if it is creating the first hidden layer uses inputLayer dummy to get the correct len(layer)
                self.hiddenLayers.append(self.newLayer(xrange(lenInput),hiddenLayerLengths[0]))
            elif layer>0:
                #every following layer it uses the previouse layer as input
                self.hiddenLayers.append(self.newLayer(self.hiddenLayers[layer-1],hiddenLayerLengths[layer]))

        #instantiates output layer with last hidden layer as input
        self.outputLayer=self.newLayer(self.hiddenLayers[-1],lenOutput)

    def run(self,inputLayer):

        #sets input layer to user's input
        self.inputLayer=inputLayer

        #sets first hidden layer's input to user's input layer
        for neuron in self.hiddenLayers[0]:
            neuron.inputs=self.inputLayer

        #fires every output neuron and returns output
        output=[]
        for neuron in self.outputLayer:
            output.append(neuron.fire())
        return output
        
