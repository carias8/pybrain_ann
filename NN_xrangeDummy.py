import random

class neuron(object):
    """neuron type takes numbers or other neurons as inputs along with their weights,
    and either returns a 1 if threshold reached or a 0 if not"""
    #feed forward only
    def __init__(self,inputs,weights,threshold):
        self.threshold=threshold
        self.inputs=inputs
        self.weights=weights
        self.output=False

    def fire(self):
        """calls input nuerons to fire and returns output value"""
        summed=0
        for neuron in range(len(self.inputs)):
            try:
                #trys to get output from neuron and sums them * weight
                summed+=(self.inputs[neuron].fire())*(self.weights[neuron])
            except AttributeError:
                #excepts static integers and sums them * weight
                summed+=(self.inputs[neuron])*(self.weights[neuron])

        e=2.0
        #sigmoid=1.0/(1.0+(e**-summed))
        sigmoid=summed

        if sigmoid>=self.threshold:
            #outputs 1 if threshold exceeded
            self.output=True

        elif sigmoid<self.threshold:
            #outputs 0 if threshold not exceeded
            self.output=False

        #print self.output
        return self.output


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
        

###Random Net###

##in0=1
##in1=1
##in2=1
##
##inLayer=[in0,in1,in2]
##hiddenLayer0=newLayer(inLayer,random.randint(3,3))
##hiddenLayer1=newLayer(hiddenLayer0,random.randint(3,3))
##hiddenLayer2=newLayer(hiddenLayer1,random.randint(3,3))
##outLayer=newLayer(hiddenLayer2,1)
##
##out=outLayer[0]

###Logic Gates###

###and
##in0=True
##in1=True
##out=neuron([in0,in1],[1,1],2)
##print in0,in1
##print out.fire()

###or
##in0=False
##in1=False
##out=neuron([in0,in1],[1,1],1)
##print in0,in1
##print out.fire()

###xor (not equal to)
##in0=True
##in1=True
##n0=neuron([in0,in1],[1,1],1)
##n1=neuron([in0,in1],[1,1],2)
##out=neuron([n0,n1],[1,-1],1)
##print in0,in1
##print out.fire()

###not
##in0=True
##n0=neuron([in0],[1],1)
##n1=neuron([in0],[1],0)
###xor
##n2=neuron([n0,n1],[1,1],1)
##n3=neuron([n0,n1],[1,1],2)
##out=neuron([n2,n3],[1,-1],1)
##print in0
##print out.fire()
