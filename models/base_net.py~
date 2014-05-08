import random

from models.neurons.input_neuron import InputNeuron
from models.neurons.sigmoid_neuron import SigmoidNeuron

class NeuralNet(object):
  #net type creates a fully connected net of neurons

  def __init__(self, input_length, hidden_layer_lengths, output_length):
    # Input Neurons
    self.input_layer = [InputNeuron() for _ in range(input_length + 1)] # +1 for bias

    # Hidden Neurons
    self.hidden_layers = []
    last_layer = self.input_layer
    for hidden_layer_length in hidden_layer_lengths:
      self.hidden_layers.append( [SigmoidNeuron(last_layer) for _ in range(hidden_layer_length)] ) 
      last_layer = self.hidden_layers[-1]
 
    # Output Neurons
    self.output_layer = [SigmoidNeuron(last_layer) for _ in range(output_length)] 
      


  def layers(self):
    return [self.input_layer] + self.hidden_layers + [self.output_layer]


  def run(self, *new_inputs):
    if new_inputs: self.update(*new_inputs)       
    return [neuron.fire() for neuron in self.output_layer]


  def update(self, *new_inputs):
      for index, neuron in enumerate(self.input_layer[0:-1]): #cause bias
        neuron.input = new_inputs[index]
      for layer in self.layers():
        for neuron in layer:
          neuron.output = neuron.fire()   


  def test(self, patterns):
    for p in patterns:
      print(p[0], '->', self.run(*p[0]))

  def weights(self):
    for idx, layer in enumerate(self.layers()):
      if idx == 0: continue
      print('layer ' + str(idx))
      for neuron in layer:
        print(neuron.weights)





