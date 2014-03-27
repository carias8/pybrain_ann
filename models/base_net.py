import random

from models.neurons.input_neuron import InputNeuron
from models.neurons.sigmoid_neuron import SigmoidNeuron
import models.full_connection

class NeuralNet(object):
  #net type creates a fully connected net of neurons and can run them

  def __init__(self, input_length, hidden_layer_lengths, output_length):
    # Input
    self.input_layer = [ InputNeuron() for x in range(0, input_length) ]

    # Hidden
    self.hidden_layers = []
    for hidden_layer_length in hidden_layer_lengths:
      self.hidden_layers.append([ SigmoidNeuron() for x in range(0, hidden_layer_length) ])

    # Output
    self.output_layer = [ SigmoidNeuron() for x in range(0, output_length) ]

    # Make the connections
    self.connect_all()


  def layers(self):
    return [self.input_layer] + self.hidden_layers + [self.output_layer]


  def run(self, *new_inputs):
    if new_inputs: self.set_inputs(*new_inputs)
    return [neuron.fire() for neuron in self.output_layer]


  def set_inputs(self, *new_inputs):
    index = 0
    for neuron in self.inputNeurons:
      neuron.input = new_inputs[index]
      index += 1
    self.reset_caches()


  def connect_all(self, connection_type = models.full_connection):
    latest_layer = self.input_layer
    for layer in self.hidden_layers + [self.output_layer]:
      connection_type.connect(latest_layer, layer)
      latest_layer = layer


  def reset_caches(self):
    for layer in self.layers:
      for neuron in layer:
        neuron.cached_value = False

    
        
