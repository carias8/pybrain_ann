import random
from models.base_neuron import Neuron
class InputNeuron(Neuron):
  def __init__(self, input = False):
    self.input = ( input or 0.0 )
    self.output = self.fire()
  
  def fire(self):
    return self.input
