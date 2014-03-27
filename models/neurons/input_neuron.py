import random
from models.base_neuron import Neuron
class InputNeuron(Neuron):
  def __init__(self, input = False):
    self.input = ( input or random.randint(-1, 1) )
  
  def fire(self):
    return self.input
