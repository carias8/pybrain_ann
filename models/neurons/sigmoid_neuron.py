from models.base_neuron import Neuron
import math

class SigmoidNeuron(Neuron):

  def activation_function(self, input_sum): 
    return math.tanh(input_sum) 

  # inverse of derivative of activation function
  def d_activation_function(self, output):
    return 1.0 - output**2	
