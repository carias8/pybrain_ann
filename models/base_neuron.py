import random

def rand(a, b):
  return (b-a)*random.random() + a

class Neuron(object):

  # neuron type takes other neurons as inputs along with their weights
  # feed forward only
  def __init__(self, inputs, weights = False):
    self.inputs = inputs    
    self.weights = ( weights or [rand(-0.2, 0.2) for _ in inputs] )
    self.output = self.fire()
    self.delta = False
    self.last_changes = [0.0] * len(self.weights)

  def activation_function(self, input_sum):
    return input_sum  
  
  # inverse of derivative of activation function
  def d_activation_function(self, output):
    return 1

  def fire(self):
    input_values = [neuron.output for neuron in self.inputs]
    weighted_inputs = [ x[0]*x[1] for x in zip(input_values, self.weights) ]
    input_sum = sum(weighted_inputs)

    return self.activation_function(input_sum)

 
      
      
