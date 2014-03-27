import random

class Neuron(object):

  # neuron type takes other neurons as inputs along with their weights
  # feed forward only
  def __init__(self, inputs = [], weights = False, threshold = False):
    self.inputs = inputs    
    self.weights = ( weights or [random.randint(-1, 1) for x in range( 0, len(inputs) )] )
    self.threshold = ( threshold or random.randint(0, 2) )
    self.cached_value = False
    self.output = 0

  def activation_function(self, input_sum):
    return input_sum  

  def output_function(self, activation_value):
    return activation_value

  def fire(self):
    if self.cached_value: return self.cached_value 

    input_values = [neuron.fire() for neuron in self.inputs]
    weighted_inputs = [ x[0]*x[1] for x in zip(input_values, self.weights) ]
    input_sum = sum(weighted_inputs)

    activation_value = self.activation_function(input_sum)

    if activation_value >= self.threshold:
        self.output = self.output_function(activation_value)
    else:
        self.output = 0
    
    self.cached_value = self.output
    return self.output

 
      
      
