import random
from math import sin, asin, pi

from models.neurons.input_neuron import InputNeuron
from models.neurons.sigmoid_neuron import SigmoidNeuron

def frange(a, b, step):
  while a <= b:
    yield a
    a += step

def rand(a, b):
  return (b-a)*random.random() + a

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

    
  def backPropagate(self, targets, N, M):
    if len(targets) != (len(self.output_layer)):
      raise ValueError('wrong number of target values')

    # calculate error terms for output
    for idx, neuron in enumerate(self.output_layer):
      error = targets[idx] - neuron.output      
      neuron.delta = neuron.d_activation_function(neuron.output) * error
      #print(error, targets[idx], neuron.output, neuron.delta)

    # update output weights
    for neuron in self.output_layer:
      for idx, input_neuron in enumerate(neuron.inputs):
        change = neuron.delta * input_neuron.output
        neuron.weights[idx] = neuron.weights[idx] + N*change + M*neuron.last_changes[idx]
        neuron.last_changes[idx] = change
        #print('chaaange', N*change, M*neuron.last_changes[idx])

    last_layer = self.output_layer

    for hidden_layer in reversed(self.hidden_layers):
      # calculate error terms for each hidden
      for idx, neuron in enumerate(hidden_layer):
        error = 0.0
        for last_layer_neuron in last_layer:
          error = error + last_layer_neuron.delta * last_layer_neuron.weights[idx]
        neuron.delta = neuron.d_activation_function(neuron.output) * error


      # update hidden weights
      for neuron in hidden_layer:
        for idx, input_neuron in enumerate(neuron.inputs):
          change = neuron.delta * input_neuron.output
          neuron.weights[idx] = neuron.weights[idx] + N*change + M*neuron.last_changes[idx]
          neuron.last_changes[idx] = change
      last_layer = hidden_layer


    # calculate error
    error = 0.0
    for k in range(len(targets)):
        error = error + 0.5*(targets[k]-self.output_layer[k].output)**2
    return error

  def weights(self):
    for idx, layer in enumerate(self.layers()):
      if idx == 0: continue
      print('layer ' + str(idx))
      for neuron in layer:
        print(neuron.weights())


  def train(self, iterations=10000, noise=0.001, N=0.5, M=0.1):
    patterns = []
    DATA_POINT_PER_T = 100
    STD_DEV_FOR_DISTRIBUTION = 0.2
    INCREMENTER = pi/25
    for t in frange(0, 2*pi, INCREMENTER):
      for i in range(DATA_POINT_PER_T):
        alpha = random.gauss(1.0, STD_DEV_FOR_DISTRIBUTION)
        beta = random.gauss(0.0, STD_DEV_FOR_DISTRIBUTION)
        y = alpha * sin(beta + (t))
        target_alpha_delta = 1 - alpha
        target_beta_delta = 0 - beta
        patterns.append([ [t, y], [target_alpha_delta, target_beta_delta] ])

    for pattern in patterns:
      inputs = pattern[0]
      targets = pattern[1]
      self.run(*inputs)
      print self.backPropagate(targets, N, M)
      
     
  def test(self, iterations = 1, noise = 0.01):
    for i in range(iterations):
      error = 0.0
      alpha = 1.0
      beta = 0.0

      INCREMENTER = pi/25
      for t in frange(0, 2*pi, INCREMENTER):
        # add noise
        alpha += rand( -1*abs(noise), abs(noise) )
        beta += rand( -1*abs(noise), abs(noise) )

        # get y
        y = alpha * sin(beta + (t))

        # get the deltas of the variables that the net output
        alpha_delta, beta_delta = self.run(t, y)

        # get target deltas
        target_alpha_delta = 1 - alpha
        target_beta_delta = 0 - beta

        # calculate the new actual values of the variables     
        alpha += alpha_delta
        beta += beta_delta
  
        # set inputs and target
        targets = [target_alpha_delta, target_beta_delta]

        # back propagate
        error += abs(target_alpha_delta - alpha_delta) + abs(target_beta_delta - beta_delta)

      print('error %-.5f' % error)




