import math

class Trainer():
  def stencil_function(self, a, b, t):
    return ( a * math.sin(b + t/10) )


  def backPropagate(self, net, inputs, output, N, M):
    old_alpha, old_beta, t = inputs
    alpha_change, beta_change = output
    new_alpha = old_alpha + alpha_change
    new_beta = old_beta + beta_change

    # calculate error terms for output
    for idx, neuron in enumerate(net.output_layer):
      if idx == 0: # if alpha neuron
        error = (self.stencil_function(new_alpha, old_beta, t) - self.stencil_function(1, 1, t))/2
      else: # if beta neuron
        error = (self.stencil_function(old_alpha, new_beta, t) - self.stencil_function(1, 1, t))/2
      #error = targets[idx] - neuron.output      
      neuron.delta = neuron.d_activation_function(neuron.output) * error
      #print(error, targets[idx], neuron.output, neuron.delta)

    # update output weights
    for neuron in net.output_layer:
      for idx, input_neuron in enumerate(neuron.inputs):
        change = neuron.delta * input_neuron.output
        neuron.weights[idx] = neuron.weights[idx] + N*change + M*neuron.last_changes[idx]
        neuron.last_changes[idx] = change
        #print('chaaange', N*change, M*neuron.last_changes[idx])

    last_layer = net.output_layer

    for hidden_layer in net.hidden_layers:
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
    error = self.stencil_function(new_alpha, new_beta, t+1) - self.stencil_function(1, 1, t+1)
    return error

  def train(self, net, patterns, iterations=1000, time_length=10000, N=0.8, M=0.1):
    # patterns looks like this: [ [initial_alpha, initial_beta], [perfect_alpha, perfect_beta] ]
    # N: learning rate
    # M: momentum factor
    for iteration in range(iterations):
      for pattern in patterns:
        initial_state, perfect_state = pattern
        alpha, beta = initial_state
        for t in range(time_length):

          #print("\n")
          #self.weights()
          error = 0.0

          inputs = (alpha, beta, t)
          alpha_change, beta_change = net.run(*inputs)
          alpha = alpha + alpha_change
          beta = beta + beta_change
          # this is where noise will get added

          error = self.backPropagate(net, inputs, (alpha_change, beta_change), N, M)
          if (iteration % 100) == 0: print("A/B: " + str(alpha) + ", " + str(beta) + "Err: " + str(error)) 
          
      


