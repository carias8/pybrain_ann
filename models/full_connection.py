def connect(early_neurons, later_neurons):
  for neuron in later_neurons:
    neuron.inputs = early_neurons
    
