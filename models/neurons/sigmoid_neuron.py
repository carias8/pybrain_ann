from models.base_neuron import Neuron
class SigmoidNeuron(Neuron):

  def activation_function(self, input_sum): 
    e = 2.0
    sigmoid = (1.0 / ( (e**-input_sum) + 1.0 ))
    return sigmoid 
