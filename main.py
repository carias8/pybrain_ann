from models.base_net import NeuralNet

patterns = [
  [[0, 0], [0.0]],
  [[0, 1], [1.0]],
  [[1, 0], [1.0]],
  [[1, 1], [0.0]]
]


# create a network with two input, two hidden, and one output nodes
n = NeuralNet(2, [5, 4], 2)
#n.weights()
# train it with some patterns
n.train()
# test it
n.test(10)
