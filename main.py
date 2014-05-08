from models.base_net import NeuralNet
from models.trainer import Trainer



# Teach network XOR function
#pats = [
#    [[0,0], [0]],
#    [[0,1], [1]],
#    [[1,0], [1]],
#    [[1,1], [0]]
#]

pats = [
    [[1.0, 1.0], [1, 1]],
    [[1.01, 0.99], [1, 1]],
]



# create a network with two input, two hidden, and one output nodes
n = NeuralNet(3, [4], 2)
#n.weights()
# train it with some patterns
trainer = Trainer()
trainer.train(n, pats)
# test it
n.test(pats)
#n.weights()
