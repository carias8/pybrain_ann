from models.base_net import NeuralNet



# Teach network XOR function
#pat = [
#    [[0,0], [0]],
#    [[0,1], [1]],
#    [[1,0], [1]],
#    [[1,1], [0]]
#]

pat = [
    [[0,0], [1]],
    [[0,1], [0]],
    [[1,0], [0]],
    [[1,1], [1]]
]


# create a network with two input, two hidden, and one output nodes
n = NeuralNet(2, [2], 1)
#n.weights()
# train it with some patterns
n.train(pat)
# test it
n.test(pat)
#n.weights()
