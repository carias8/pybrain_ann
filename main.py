from models.base_net import NeuralNet



# Teach network XOR function
#pat = [
#    [[0,0], [0]],
#    [[0,1], [1]],
#    [[1,0], [1]],
#    [[1,1], [0]]
#]

pat = [
    [[0,0], [0]],
    [[0,1], [1]],
    [[1,0], [1]],
    [[1,1], [0]]
]


# create a network with two input, two hidden, and one output nodes
n = NeuralNet(2, [3, 3, 3], 2)
#n.weights()
# train it with some patterns
n.train(5)
# test it
n.test(5)
#n.weights()
