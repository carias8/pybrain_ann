from models.base_net import NeuralNet


foo = NeuralNet(2, [2, 2, 2], 2)

print foo.input_layer
print foo.output_layer
print foo.layers()
print foo.run()
print foo.run()
print foo.run()
print foo.run()
print foo.run()

