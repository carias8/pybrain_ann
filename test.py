from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

net = buildNetwork(4, 8, 4, hiddenclass=TanhLayer)
ds = SupervisedDataSet(4, 4)

#samples = {}
#for inp, targ in samples:
#    ds.appendLinked(inp, targ)


trainer = BackpropTrainer(net, ds)

trainer.trainUntilConvergence()
