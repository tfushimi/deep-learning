import sys
sys.path.append("..")
from dataset.mnist import load_mnist
import numpy as np
from networks import TwoLayerNet
from common.optimizers import SGD
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

max_epoch = 100
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

model = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
optimizer = SGD(lr=learning_rate)

data_size = len(x_train)
max_iters = data_size // batch_size

trainer = Trainer(model, optimizer)
trainer.fit(x_train, t_train, max_epoch, batch_size)
trainer.plot()

pred = np.argmax(model.predict(x_test), axis=1)
label = np.argmax(t_test, axis=1)
print("Accuracy on test is %.3f" % np.mean(pred==label))
