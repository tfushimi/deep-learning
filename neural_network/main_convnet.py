import sys
sys.path.append("..")
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt
import numpy as np
from networks import SimpleConvNet
from common.optimizers import Adam
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, one_hot_label=True)

max_epoch = 50
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

model = SimpleConvNet(input_dim=(1,28,28),
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
optimizer = Adam()

data_size = len(x_train)
max_iters = data_size // batch_size

trainer = Trainer(model, optimizer)
trainer.fit(x_train, t_train, max_epoch, batch_size)
trainer.plot()


pred = np.argmax(model.predict(x_test), axis=1)
label = np.argmax(t_test, axis=1)
print("Accuracy on test is %.3f" % np.mean(pred==label))
