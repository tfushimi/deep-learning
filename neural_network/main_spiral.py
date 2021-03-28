import sys
sys.path.append("..")
from dataset import spiral
import matplotlib.pyplot as plt
import numpy as np
from networks import TwoLayerNet
from common.optimizers import SGD
from common.trainer import Trainer

max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

x, t = spiral.load_data()
model = TwoLayerNet(input_size = 2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr=learning_rate)

data_size = len(x)
max_iters = data_size // batch_size
total_loss = 0
loss_count = 0
loss_list = []

trainer = Trainer(model, optimizer)
trainer.fit(x, t, max_epoch, batch_size)
trainer.plot()

# # 境界領域のプロット
# h = 0.001
# x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
# y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# X = np.c_[xx.ravel(), yy.ravel()]
# score = model.predict(X)
# predict_cls = np.argmax(score, axis=1)
# Z = predict_cls.reshape(xx.shape)
# plt.contourf(xx, yy, Z)
# plt.axis('off')
#
# # データ点のプロット
# x, t = spiral.load_data()
# N = 100
# CLS_NUM = 3
# markers = ['o', 'x', '^']
# for i in range(CLS_NUM):
#     plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
# plt.show()
