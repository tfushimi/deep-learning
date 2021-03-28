from common.layers import *

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        W1 = weight_init_std * np.random.randn(input_size, hidden_size)
        b1 = np.zeros(hidden_size)
        W2 = weight_init_std * np.random.randn(hidden_size, output_size)
        b2 = np.zeros(output_size)

        self.layers = [Affine(W1, b1), Sigmoid(), Affine(W2, b2)]
        self.loss_layer = SoftmaxWithLoss()

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    # x: input data, t = true label
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={"filter_num": 30, "filter_size": 5, "pad": 0, "stride": 1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param["filter_num"]
        filter_size = conv_param["filter_size"]
        filter_pad = conv_param["pad"]
        filter_stride = conv_param["stride"]
        input_size = input_dim[1]
        conv_output_size = int((input_size - filter_size + 2 * filter_pad) // filter_stride + 1)
        pool_output_size = int(filter_num * (conv_output_size//2) * (conv_output_size//2))

        W1 = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        b1 = np.zeros(filter_num)
        W2 = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        b2 = np.zeros(hidden_size)
        W3 = weight_init_std * np.random.randn(hidden_size, output_size)
        b3 = np.zeros(output_size)

        self.layers = [Convolution(W1, b1, conv_param["stride"], conv_param["pad"]),
                       ReLU(),
                       Pooling(pool_h=2, pool_w=2, stride=2),
                       Affine(W2, b2),
                       ReLU(),
                       Affine(W3, b3)]
        self.loss_layer = SoftmaxWithLoss()

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    # x: input data, t = true label
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
