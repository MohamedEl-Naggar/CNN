from PIL import Image
import os
import numpy as np
import cupy as cp
from tensorflow import keras
import matplotlib.pyplot as plt
np.random.seed(10)
import time as time


def initializeFilter(size, scale = 1.0):

    stddev = scale/np.sqrt(np.prod(size))
    return np.random.normal(loc = 0, scale = stddev, size = size)
    
def softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)
def cross_entropy(y_label, out, reg_loss):
    out = softmax(out)
    losses = np.matmul(y_label.T, out)
    
    gradient = out - y_label
    loss = np.negative(np.log(losses)) + reg_loss
    if losses == np.amax(out):
        correct = 1
    else:
        correct = 0
    return loss, gradient, correct

def initialize_parameters(n_in, n_out, ini_type='plain'):
    params = dict() 

    if ini_type == 'plain':
        params['W'] = np.random.randn(n_out, n_in) *0.01  # set weights 'W' to small random gaussian
    elif ini_type == 'xavier':
        params['W'] = np.random.randn(n_out, n_in) / (np.sqrt(n_in))  # set variance of W to 1/n
    elif ini_type == 'he':
        params['W'] = np.random.randn(n_out, n_in) * np.sqrt(2/n_in)  # set variance of W to 2/n

    params['b'] = np.zeros((n_out, 1))    # set bias 'b' to zeros

    return params

def nanargmax(arr):
    idx = np.nanargmax(arr)
    idxs = np.unravel_index(idx, arr.shape)
    return idxs

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0], batchsize):
        end_idx = min(start_idx + batchsize, inputs.shape[0])
        if shuffle:
            excerpt = indices[start_idx:end_idx]
        else:
            excerpt = slice(start_idx, end_idx)
        yield inputs[excerpt], targets[excerpt]

# Define Dense layer class---------------------------------------------------
class dense_layer:
    def __init__ (self, in_shape, num_neurons = 10, weigh_init_type = "xavier", activ_fun = "relu", lamb = 0.1):
        self.activation_fun = activ_fun
        self.lamb = lamb
        self.parameters = initialize_parameters(in_shape, num_neurons, weigh_init_type)
        self.weights = self.parameters['W']
        self.bias = self.parameters['b']
        self.output = np.empty(num_neurons)  #1111
        self.dw = 0
        self.db = 0
        self.mo_w = 0
        self.acc_w = 0
        self.mo_b = 0
        self.acc_b = 0

    def forward (self, data_in, reg_loss_prev):
        self.data_in = np.reshape(data_in, (data_in.shape[0], 1))
        self.output = np.matmul(self.weights, self.data_in) + self.bias

        if self.activation_fun == "relu": #leaky relu
            self.output = np.maximum(0.01*self.output, self.output)
        elif self.activation_fun == "linear":
            self.output = self.output
        
        self.reg_loss = 0.5*self.lamb*np.sum(self.weights*self.weights) + reg_loss_prev
        self.output = np.reshape(self.output, self.output.shape[0])
    def backprop (self, gradient):
        if self.activation_fun == "relu": #leaky relu
            dx = np.ones_like(self.output)
            dx[self.output<0] = 0.01
            dout = gradient * dx
        elif self.activation_fun == "linear":
            dout = gradient

        dout = np.reshape(dout, (dout.shape[0], 1))
        self.data_in = np.reshape(self.data_in, (self.data_in.shape[0], 1))
        
        self.dw += cp.dot(dout, cp.transpose(self.data_in)) + self.lamb * self.weights
        self.db += cp.sum(dout, axis = 1, keepdims=True)
        self.Ddata_in = cp.dot(cp.transpose(self.weights), dout)

        self.Ddata_in = np.reshape(self.Ddata_in, self.Ddata_in.shape[0])

    def update (self, batch_size, lr = 0.001, beta1 = 0.7, beta2 = 0.9):
        self.dw /= batch_size
        self.bias /= batch_size

        self.mo_w = beta1 * self.mo_w + (1-beta1) * self.dw
        self.acc_w = beta2 * self.acc_w + (1-beta2) * (self.dw*self.dw)

        self.mo_b = beta1 * self.mo_b + (1-beta1) * self.bias
        self.acc_b = beta2 * self.acc_b + (1-beta2) * (self.bias*self.bias)

        self.weights += - lr * self.mo_w / (np.sqrt(self.acc_w) + 1e-7)
        self.bias += - lr * self.mo_b / (np.sqrt(self.acc_b) + 1e-7)

        self.dw = 0
        self.db = 0

# Define convolutional layer class---------------------------------------------------
class conv_layer:
    def __init__ (self, filt_size, stride, image_size, pad):
        self.pad = pad
        self.filter = initializeFilter(size = filt_size)
        (self.num_filt, self.num_ch_filt, self.filt_size, _) = self.filter.shape
        self.bias = np.zeros((self.num_filt, 1))
        self.stride = stride
        out_size = int((image_size + 2*pad - self.filt_size)/self.stride) + 1
        self.out = np.empty((self.num_filt ,out_size, out_size))

        self.df = np.zeros(self.filter.shape)
        self.db = np.zeros(self.bias.shape)

        self.mo_f = 0
        self.acc_f = 0
        self.mo_b = 0
        self.acc_b = 0
        print(self.out.shape)

    def forward (self, image):
        self.image_size_before = image.shape
        self.image = np.pad(image, ((0,0), (self.pad,self.pad), (self.pad,self.pad)), 'constant', constant_values = 0)
        self.image_size = self.image.shape[1]
        for curr_f in range(self.num_filt):
            curr_y = out_y = 0
            while curr_y + self.filt_size <= self.image_size:
                curr_x = out_x = 0
                while curr_x + self.filt_size <= self.image_size:
                    self.out[curr_f, out_y, out_x] = cp.sum(self.filter[curr_f] * self.image[:, curr_y:curr_y+self.filt_size, curr_x:curr_x+self.filt_size]) + self.bias[curr_f]
                    curr_x += self.stride
                    out_x += 1
                curr_y += self.stride
                out_y += 1
        self.out = np.maximum(self.out*0.01, self.out) # relu
    
    def backprop(self, gradient):
        self.dout = np.zeros(self.image_size_before)
        df_temp = np.zeros(self.filter.shape)

        # gradient[self.out<=0] = 0 # relu derivative

        dx = np.ones_like(self.out)
        dx[self.out<0] = 0.01
        gradient = gradient * dx

        for curr_f in range(self.num_filt):
            curr_y = out_y = 0
            while curr_y + self.filt_size <= self.image_size_before[1]:
                curr_x = out_x = 0
                while curr_x + self.filt_size <= self.image_size_before[1]:
                    df_temp[curr_f] += gradient[curr_f, out_y, out_x] * self.image[:, curr_y:curr_y+self.filt_size, curr_x:curr_x+self.filt_size]
                    self.dout[:, curr_y:curr_y+self.filt_size, curr_x:curr_x+self.filt_size] += gradient[curr_f, out_y, out_x] * self.filter[curr_f]

                    curr_x += self.stride
                    out_x += 1
                curr_y += self.stride
                out_y += 1

            self.db[curr_f] += cp.sum(gradient[curr_f])
        self.df += df_temp
    
    def update (self, batch_size, lr = 0.001, beta1 = 0.7, beta2 = 0.9):
        self.df /= batch_size
        self.db /= batch_size

        self.mo_f = beta1 * self.mo_f + (1-beta1) * self.df
        self.acc_f = beta2 * self.acc_f + (1-beta2) * (self.df*self.df)

        self.mo_b = beta1 * self.mo_b + (1-beta1) * self.db
        self.acc_b = beta2 * self.acc_b + (1-beta2) * (self.db*self.db)

        self.filter += - lr * self.mo_f / (np.sqrt(self.acc_f) + 1e-7)
        self.bias += - lr * self.mo_b / (np.sqrt(self.acc_b) + 1e-7)

        self.df = np.zeros(self.df.shape)
        self.db = np.zeros(self.db.shape)

# Define Max Pooling layer class---------------------------------------------------
class maxpool:
    def __init__ (self, filt_size, stride, image_size):
        self.filt_size = filt_size
        self.stride = stride
        self.num_ch, self.im_width, self.im_height = image_size
        out_size = int((self.im_width - self.filt_size)/self.stride) + 1
        self.out = np.empty((self.num_ch, out_size, out_size))
        print(self.out.shape)

    def forward (self, image):
        self.image = image
        for ch in range(self.num_ch):
            curr_y = out_y = 0
            while curr_y + self.filt_size <= self.im_height:
                curr_x = out_x = 0
                while curr_x + self.filt_size <= self.im_width:
                    self.out[ch, out_y, out_x] = cp.max(image[ch, curr_y:curr_y+self.filt_size, curr_x:curr_x+self.filt_size])
                    curr_x += self.stride
                    out_x += 1
                curr_y += self.stride
                out_y += 1

    def backprop (self, gradient):
        self.dout = np.zeros(self.image.shape) 

        for ch in range(self.num_ch):
            curr_y = out_y = 0
            while curr_y + self.filt_size <= self.im_height:
                curr_x = out_x = 0
                while curr_x + self.filt_size <= self.im_width:
                    (index_y, index_x) = nanargmax(self.image[ch, curr_y:curr_y+self.filt_size, curr_x:curr_x+self.filt_size])
                    self.dout[ch, curr_y+index_y, curr_x+index_x] = gradient[ch, out_y, out_x]
                    curr_x += self.stride
                    out_x += 1
                curr_y += self.stride
                out_y += 1

# load the images--------------------------------------------------
data = []
data_label = []
test_data = []
test_label = []
newsize = (20,20)

folder = 'C:/Users/moham/OneDrive/Desktop/5/PMDL/Assig.3/flower_photos'
# folder = '/home/ml2/Desktop/flower_photos'
counter = 0
for foldername in os.listdir(folder):
    for filename in os.listdir(os.path.join(folder, foldername)):
        image = Image.open(os.path.join(folder, foldername, filename))
        image = image.resize(newsize)
        data.append(np.asarray(image))
        data_label.append(counter)
    counter += 1
data = np.array(data)
data = data.astype('float32')
data /= 255
data = np.swapaxes(data, 1, 3)
data_label = np.array(data_label)
data_label = keras.utils.to_categorical(data_label, 5)
print("All Data size: ", data.shape)

a = np.random.permutation(data.shape[0])
np.take(data, a, axis=0, out=data)
np.take(data_label, a, axis=0, out=data_label)

folder = 'C:/Users/moham/OneDrive/Desktop/5/PMDL/Assig.3/test'
# folder = '/home/ml2/Desktop/test'
counter = 0
for foldername in os.listdir(folder):
    for filename in os.listdir(os.path.join(folder, foldername)):
        image = Image.open(os.path.join(folder, foldername, filename))
        image = image.resize(newsize)
        test_data.append(np.asarray(image))
        test_label.append(counter)
    counter += 1
test_data = np.array(test_data)
test_data = test_data.astype('float32')
test_data /= 255
test_data -= np.mean(test_data, axis=0)
test_data = np.swapaxes(test_data, 1, 3)
test_label = np.array(test_label)
test_label = keras.utils.to_categorical(test_label, 5)

data, val_data = data[:2536, :], data[2536:, :] # 80% training and 20% validation.
data_label, val_label = data_label[:2536, :], data_label[2536:, :]

val_data = test_data
val_label = test_label
print("Data size: ", data.shape)
print("Data Label size: ", data_label.shape)
print("Validation size: ", test_data.shape)
print("Validation Label size: ", test_label.shape)
# print("Test size: ", test_data.shape)
# print("Test Label size: ", test_label.shape)

#####################################################################
num_epochs = 37
lr = 0.001
beta1 = 0.9
beta2 = 0.99
lamb = 0.01 #change to 0.01

conv1 = conv_layer(filt_size = (8, data.shape[1], 3, 3), stride = 1, image_size = data.shape[2], pad = 1)
conv2 = conv_layer(filt_size = (8, conv1.num_filt, 3, 3), stride = 1, image_size = conv1.out.shape[1], pad = 1)
pool2 = maxpool(filt_size = 2, stride = 2, image_size = conv2.out.shape)

conv3 = conv_layer(filt_size = (16, conv2.num_filt, 3, 3), stride = 1, image_size = pool2.out.shape[1], pad = 1)
pool3 = maxpool(filt_size = 2, stride = 2, image_size = conv3.out.shape)

flatten_shape = pool3.out.shape[0]*pool3.out.shape[1]*pool3.out.shape[2]
hidden1 = dense_layer(in_shape = flatten_shape, num_neurons = 64, weigh_init_type = "he", lamb = lamb)
# hidden2 = dense_layer(in_shape = hidden1.output.shape[0], num_neurons = 64, weigh_init_type = "he", lamb = lamb)
out_layer = dense_layer(in_shape = hidden1.output.shape[0], num_neurons = 5, weigh_init_type = "he", lamb = lamb, activ_fun = 'linear')

costs = []
costs_test = []
acc = []
acc_test = np.zeros(5)
for epoch in range(num_epochs):
    total_cost = 0
    total_acc = 0
    for batch in iterate_minibatches(data, data_label, 32, shuffle=True):
        data_batch, data_label_batch = batch
        data_batch -= np.mean(data_batch, axis=0) # batch norm
        for i in range(len(data_batch)):
            # t = time.time()
            conv1.forward(data_batch[i])
            conv2.forward(conv1.out)
            pool2.forward(conv2.out)

            conv3.forward(pool2.out)
            pool3.forward(conv3.out)
            
            flatten = np.ndarray.flatten(pool3.out)
            hidden1.forward(flatten, 0)
            # hidden2.forward(hidden1.output, hidden1.reg_loss)
            out_layer.forward(hidden1.output, hidden1.reg_loss)

            cost, dA, correct = cross_entropy(y_label = data_label_batch[i], out = out_layer.output, reg_loss = out_layer.reg_loss)
            total_cost += cost
            total_acc += correct
            # print("Firdt time", time.time()-t)

            out_layer.backprop(dA)
            # hidden2.backprop(out_layer.Ddata_in)
            hidden1.backprop(out_layer.Ddata_in)

            pool3.backprop(np.reshape(hidden1.Ddata_in, pool3.out.shape))
            conv3.backprop(pool3.dout)

            pool2.backprop(conv3.dout)
            conv2.backprop(pool2.dout)
            conv1.backprop(conv2.dout)
            # print("time: ", time.time()-t)
        out_layer.update(lr = lr, beta1 = beta1, beta2 = beta2, batch_size = data_batch.shape[0])
        hidden1.update(lr = lr, beta1 = beta1, beta2 = beta2, batch_size = data_batch.shape[0])
        conv3.update(lr = lr, beta1 = beta1, beta2 = beta2, batch_size = data_batch.shape[0])
        conv2.update(lr = lr, beta1 = beta1, beta2 = beta2, batch_size = data_batch.shape[0])
        conv1.update(lr = lr, beta1 = beta1, beta2 = beta2, batch_size = data_batch.shape[0])

    print("Training loss at epoch#{}: {}".format(epoch, total_cost/len(data)))
    print("Training accuracy at epoch#{}: {}".format(epoch, total_acc/len(data)))
    costs.append(total_cost/len(data))
    acc.append(total_acc/len(data))


total_cost_test = 0
total_acc_test = 0
for i in range(len(test_data)): 
    conv1.forward(test_data[i])
    conv2.forward(conv1.out)
    pool2.forward(conv2.out)

    conv3.forward(pool2.out)
    pool3.forward(conv3.out)
    
    flatten = np.ndarray.flatten(pool3.out)
    hidden1.forward(flatten, 0)
    # hidden2.forward(hidden1.output, hidden1.reg_loss)
    out_layer.forward(hidden1.output, hidden1.reg_loss)

    cost_test, dumb, correct_test = cross_entropy(y_label = test_label[i], out = out_layer.output, reg_loss = out_layer.reg_loss)
    total_cost_test += cost_test
    total_acc_test += correct_test
    if i == 99:
        acc_test[0] = total_acc_test
        total_acc_test = 0
    elif i == 199:
        acc_test[1] = total_acc_test
        total_acc_test = 0
    elif i == 299:
        acc_test[2] = total_acc_test
        total_acc_test = 0
    elif i == 399:
        acc_test[3] = total_acc_test
        total_acc_test = 0
    elif i == 499:
        acc_test[4] = total_acc_test

print("Testing loss at epoch#{}: {}".format(epoch, total_cost_test/len(test_data)))

# plotting
x_axis = ["Daisy", "Dandelion", "Roses", "Sunflower", "Tulips"]
plt.plot(x_axis, acc_test, 'x')
plt.grid()
plt.xlabel('type')
plt.ylabel('accuracy')
plt.show()