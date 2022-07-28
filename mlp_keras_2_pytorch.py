import torch.nn as nn
import torch.nn.functional as F
import torch

from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

import numpy as np
        
class FcNet(nn.Module):
    def __init__(self):
        super(FcNet, self).__init__()
        layers = []
        
        layers.append(nn.Linear(784, 200))
        layers.append(nn.Sigmoid())
        layers.append(nn.Linear(200, 10))
        
        self.model = nn.Sequential(*layers) 

    def forward(self,x):
        N = x.shape[0]
        x = x.view(N, -1)
        out = self.model(x)
        return out
        
def keras_to_pyt(keras_model, torch_model):
    parameters = keras_model.get_weights()
    
    pyt_state_dict = torch_model.state_dict()
    para_count = 0
    for key in pyt_state_dict:
        if para_count % 2 == 0:
            keras_weights = parameters[para_count]
            pyt_state_dict[key] = torch.Tensor(keras_weights.T)
        else:
            keras_bias = parameters[para_count]
            pyt_state_dict[key] = torch.Tensor(keras_bias)
        para_count += 1
    
    torch_model.load_state_dict(pyt_state_dict)
    
    return torch_model

def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted)


if __name__ == '__main__':
    
    path_prefix = "models/one_layer_models/"
    model_name = "mnist_fnn_1x200_sigmoid_local.h5"
    keras_model = load_model(path_prefix + model_name, custom_objects={'fn':fn, 'tf':tf})
    # keras_model = load_model(path_prefix + model_name)
    keras_model.summary()
                     
    pytorch_model = FcNet()
    pytorch_model = keras_to_pyt(keras_model, pytorch_model)
 
    path_prefix = "models/one_layer_models/"
    pytorch_model_name = "mnist_fnn_1x200_sigmoid_local"
    torch.save(pytorch_model.state_dict(), path_prefix + pytorch_model_name + ".pth")
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test.astype('float32') / 255.
    # if the model is from CNN-Cert
    # x_test -= 0.5
    y_test = y_test.astype('float32')
    y_test_a = to_categorical(y_test, 10)
    
    # mnist fashion_mnist
    inp = np.expand_dims(x_test[0:100], axis=1)
    
    # cifar10
    # inp = x_test[0:100]
    
    print('inp.shape', inp.shape)

    inp_pyt = torch.autograd.Variable(torch.from_numpy(inp.copy()).float())
    # mnist fashion_mnist
    inp_pyt = inp_pyt.reshape(-1, 28*28)
    inp_keras = np.transpose(inp.copy(), (0, 2, 3, 1))
    
    # cifar
    # inp_pyt = inp_pyt.reshape(-1, 32*32*3)
    # inp_keras = inp.copy()
    
    # mnist fashion_mnist
    inp_keras = inp_keras.reshape(-1, 28*28)
    
    # cifar
    # inp_keras = inp_keras.reshape(-1, 32*32*3)
    pyt_res = pytorch_model(inp_pyt).data.numpy()
    keras_res = keras_model.predict(x=inp_keras, verbose=1)
    cor_num = 0
    a_num = 0
    for i in range(100):
        predict1 = np.argmax(pyt_res[i])
        predict2 = np.argmax(keras_res[i])
        true_label = np.argmax(y_test_a[i])
        if predict1 != predict2:
            print("ERROR: Two model output are different!")
        elif predict1 != true_label:
            print("The model predict for {}th image is wrong".format(i+1))
            
        print('predict1 == predict2 ?', predict1==predict2)
        if predict1 == predict2:
            cor_num += 1
        if predict1 == true_label:
            a_num += 1 
    print('cor_num: ', cor_num)
    print('a_num: ', a_num)
    
    