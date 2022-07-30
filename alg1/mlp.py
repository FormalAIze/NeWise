import torch
import torch.nn as nn

class BinarizeActivation(nn.Module):
    def __init__(self):
        super(BinarizeActivation, self).__init__()
    def forward(self, a):
        a = torch.clamp(a, min=-1, max=1)

        # set the data s.t. return binary output (+1,-1), and only compute grad on clamp function (for training stability)
        a.data = torch.sign(a.data)
        return a

Activation_Function = {'relu':nn.ReLU(),'relu_adaptive':nn.ReLU(), 'ba':BinarizeActivation(),
                'tanh':nn.Tanh(), 'sigmoid':nn.Sigmoid()}

class FcNet(nn.Module):
    def __init__(self, num_layers, num_neurons, input_dimension, output_dimension, 
                bn=True, affine=False, activation='ba', dropout=False, final_bn=True):
        super(FcNet, self).__init__()
        layers = []
        self.bn = bn #bn have a huge influence on the final model accuracy
        self.activation = activation
        self.num_layers = num_layers
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        
        layers.append(nn.Linear(input_dimension, num_neurons))
        layers.append(Activation_Function[activation])
        layers.append(nn.Linear(num_neurons, output_dimension))
        
        self.model = nn.Sequential(*layers)
        self.model_seq = None # used to track preactivations

    def forward(self,x):
        N = x.shape[0]
        x = x.view(N, -1)
        out = self.model(x)
        return out



