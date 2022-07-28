import numpy as np
from cnn_bounds_full_core_with_LP import run_certified_bounds_core
from cnn_bounds_full_with_LP import run_certified_bounds

import time as timing
import datetime

def printlog(s, log_name):
    print(s, file=open("logs/"+log_name+".txt", "a"))

if __name__ == '__main__':
    
    path_prefix = "models/models_with_positive_weights/sigmoid/"
    log_name = "table_two"
    
    # MNIST
    run_certified_bounds(path_prefix + 'mnist_ffnn_5x100_with_positive_weights_8858.h5', 100, 105, 1, method='NeWise', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_5x100_with_positive_weights_8858.h5', 100, 105, 1, method='DeepCert', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_5x100_with_positive_weights_8858.h5', 100, 105, 1, method='VeriNet', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_5x100_with_positive_weights_8858.h5', 100, 105, 1, method='RobustVerifier', mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x700_with_positive_weights_9537.h5', 100, 105, 1, method='NeWise', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x700_with_positive_weights_9537.h5', 100, 105, 1, method='DeepCert', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x700_with_positive_weights_9537.h5', 100, 105, 1, method='VeriNet', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x700_with_positive_weights_9537.h5', 100, 105, 1, method='RobustVerifier', mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3_with_positive_weights.h5', 100, 105, 1, method='NeWise', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3_with_positive_weights.h5', 100, 105, 1, method='DeepCert', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3_with_positive_weights.h5', 100, 105, 1, method='VeriNet', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3_with_positive_weights.h5', 100, 105, 1, method='RobustVerifier', mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x50_with_positive_weights_9118.h5', 100, 105, 1, method='NeWise', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x50_with_positive_weights_9118.h5', 100, 105, 1, method='DeepCert', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x50_with_positive_weights_9118.h5', 100, 105, 1, method='VeriNet', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x50_with_positive_weights_9118.h5', 100, 105, 1, method='RobustVerifier', mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x100_with_positive_weights_9162.h5', 100, 105, 1, method='NeWise', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x100_with_positive_weights_9162.h5', 100, 105, 1, method='DeepCert', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x100_with_positive_weights_9162.h5', 100, 105, 1, method='VeriNet', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x100_with_positive_weights_9162.h5', 100, 105, 1, method='RobustVerifier', mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3_with_positive_weights.h5', 100, 105, 1, method='NeWise', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3_with_positive_weights.h5', 100, 105, 1, method='DeepCert', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3_with_positive_weights.h5', 100, 105, 1, method='VeriNet', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3_with_positive_weights.h5', 100, 105, 1, method='RobustVerifier', mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_2_3_with_positive_weights_9120.h5', 100, 105, 1, method='NeWise', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_2_3_with_positive_weights_9120.h5', 100, 105, 1, method='DeepCert', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_2_3_with_positive_weights_9120.h5', 100, 105, 1, method='VeriNet', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_2_3_with_positive_weights_9120.h5', 100, 105, 1, method='RobustVerifier', mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_with_positive_weights_8563.h5', 100, 105, 1, method='NeWise', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_with_positive_weights_8563.h5', 100, 105, 1, method='DeepCert', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_with_positive_weights_8563.h5', 100, 105, 1, method='VeriNet', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_with_positive_weights_8563.h5', 100, 105, 1, method='RobustVerifier', mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_4_3_with_positive_weights_9151.h5', 100, 105, 1, method='NeWise', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_4_3_with_positive_weights_9151.h5', 100, 105, 1, method='DeepCert', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_4_3_with_positive_weights_9151.h5', 100, 105, 1, method='VeriNet', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_4_3_with_positive_weights_9151.h5', 100, 105, 1, method='RobustVerifier', mnist=True, log_name=log_name)
    
    # Fashion MNIST
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_4x100_with_positive_weights_sigmoid.h5', 100, 105, 1, method='NeWise', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_4x100_with_positive_weights_sigmoid.h5', 100, 105, 1, method='DeepCert', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_4x100_with_positive_weights_sigmoid.h5', 100, 105, 1, method='VeriNet', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_4x100_with_positive_weights_sigmoid.h5', 100, 105, 1, method='RobustVerifier', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_3x100_with_positive_weights_sigmoid.h5', 100, 105, 1, method='NeWise', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_3x100_with_positive_weights_sigmoid.h5', 100, 105, 1, method='DeepCert', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_3x100_with_positive_weights_sigmoid.h5', 100, 105, 1, method='VeriNet', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_3x100_with_positive_weights_sigmoid.h5', 100, 105, 1, method='RobustVerifier', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_5layer_5_3_with_positive_weights_sigmoid.h5', 100, 105, 1, method='NeWise', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_5layer_5_3_with_positive_weights_sigmoid.h5', 100, 105, 1, method='DeepCert', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_5layer_5_3_with_positive_weights_sigmoid.h5', 100, 105, 1, method='VeriNet', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_5layer_5_3_with_positive_weights_sigmoid.h5', 100, 105, 1, method='RobustVerifier', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_3layer_2_3_with_positive_weights_sigmoid.h5', 100, 105, 1, method='NeWise', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_3layer_2_3_with_positive_weights_sigmoid.h5', 100, 105, 1, method='DeepCert', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_3layer_2_3_with_positive_weights_sigmoid.h5', 100, 105, 1, method='VeriNet', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_3layer_2_3_with_positive_weights_sigmoid.h5', 100, 105, 1, method='RobustVerifier', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_4layer_5_3_with_positive_weights_sigmoid.h5', 100, 105, 1, method='NeWise', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_4layer_5_3_with_positive_weights_sigmoid.h5', 100, 105, 1, method='DeepCert', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_4layer_5_3_with_positive_weights_sigmoid.h5', 100, 105, 1, method='VeriNet', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_4layer_5_3_with_positive_weights_sigmoid.h5', 100, 105, 1, method='RobustVerifier', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_3layer_4_3_with_positive_weights_sigmoid.h5', 100, 105, 1, method='NeWise', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_3layer_4_3_with_positive_weights_sigmoid.h5', 100, 105, 1, method='DeepCert', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_3layer_4_3_with_positive_weights_sigmoid.h5', 100, 105, 1, method='VeriNet', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_3layer_4_3_with_positive_weights_sigmoid.h5', 100, 105, 1, method='RobustVerifier', fashion_mnist=True, log_name=log_name)
    
    # Cifar10
    run_certified_bounds(path_prefix + 'cifar10_ffnn_9x100_with_positive_weights_2913_cpu.h5', 100, 105, 1, method='NeWise', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_9x100_with_positive_weights_2913_cpu.h5', 100, 105, 1, method='DeepCert', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_9x100_with_positive_weights_2913_cpu.h5', 100, 105, 1, method='VeriNet', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_9x100_with_positive_weights_2913_cpu.h5', 100, 105, 1, method='RobustVerifier', cifar=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'cifar10_ffnn_6x100_with_positive_weights_3638_cpu.h5', 100, 105, 1, method='NeWise', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_6x100_with_positive_weights_3638_cpu.h5', 100, 105, 1, method='DeepCert', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_6x100_with_positive_weights_3638_cpu.h5', 100, 105, 1, method='VeriNet', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_6x100_with_positive_weights_3638_cpu.h5', 100, 105, 1, method='RobustVerifier', cifar=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100_with_positive_weights_4033_cpu.h5', 100, 105, 1, method='NeWise', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100_with_positive_weights_4033_cpu.h5', 100, 105, 1, method='DeepCert', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100_with_positive_weights_4033_cpu.h5', 100, 105, 1, method='VeriNet', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100_with_positive_weights_4033_cpu.h5', 100, 105, 1, method='RobustVerifier', cifar=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x50_with_positive_weights_4056_cpu.h5', 100, 105, 1, method='NeWise', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x50_with_positive_weights_4056_cpu.h5', 100, 105, 1, method='DeepCert', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x50_with_positive_weights_4056_cpu.h5', 100, 105, 1, method='VeriNet', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x50_with_positive_weights_4056_cpu.h5', 100, 105, 1, method='RobustVerifier', cifar=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'cifar10_ffnn_4x100_with_positive_weights_3988_cpu.h5', 100, 105, 1, method='NeWise', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_4x100_with_positive_weights_3988_cpu.h5', 100, 105, 1, method='DeepCert', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_4x100_with_positive_weights_3988_cpu.h5', 100, 105, 1, method='VeriNet', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_4x100_with_positive_weights_3988_cpu.h5', 100, 105, 1, method='RobustVerifier', cifar=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_4_3_with_positive_weights_3597.h5', 100, 105, 1, method='NeWise', cifar=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_4_3_with_positive_weights_3597.h5', 100, 105, 1, method='DeepCert', cifar=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_4_3_with_positive_weights_3597.h5', 100, 105, 1, method='VeriNet', cifar=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_4_3_with_positive_weights_3597.h5', 100, 105, 1, method='RobustVerifier', cifar=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3_with_positive_weights_3812.h5', 100, 105, 1, method='NeWise', cifar=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3_with_positive_weights_3812.h5', 100, 105, 1, method='DeepCert', cifar=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3_with_positive_weights_3812.h5', 100, 105, 1, method='VeriNet', cifar=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3_with_positive_weights_3812.h5', 100, 105, 1, method='RobustVerifier', cifar=True, log_name=log_name)
    
    printlog('generat the log of table 2 finished!!!', log_name)