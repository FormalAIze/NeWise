import numpy as np
from cnn_bounds_full_core_with_LP import run_certified_bounds_core
from cnn_bounds_full_with_LP import run_certified_bounds

import time as timing
import datetime

def printlog(s, log_name):
    print(s, file=open("logs/"+log_name+".txt", "a"))

if __name__ == '__main__':
    
    path_prefix = "models/models_with_positive_weights/tanh/"
    log_name = "table_seven"
    
    # MNIST
    run_certified_bounds(path_prefix + 'mnist_ffnn_5x100_with_positive_weights_tanh_9573.h5', 100, 105, 1, method='NeWise', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_5x100_with_positive_weights_tanh_9573.h5', 100, 105, 1, method='DeepCert', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_5x100_with_positive_weights_tanh_9573.h5', 100, 105, 1, method='VeriNet', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_5x100_with_positive_weights_tanh_9573.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x700_with_positive_weights_tanh_9624.h5', 100, 105, 1, method='NeWise', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x700_with_positive_weights_tanh_9624.h5', 100, 105, 1, method='DeepCert', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x700_with_positive_weights_tanh_9624.h5', 100, 105, 1, method='VeriNet', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x700_with_positive_weights_tanh_9624.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x400_with_positive_weights_tanh_9648.h5', 100, 105, 1, method='NeWise', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x400_with_positive_weights_tanh_9648.h5', 100, 105, 1, method='DeepCert', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x400_with_positive_weights_tanh_9648.h5', 100, 105, 1, method='VeriNet', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x400_with_positive_weights_tanh_9648.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3_with_positive_weights_tanh_9449.h5', 100, 105, 1, method='NeWise', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3_with_positive_weights_tanh_9449.h5', 100, 105, 1, method='DeepCert', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3_with_positive_weights_tanh_9449.h5', 100, 105, 1, method='VeriNet', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3_with_positive_weights_tanh_9449.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x50_with_positive_weights_tanh_9546.h5', 100, 105, 1, method='NeWise', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x50_with_positive_weights_tanh_9546.h5', 100, 105, 1, method='DeepCert', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x50_with_positive_weights_tanh_9546.h5', 100, 105, 1, method='VeriNet', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x50_with_positive_weights_tanh_9546.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x100_with_positive_weights_tanh_9644.h5', 100, 105, 1, method='NeWise', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x100_with_positive_weights_tanh_9644.h5', 100, 105, 1, method='DeepCert', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x100_with_positive_weights_tanh_9644.h5', 100, 105, 1, method='VeriNet',activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x100_with_positive_weights_tanh_9644.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x200_with_positive_weights_tanh_9637.h5', 100, 105, 1, method='NeWise', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x200_with_positive_weights_tanh_9637.h5', 100, 105, 1, method='DeepCert', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x200_with_positive_weights_tanh_9637.h5', 100, 105, 1, method='VeriNet', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x200_with_positive_weights_tanh_9637.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_4_3_with_positive_weights_tanh_9612.h5', 100, 105, 1, method='NeWise', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_4_3_with_positive_weights_tanh_9612.h5', 100, 105, 1, method='DeepCert', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_4_3_with_positive_weights_tanh_9612.h5', 100, 105, 1, method='VeriNet', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_4_3_with_positive_weights_tanh_9612.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3_with_positive_weights_tanh_9457.h5', 100, 105, 1, method='NeWise', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3_with_positive_weights_tanh_9457.h5', 100, 105, 1, method='DeepCert', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3_with_positive_weights_tanh_9457.h5', 100, 105, 1, method='VeriNet', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3_with_positive_weights_tanh_9457.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_with_positive_weights_tanh_9655.h5', 100, 105, 1, method='NeWise', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_with_positive_weights_tanh_9655.h5', 100, 105, 1, method='DeepCert', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_with_positive_weights_tanh_9655.h5', 100, 105, 1, method='VeriNet', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_with_positive_weights_tanh_9655.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_2_3_with_positive_weights_tanh_9530.h5', 100, 105, 1, method='NeWise', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_2_3_with_positive_weights_tanh_9530.h5', 100, 105, 1, method='DeepCert', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_2_3_with_positive_weights_tanh_9530.h5', 100, 105, 1, method='VeriNet', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_2_3_with_positive_weights_tanh_9530.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', mnist=True, log_name=log_name)
    
    # Fashion MNIST
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_3x100_with_positive_weights_tanh.h5', 100, 105, 1, method='NeWise', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_3x100_with_positive_weights_tanh.h5', 100, 105, 1, method='DeepCert', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_3x100_with_positive_weights_tanh.h5', 100, 105, 1, method='VeriNet', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_3x100_with_positive_weights_tanh.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_2x200_with_positive_weights_tanh.h5', 100, 105, 1, method='NeWise', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_2x200_with_positive_weights_tanh.h5', 100, 105, 1, method='DeepCert', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_2x200_with_positive_weights_tanh.h5', 100, 105, 1, method='VeriNet', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_2x200_with_positive_weights_tanh.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_6layer_5_3_with_positive_weights_tanh.h5', 100, 105, 1, method='NeWise', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_6layer_5_3_with_positive_weights_tanh.h5', 100, 105, 1, method='DeepCert', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_6layer_5_3_with_positive_weights_tanh.h5', 100, 105, 1, method='VeriNet', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_6layer_5_3_with_positive_weights_tanh.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_2x100_with_positive_weights_tanh.h5', 100, 105, 1, method='NeWise', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_2x100_with_positive_weights_tanh.h5', 100, 105, 1, method='DeepCert', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_2x100_with_positive_weights_tanh.h5', 100, 105, 1, method='VeriNet', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_2x100_with_positive_weights_tanh.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_5layer_5_3_with_positive_weights_tanh.h5', 100, 105, 1, method='NeWise', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_5layer_5_3_with_positive_weights_tanh.h5', 100, 105, 1, method='DeepCert', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_5layer_5_3_with_positive_weights_tanh.h5', 100, 105, 1, method='VeriNet', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_5layer_5_3_with_positive_weights_tanh.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_4layer_5_3_with_positive_weights_tanh.h5', 100, 105, 1, method='NeWise', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_4layer_5_3_with_positive_weights_tanh.h5', 100, 105, 1, method='DeepCert', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_4layer_5_3_with_positive_weights_tanh.h5', 100, 105, 1, method='VeriNet', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_4layer_5_3_with_positive_weights_tanh.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_3layer_2_3_with_positive_weights_tanh.h5', 100, 105, 1, method='NeWise', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_3layer_2_3_with_positive_weights_tanh.h5', 100, 105, 1, method='DeepCert', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_3layer_2_3_with_positive_weights_tanh.h5', 100, 105, 1, method='VeriNet', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_3layer_2_3_with_positive_weights_tanh.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_3layer_4_3_with_positive_weights_tanh.h5', 100, 105, 1, method='NeWise', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_3layer_4_3_with_positive_weights_tanh.h5', 100, 105, 1, method='DeepCert', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_3layer_4_3_with_positive_weights_tanh.h5', 100, 105, 1, method='VeriNet', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_3layer_4_3_with_positive_weights_tanh.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_3x200_with_positive_weights_tanh.h5', 100, 105, 1, method='NeWise', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_3x200_with_positive_weights_tanh.h5', 100, 105, 1, method='DeepCert', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_3x200_with_positive_weights_tanh.h5', 100, 105, 1, method='VeriNet', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_3x200_with_positive_weights_tanh.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', fashion_mnist=True, log_name=log_name)
    
    # CIFAR10
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x200_with_positive_weights_tanh_3581.h5', 100, 105, 1, method='NeWise', activation='tanh', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x200_with_positive_weights_tanh_3581.h5', 100, 105, 1, method='DeepCert', activation='tanh', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x200_with_positive_weights_tanh_3581.h5', 100, 105, 1, method='VeriNet', activation='tanh', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x200_with_positive_weights_tanh_3581.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', cifar=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'cifar10_ffnn_4x100_with_positive_weights_tanh_3114.h5', 100, 105, 1, method='NeWise', activation='tanh', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_4x100_with_positive_weights_tanh_3114.h5', 100, 105, 1, method='DeepCert', activation='tanh', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_4x100_with_positive_weights_tanh_3114.h5', 100, 105, 1, method='VeriNet', activation='tanh', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_4x100_with_positive_weights_tanh_3114.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', cifar=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x700_with_positive_weights_tanh_2906.h5', 100, 105, 1, method='NeWise', activation='tanh', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x700_with_positive_weights_tanh_2906.h5', 100, 105, 1, method='DeepCert', activation='tanh', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x700_with_positive_weights_tanh_2906.h5', 100, 105, 1, method='VeriNet', activation='tanh', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x700_with_positive_weights_tanh_2906.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', cifar=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x400_with_positive_weights_tanh_2988.h5', 100, 105, 1, method='NeWise', activation='tanh', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x400_with_positive_weights_tanh_2988.h5', 100, 105, 1, method='DeepCert', activation='tanh', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x400_with_positive_weights_tanh_2988.h5', 100, 105, 1, method='VeriNet', activation='tanh', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x400_with_positive_weights_tanh_2988.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', cifar=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x100_with_positive_weights_tanh_3281.h5', 100, 105, 1, method='NeWise', activation='tanh', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x100_with_positive_weights_tanh_3281.h5', 100, 105, 1, method='DeepCert', activation='tanh', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x100_with_positive_weights_tanh_3281.h5', 100, 105, 1, method='VeriNet', activation='tanh', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x100_with_positive_weights_tanh_3281.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', cifar=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100_with_positive_weights_tanh_2823.h5', 100, 105, 1, method='NeWise', activation='tanh', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100_with_positive_weights_tanh_2823.h5', 100, 105, 1, method='DeepCert', activation='tanh', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100_with_positive_weights_tanh_2823.h5', 100, 105, 1, method='VeriNet', activation='tanh', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100_with_positive_weights_tanh_2823.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', cifar=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x50_with_positive_weights_tanh_38.h5', 100, 105, 1, method='NeWise', activation='tanh', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x50_with_positive_weights_tanh_38.h5', 100, 105, 1, method='DeepCert', activation='tanh', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x50_with_positive_weights_tanh_38.h5', 100, 105, 1, method='VeriNet', activation='tanh', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x50_with_positive_weights_tanh_38.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', cifar=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3_with_positive_weights_tanh_3853.h5', 100, 105, 1, method='NeWise', activation='tanh', cifar=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3_with_positive_weights_tanh_3853.h5', 100, 105, 1, method='DeepCert', activation='tanh', cifar=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3_with_positive_weights_tanh_3853.h5', 100, 105, 1, method='VeriNet', activation='tanh', cifar=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3_with_positive_weights_tanh_3853.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', cifar=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_4_3_with_positive_weights_tanh_4031.h5', 100, 105, 1, method='NeWise', activation='tanh', cifar=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_4_3_with_positive_weights_tanh_4031.h5', 100, 105, 1, method='DeepCert', activation='tanh', cifar=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_4_3_with_positive_weights_tanh_4031.h5', 100, 105, 1, method='VeriNet', activation='tanh', cifar=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_4_3_with_positive_weights_tanh_4031.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', cifar=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_5_3_with_positive_weights_tanh_4159.h5', 100, 105, 1, method='NeWise', activation='tanh', cifar=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_5_3_with_positive_weights_tanh_4159.h5', 100, 105, 1, method='DeepCert', activation='tanh', cifar=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_5_3_with_positive_weights_tanh_4159.h5', 100, 105, 1, method='VeriNet', activation='tanh', cifar=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_5_3_with_positive_weights_tanh_4159.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', cifar=True, log_name=log_name)
    
    printlog('generat the log of table 7 finished!!!', log_name)