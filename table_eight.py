import numpy as np
from cnn_bounds_full_core_with_LP import run_certified_bounds_core
from cnn_bounds_full_with_LP import run_certified_bounds

import time as timing
import datetime

def printlog(s, log_name):
    print(s, file=open("logs/"+log_name+".txt", "a"))

if __name__ == '__main__':

    path_prefix = "models/models_with_positive_weights/arctan/"
    log_name= "table_eight"
    
    # MNIST
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3_with_positive_weights_atan_9389.h5', 100, 105, 1, method='NeWise', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3_with_positive_weights_atan_9389.h5', 100, 105, 1, method='DeepCert', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3_with_positive_weights_atan_9389.h5', 100, 105, 1, method='VeriNet', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3_with_positive_weights_atan_9389.h5', 100, 105, 1, method='RobustVerifier', activation='atan', mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'mnist_ffnn_5x100_with_positive_weights_atan_9233.h5', 100, 105, 1, method='NeWise', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_5x100_with_positive_weights_atan_9233.h5', 100, 105, 1, method='DeepCert', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_5x100_with_positive_weights_atan_9233.h5', 100, 105, 1, method='VeriNet', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_5x100_with_positive_weights_atan_9233.h5', 100, 105, 1, method='RobustVerifier', activation='atan', mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x700_with_positive_weights_atan_9650.h5', 100, 105, 1, method='NeWise', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x700_with_positive_weights_atan_9650.h5', 100, 105, 1, method='DeepCert', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x700_with_positive_weights_atan_9650.h5', 100, 105, 1, method='VeriNet', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x700_with_positive_weights_atan_9650.h5', 100, 105, 1, method='RobustVerifier', activation='atan', mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x50_with_positive_weights_atan_9327.h5', 100, 105, 1, method='NeWise', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x50_with_positive_weights_atan_9327.h5', 100, 105, 1, method='DeepCert', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x50_with_positive_weights_atan_9327.h5', 100, 105, 1, method='VeriNet', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x50_with_positive_weights_atan_9327.h5', 100, 105, 1, method='RobustVerifier', activation='atan', mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x400_with_positive_weights_atan_9706.h5', 100, 105, 1, method='NeWise', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x400_with_positive_weights_atan_9706.h5', 100, 105, 1, method='DeepCert', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x400_with_positive_weights_atan_9706.h5', 100, 105, 1, method='VeriNet', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x400_with_positive_weights_atan_9706.h5', 100, 105, 1, method='RobustVerifier', activation='atan', mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x200_with_positive_weights_atan_9700.h5', 100, 105, 1, method='NeWise', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x200_with_positive_weights_atan_9700.h5', 100, 105, 1, method='DeepCert', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x200_with_positive_weights_atan_9700.h5', 100, 105, 1, method='VeriNet', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x200_with_positive_weights_atan_9700.h5', 100, 105, 1, method='RobustVerifier', activation='atan', mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x100_with_positive_weights_atan_9662.h5', 100, 105, 1, method='NeWise', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x100_with_positive_weights_atan_9662.h5', 100, 105, 1, method='DeepCert', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x100_with_positive_weights_atan_9662.h5', 100, 105, 1, method='VeriNet', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x100_with_positive_weights_atan_9662.h5', 100, 105, 1, method='RobustVerifier', activation='atan', mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3_with_positive_weights_atan_9575.h5', 100, 105, 1, method='NeWise', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3_with_positive_weights_atan_9575.h5', 100, 105, 1, method='DeepCert', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3_with_positive_weights_atan_9575.h5', 100, 105, 1, method='VeriNet', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3_with_positive_weights_atan_9575.h5', 100, 105, 1, method='RobustVerifier', activation='atan', mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_2_3_with_positive_weights_atan_9499.h5', 100, 105, 1, method='NeWise', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_2_3_with_positive_weights_atan_9499.h5', 100, 105, 1, method='DeepCert', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_2_3_with_positive_weights_atan_9499.h5', 100, 105, 1, method='VeriNet', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_2_3_with_positive_weights_atan_9499.h5', 100, 105, 1, method='RobustVerifier', activation='atan', mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_with_positive_weights_atan_9678.h5', 100, 105, 1, method='NeWise', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_with_positive_weights_atan_9678.h5', 100, 105, 1, method='DeepCert', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_with_positive_weights_atan_9678.h5', 100, 105, 1, method='VeriNet', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_with_positive_weights_atan_9678.h5', 100, 105, 1, method='RobustVerifier', activation='atan', mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_4_3_with_positive_weights_atan_9481.h5', 100, 105, 1, method='NeWise', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_4_3_with_positive_weights_atan_9481.h5', 100, 105, 1, method='DeepCert', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_4_3_with_positive_weights_atan_9481.h5', 100, 105, 1, method='VeriNet', activation='atan', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_4_3_with_positive_weights_atan_9481.h5', 100, 105, 1, method='RobustVerifier', activation='atan', mnist=True, log_name=log_name)
    
    # Fashion MNIST
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_4x100_with_positive_weights_atan.h5', 100, 105, 1, method='NeWise', activation='atan', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_4x100_with_positive_weights_atan.h5', 100, 105, 1, method='DeepCert', activation='atan', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_4x100_with_positive_weights_atan.h5', 100, 105, 1, method='VeriNet', activation='atan', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_4x100_with_positive_weights_atan.h5', 100, 105, 1, method='RobustVerifier', activation='atan', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_3x100_with_positive_weights_atan.h5', 100, 105, 1, method='NeWise', activation='atan', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_3x100_with_positive_weights_atan.h5', 100, 105, 1, method='DeepCert', activation='atan', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_3x100_with_positive_weights_atan.h5', 100, 105, 1, method='VeriNet', activation='atan', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_3x100_with_positive_weights_atan.h5', 100, 105, 1, method='RobustVerifier', activation='atan', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_3x200_with_positive_weights_atan.h5', 100, 105, 1, method='NeWise', activation='atan', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_3x200_with_positive_weights_atan.h5', 100, 105, 1, method='DeepCert', activation='atan', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_3x200_with_positive_weights_atan.h5', 100, 105, 1, method='VeriNet', activation='atan', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_3x200_with_positive_weights_atan.h5', 100, 105, 1, method='RobustVerifier', activation='atan', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_2x100_with_positive_weights_atan.h5', 100, 105, 1, method='NeWise', activation='atan', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_2x100_with_positive_weights_atan.h5', 100, 105, 1, method='DeepCert', activation='atan', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_2x100_with_positive_weights_atan.h5', 100, 105, 1, method='VeriNet', activation='atan', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_2x100_with_positive_weights_atan.h5', 100, 105, 1, method='RobustVerifier', activation='atan', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_6layer_5_3_with_positive_weights_atan.h5', 100, 105, 1, method='NeWise', activation='atan', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_6layer_5_3_with_positive_weights_atan.h5', 100, 105, 1, method='DeepCert', activation='atan', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_6layer_5_3_with_positive_weights_atan.h5', 100, 105, 1, method='VeriNet', activation='atan', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_6layer_5_3_with_positive_weights_atan.h5', 100, 105, 1, method='RobustVerifier', activation='atan', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_2x200_with_positive_weights_atan.h5', 100, 105, 1, method='NeWise', activation='atan', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_2x200_with_positive_weights_atan.h5', 100, 105, 1, method='DeepCert', activation='atan', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_2x200_with_positive_weights_atan.h5', 100, 105, 1, method='VeriNet', activation='atan', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_2x200_with_positive_weights_atan.h5', 100, 105, 1, method='RobustVerifier', activation='atan', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_3layer_2_3_with_positive_weights_atan.h5', 100, 105, 1, method='NeWise', activation='atan', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_3layer_2_3_with_positive_weights_atan.h5', 100, 105, 1, method='DeepCert', activation='atan', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_3layer_2_3_with_positive_weights_atan.h5', 100, 105, 1, method='VeriNet', activation='atan', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_3layer_2_3_with_positive_weights_atan.h5', 100, 105, 1, method='RobustVerifier', activation='atan', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_4layer_5_3_with_positive_weights_atan.h5', 100, 105, 1, method='NeWise', activation='atan', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_4layer_5_3_with_positive_weights_atan.h5', 100, 105, 1, method='DeepCert', activation='atan', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_4layer_5_3_with_positive_weights_atan.h5', 100, 105, 1, method='VeriNet', activation='atan', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_4layer_5_3_with_positive_weights_atan.h5', 100, 105, 1, method='RobustVerifier', activation='atan', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_5layer_5_3_with_positive_weights_atan.h5', 100, 105, 1, method='NeWise', activation='atan', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_5layer_5_3_with_positive_weights_atan.h5', 100, 105, 1, method='DeepCert', activation='atan', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_5layer_5_3_with_positive_weights_atan.h5', 100, 105, 1, method='VeriNet', activation='atan', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_5layer_5_3_with_positive_weights_atan.h5', 100, 105, 1, method='RobustVerifier', activation='atan', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_3layer_4_3_with_positive_weights_atan.h5', 100, 105, 1, method='NeWise', activation='atan', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_3layer_4_3_with_positive_weights_atan.h5', 100, 105, 1, method='DeepCert', activation='atan', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_3layer_4_3_with_positive_weights_atan.h5', 100, 105, 1, method='VeriNet', activation='atan', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_3layer_4_3_with_positive_weights_atan.h5', 100, 105, 1, method='RobustVerifier', activation='atan', fashion_mnist=True, log_name=log_name)

    # CIFAR10
    run_certified_bounds(path_prefix + 'cifar10_ffnn_4x100_with_positive_weights_atan.h5', 100, 105, 1, method='NeWise', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_4x100_with_positive_weights_atan.h5', 100, 105, 1, method='DeepCert', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_4x100_with_positive_weights_atan.h5', 100, 105, 1, method='VeriNet', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_4x100_with_positive_weights_atan.h5', 100, 105, 1, method='RobustVerifier', activation='atan', cifar=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100_with_positive_weights_atan.h5', 100, 105, 1, method='NeWise', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100_with_positive_weights_atan.h5', 100, 105, 1, method='DeepCert', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100_with_positive_weights_atan.h5', 100, 105, 1, method='VeriNet', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100_with_positive_weights_atan.h5', 100, 105, 1, method='RobustVerifier', activation='atan', cifar=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x50_with_positive_weights_atan_3875.h5', 100, 105, 1, method='NeWise', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x50_with_positive_weights_atan_3875.h5', 100, 105, 1, method='DeepCert', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x50_with_positive_weights_atan_3875.h5', 100, 105, 1, method='VeriNet', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x50_with_positive_weights_atan_3875.h5', 100, 105, 1, method='RobustVerifier', activation='atan', cifar=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'cifar10_ffnn_6x100_with_positive_weights_atan.h5', 100, 105, 1, method='NeWise', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_6x100_with_positive_weights_atan.h5', 100, 105, 1, method='DeepCert', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_6x100_with_positive_weights_atan.h5', 100, 105, 1, method='VeriNet', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_6x100_with_positive_weights_atan.h5', 100, 105, 1, method='RobustVerifier', activation='atan', cifar=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x100_with_positive_weights_atan_4078.h5', 100, 105, 1, method='NeWise', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x100_with_positive_weights_atan_4078.h5', 100, 105, 1, method='DeepCert', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x100_with_positive_weights_atan_4078.h5', 100, 105, 1, method='VeriNet', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x100_with_positive_weights_atan_4078.h5', 100, 105, 1, method='RobustVerifier', activation='atan', cifar=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x200_with_positive_weights_atan_4291.h5', 100, 105, 1, method='NeWise', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x200_with_positive_weights_atan_4291.h5', 100, 105, 1, method='DeepCert', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x200_with_positive_weights_atan_4291.h5', 100, 105, 1, method='VeriNet', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x200_with_positive_weights_atan_4291.h5', 100, 105, 1, method='RobustVerifier', activation='atan', cifar=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x700_with_positive_weights_atan.h5', 100, 105, 1, method='NeWise', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x700_with_positive_weights_atan.h5', 100, 105, 1, method='DeepCert', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x700_with_positive_weights_atan.h5', 100, 105, 1, method='VeriNet', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x700_with_positive_weights_atan.h5', 100, 105, 1, method='RobustVerifier', activation='atan', cifar=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x400_with_positive_weights_atan.h5', 100, 105, 1, method='NeWise', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x400_with_positive_weights_atan.h5', 100, 105, 1, method='DeepCert', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x400_with_positive_weights_atan.h5', 100, 105, 1, method='VeriNet', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x400_with_positive_weights_atan.h5', 100, 105, 1, method='RobustVerifier', activation='atan', cifar=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3_with_positive_weights_atan_3782.h5', 100, 105, 1, method='NeWise', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3_with_positive_weights_atan_3782.h5', 100, 105, 1, method='DeepCert', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3_with_positive_weights_atan_3782.h5', 100, 105, 1, method='VeriNet', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3_with_positive_weights_atan_3782.h5', 100, 105, 1, method='RobustVerifier', activation='atan', cifar=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_4_3_with_positive_weights_atan.h5', 100, 105, 1, method='NeWise', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_4_3_with_positive_weights_atan.h5', 100, 105, 1, method='DeepCert', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_4_3_with_positive_weights_atan.h5', 100, 105, 1, method='VeriNet', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_4_3_with_positive_weights_atan.h5', 100, 105, 1, method='RobustVerifier', activation='atan', cifar=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_5_3_with_positive_weights_atan.h5', 100, 105, 1, method='NeWise', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_5_3_with_positive_weights_atan.h5', 100, 105, 1, method='DeepCert', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_5_3_with_positive_weights_atan.h5', 100, 105, 1, method='VeriNet', activation='atan', cifar=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_5_3_with_positive_weights_atan.h5', 100, 105, 1, method='RobustVerifier', activation='atan', cifar=True, log_name=log_name)
    
    printlog('generat the log of table 8 finished!!!', log_name)