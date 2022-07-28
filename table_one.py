import numpy as np
from cnn_bounds_full_core_with_LP import run_certified_bounds_core
from cnn_bounds_full_with_LP import run_certified_bounds

import time as timing
import datetime

def printlog(s, log_name):
    print(s, file=open("logs/"+log_name+".txt", "a"))

if __name__ == '__main__':
    
    path_prefix = "models/mixed_models/"
    log_name = "table_one"
    
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x50.h5', 100, 105, 1, method='NeWise', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x50.h5', 100, 105, 1, method='DeepCert', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x50.h5', 100, 105, 1, method='VeriNet', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x50.h5', 100, 105, 1, method='RobustVerifier', mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x100.h5', 100, 105, 1, method='NeWise', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x100.h5', 100, 105, 1, method='DeepCert', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x100.h5', 100, 105, 1, method='VeriNet', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x100.h5', 100, 105, 1, method='RobustVerifier', mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x200.h5', 100, 105, 1, method='NeWise', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x200.h5', 100, 105, 1, method='DeepCert', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x200.h5', 100, 105, 1, method='VeriNet', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_3x200.h5', 100, 105, 1, method='RobustVerifier', mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'mnist_ffnn_5x100.h5', 100, 105, 1, method='NeWise', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_5x100.h5', 100, 105, 1, method='DeepCert', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_5x100.h5', 100, 105, 1, method='VeriNet', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_ffnn_5x100.h5', 100, 105, 1, method='RobustVerifier', mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'ffnnSIGMOID__Point_6_500.h5', 100, 105, 1, method='NeWise', eran_fnn=True, mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'ffnnSIGMOID__Point_6_500.h5', 100, 105, 1, method='DeepCert', eran_fnn=True, mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'ffnnSIGMOID__Point_6_500.h5', 100, 105, 1, method='VeriNet', eran_fnn=True, mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'ffnnSIGMOID__Point_6_500.h5', 100, 105, 1, method='RobustVerifier', eran_fnn=True, mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_2_3.h5', 100, 105, 1, method='NeWise', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_2_3.h5', 100, 105, 1, method='DeepCert', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_2_3.h5', 100, 105, 1, method='VeriNet', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_2_3.h5', 100, 105, 1, method='RobustVerifier', mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_4_3.h5', 100, 105, 1, method='NeWise', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_4_3.h5', 100, 105, 1, method='DeepCert', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_4_3.h5', 100, 105, 1, method='VeriNet', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_3layer_4_3.h5', 100, 105, 1, method='RobustVerifier', mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_sigmoid', 100, 105, 1, method='NeWise', cnn_cert_model=True, mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_sigmoid', 100, 105, 1, method='DeepCert', cnn_cert_model=True, mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_sigmoid', 100, 105, 1, method='VeriNet', cnn_cert_model=True, mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_sigmoid', 100, 105, 1, method='RobustVerifier', cnn_cert_model=True, mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3.h5', 100, 105, 1, method='NeWise', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3.h5', 100, 105, 1, method='DeepCert', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3.h5', 100, 105, 1, method='VeriNet', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3.h5', 100, 105, 1, method='RobustVerifier', mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3.h5', 100, 105, 1, method='NeWise', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3.h5', 100, 105, 1, method='DeepCert', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3.h5', 100, 105, 1, method='VeriNet', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3.h5', 100, 105, 1, method='RobustVerifier', mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_8layer_5_3_sigmoid', 100, 105, 1, method='NeWise', cnn_cert_model=True, mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_8layer_5_3_sigmoid', 100, 105, 1, method='DeepCert', cnn_cert_model=True, mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_8layer_5_3_sigmoid', 100, 105, 1, method='VeriNet', cnn_cert_model=True, mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_8layer_5_3_sigmoid', 100, 105, 1, method='RobustVerifier', cnn_cert_model=True, mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_3x50.h5', 100, 105, 1, method='NeWise', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_3x50.h5', 100, 105, 1, method='DeepCert', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_3x50.h5', 100, 105, 1, method='VeriNet', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_3x50.h5', 100, 105, 1, method='RobustVerifier', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_5x100.h5', 100, 105, 1, method='NeWise', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_5x100.h5', 100, 105, 1, method='DeepCert', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_5x100.h5', 100, 105, 1, method='VeriNet', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_5x100.h5', 100, 105, 1, method='RobustVerifier', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_4layer_5_3.h5', 100, 105, 1, method='NeWise', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_4layer_5_3.h5', 100, 105, 1, method='DeepCert', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_4layer_5_3.h5', 100, 105, 1, method='VeriNet', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_4layer_5_3.h5', 100, 105, 1, method='RobustVerifier', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_5layer_5_3.h5', 100, 105, 1, method='NeWise', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_5layer_5_3.h5', 100, 105, 1, method='DeepCert', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_5layer_5_3.h5', 100, 105, 1, method='VeriNet', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_5layer_5_3.h5', 100, 105, 1, method='RobustVerifier', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_6layer_5_3.h5', 100, 105, 1, method='NeWise', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_6layer_5_3.h5', 100, 105, 1, method='DeepCert', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_6layer_5_3.h5', 100, 105, 1, method='VeriNet', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_6layer_5_3.h5', 100, 105, 1, method='RobustVerifier', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x50.h5', 100, 105, 1, method='NeWise', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x50.h5', 100, 105, 1, method='DeepCert', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x50.h5', 100, 105, 1, method='VeriNet', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x50.h5', 100, 105, 1, method='RobustVerifier', cifar=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100.h5', 100, 105, 1, method='NeWise', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100.h5', 100, 105, 1, method='DeepCert', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100.h5', 100, 105, 1, method='VeriNet', cifar=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100.h5', 100, 105, 1, method='RobustVerifier', cifar=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3.h5', 100, 105, 1, method='NeWise', cifar=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3.h5', 100, 105, 1, method='DeepCert', cifar=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3.h5', 100, 105, 1, method='VeriNet', cifar=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3.h5', 100, 105, 1, method='RobustVerifier', cifar=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_6layer_5_3.h5', 100, 105, 1, method='NeWise', cifar=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_6layer_5_3.h5', 100, 105, 1, method='DeepCert', cifar=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_6layer_5_3.h5', 100, 105, 1, method='VeriNet', cifar=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_6layer_5_3.h5', 100, 105, 1, method='RobustVerifier', cifar=True, log_name=log_name)
    
    printlog('generat the log of table 1 finished!!!', log_name)

    