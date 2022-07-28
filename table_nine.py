import numpy as np
from cnn_bounds_full_core_with_LP import run_certified_bounds_core
from cnn_bounds_full_with_LP import run_certified_bounds

import time as timing
import datetime

def printlog(s, log_name):
    print(s, file=open("logs/"+log_name+".txt", "a"))

if __name__ == '__main__':
    
    path_prefix = "models/one_layer_models/"
    log_name = "table_nine"

    # CNN
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_1_5_sigmoid_local.h5', 10, 105, 1, method='NeWise', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_1_5_sigmoid_local.h5', 10, 105, 1, method='DeepCert', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_1_5_sigmoid_local.h5', 10, 105, 1, method='VeriNet', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_1_5_sigmoid_local.h5', 10, 105, 1, method='RobustVerifier', mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_1_5_tanh_local.h5', 10, 105, 1, method='NeWise', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_1_5_tanh_local.h5', 10, 105, 1, method='DeepCert', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_1_5_tanh_local.h5', 10, 105, 1, method='VeriNet', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_1_5_tanh_local.h5', 10, 105, 1, method='RobustVerifier', activation='tanh', mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_2_5_sigmoid_local.h5', 100, 105, 1, method='NeWise', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_2_5_sigmoid_local.h5', 100, 105, 1, method='DeepCert', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_2_5_sigmoid_local.h5', 100, 105, 1, method='VeriNet', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_2_5_sigmoid_local.h5', 100, 105, 1, method='RobustVerifier', mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_2_5_tanh_local.h5', 100, 105, 1, method='NeWise', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_2_5_tanh_local.h5', 100, 105, 1, method='DeepCert', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_2_5_tanh_local.h5', 100, 105, 1, method='VeriNet', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_2_5_tanh_local.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_3_5_sigmoid_local.h5', 100, 105, 1, method='NeWise', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_3_5_sigmoid_local.h5', 100, 105, 1, method='DeepCert', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_3_5_sigmoid_local.h5', 100, 105, 1, method='VeriNet', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_3_5_sigmoid_local.h5', 100, 105, 1, method='RobustVerifier', mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_3_5_tanh_local.h5', 100, 105, 1, method='NeWise', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_3_5_tanh_local.h5', 100, 105, 1, method='DeepCert', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_3_5_tanh_local.h5', 100, 105, 1, method='VeriNet', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_3_5_tanh_local.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_4_5_sigmoid_local.h5', 10, 105, 1, method='NeWise', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_4_5_sigmoid_local.h5', 10, 105, 1, method='DeepCert', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_4_5_sigmoid_local.h5', 10, 105, 1, method='VeriNet', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_4_5_sigmoid_local.h5', 10, 105, 1, method='RobustVerifier', mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_4_5_tanh_local.h5', 10, 105, 1, method='NeWise', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_4_5_tanh_local.h5', 10, 105, 1, method='DeepCert', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_4_5_tanh_local.h5', 10, 105, 1, method='VeriNet', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_4_5_tanh_local.h5', 10, 105, 1, method='RobustVerifier', activation='tanh', mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_5_3_sigmoid.h5', 100, 105, 1, method='NeWise', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_5_3_sigmoid.h5', 100, 105, 1, method='DeepCert', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_5_3_sigmoid.h5', 100, 105, 1, method='VeriNet', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_5_3_sigmoid.h5', 100, 105, 1, method='RobustVerifier', mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_5_3_tanh.h5', 100, 105, 1, method='NeWise', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_5_3_tanh.h5', 100, 105, 1, method='DeepCert', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_5_3_tanh.h5', 100, 105, 1, method='VeriNet', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_5_3_tanh.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_1_5_sigmoid_local.h5', 100, 105, 1, method='NeWise', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_1_5_sigmoid_local.h5', 100, 105, 1, method='DeepCert', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_1_5_sigmoid_local.h5', 100, 105, 1, method='VeriNet', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_1_5_sigmoid_local.h5', 100, 105, 1, method='RobustVerifier', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_1_5_tanh_local.h5', 100, 105, 1, method='NeWise', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_1_5_tanh_local.h5', 100, 105, 1, method='DeepCert', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_1_5_tanh_local.h5', 100, 105, 1, method='VeriNet', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_1_5_tanh_local.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', fashion_mnist=True, log_name=log_name)

    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_2_5_sigmoid_local.h5', 100, 105, 1, method='NeWise', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_2_5_sigmoid_local.h5', 100, 105, 1, method='DeepCert', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_2_5_sigmoid_local.h5', 100, 105, 1, method='VeriNet', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_2_5_sigmoid_local.h5', 100, 105, 1, method='RobustVerifier', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_2_5_tanh_local.h5', 100, 105, 1, method='NeWise', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_2_5_tanh_local.h5', 100, 105, 1, method='DeepCert', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_2_5_tanh_local.h5', 100, 105, 1, method='VeriNet', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_2_5_tanh_local.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_3_5_sigmoid_local.h5', 100, 105, 1, method='NeWise', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_3_5_sigmoid_local.h5', 100, 105, 1, method='DeepCert', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_3_5_sigmoid_local.h5', 100, 105, 1, method='VeriNet', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_3_5_sigmoid_local.h5', 100, 105, 1, method='RobustVerifier', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_3_5_tanh_local.h5', 100, 105, 1, method='NeWise', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_3_5_tanh_local.h5', 100, 105, 1, method='DeepCert', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_3_5_tanh_local.h5', 100, 105, 1, method='VeriNet', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_3_5_tanh_local.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_4_5_sigmoid_local.h5', 100, 105, 1, method='NeWise', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_4_5_sigmoid_local.h5', 100, 105, 1, method='DeepCert', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_4_5_sigmoid_local.h5', 100, 105, 1, method='VeriNet', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_4_5_sigmoid_local.h5', 100, 105, 1, method='RobustVerifier', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_4_5_tanh_local.h5', 100, 105, 1, method='NeWise', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_4_5_tanh_local.h5', 100, 105, 1, method='DeepCert', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_4_5_tanh_local.h5', 100, 105, 1, method='VeriNet', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_4_5_tanh_local.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_5_3_sigmoid.h5', 100, 105, 1, method='NeWise', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_5_3_sigmoid.h5', 100, 105, 1, method='DeepCert', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_5_3_sigmoid.h5', 100, 105, 1, method='VeriNet', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_5_3_sigmoid.h5', 100, 105, 1, method='RobustVerifier', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_5_3_tanh.h5', 100, 105, 1, method='NeWise', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_5_3_tanh.h5', 100, 105, 1, method='DeepCert', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_5_3_tanh.h5', 100, 105, 1, method='VeriNet', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_5_3_tanh.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', fashion_mnist=True, log_name=log_name)
    
    # FNN
    run_certified_bounds(path_prefix + 'mnist_fnn_1x50_sigmoid_local.h5', 100, 105, 1, method='NeWise', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x50_sigmoid_local.h5', 100, 105, 1, method='DeepCert', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x50_sigmoid_local.h5', 100, 105, 1, method='VeriNet', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x50_sigmoid_local.h5', 100, 105, 1, method='RobustVerifier', mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'mnist_fnn_1x50_tanh_local.h5', 100, 105, 1, method='NeWise', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x50_tanh_local.h5', 100, 105, 1, method='DeepCert', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x50_tanh_local.h5', 100, 105, 1, method='VeriNet', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x50_tanh_local.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'mnist_fnn_1x100_sigmoid_local.h5', 100, 105, 1, method='NeWise', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x100_sigmoid_local.h5', 100, 105, 1, method='DeepCert', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x100_sigmoid_local.h5', 100, 105, 1, method='VeriNet', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x100_sigmoid_local.h5', 100, 105, 1, method='RobustVerifier', mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'mnist_fnn_1x100_tanh_local.h5', 100, 105, 1, method='NeWise', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x100_tanh_local.h5', 100, 105, 1, method='DeepCert', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x100_tanh_local.h5', 100, 105, 1, method='VeriNet', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x100_tanh_local.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'mnist_fnn_1x150_sigmoid_local.h5', 100, 105, 1, method='NeWise', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x150_sigmoid_local.h5', 100, 105, 1, method='DeepCert', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x150_sigmoid_local.h5', 100, 105, 1, method='VeriNet', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x150_sigmoid_local.h5', 100, 105, 1, method='RobustVerifier', mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'mnist_fnn_1x150_tanh_local.h5', 100, 105, 1, method='NeWise', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x150_tanh_local.h5', 100, 105, 1, method='DeepCert', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x150_tanh_local.h5', 100, 105, 1, method='VeriNet', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x150_tanh_local.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'mnist_fnn_1x200_sigmoid_local.h5', 10, 105, 1, method='NeWise', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x200_sigmoid_local.h5', 10, 105, 1, method='DeepCert', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x200_sigmoid_local.h5', 10, 105, 1, method='VeriNet', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x200_sigmoid_local.h5', 10, 105, 1, method='RobustVerifier', mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'mnist_fnn_1x200_tanh_local.h5', 10, 105, 1, method='NeWise', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x200_tanh_local.h5', 10, 105, 1, method='DeepCert', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x200_tanh_local.h5', 10, 105, 1, method='VeriNet', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x200_tanh_local.h5', 10, 105, 1, method='RobustVerifier', activation='tanh', mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'mnist_fnn_1x250_sigmoid_local.h5', 10, 105, 1, method='NeWise', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x250_sigmoid_local.h5', 10, 105, 1, method='DeepCert', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x250_sigmoid_local.h5', 10, 105, 1, method='VeriNet', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x250_sigmoid_local.h5', 10, 105, 1, method='RobustVerifier', mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'mnist_fnn_1x250_tanh_local.h5', 10, 105, 1, method='NeWise', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x250_tanh_local.h5', 10, 105, 1, method='DeepCert', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x250_tanh_local.h5', 10, 105, 1, method='VeriNet', activation='tanh', mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x250_tanh_local.h5', 10, 105, 1, method='RobustVerifier', activation='tanh', mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x50_sigmoid_local.h5', 100, 105, 1, method='NeWise', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x50_sigmoid_local.h5', 100, 105, 1, method='DeepCert', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x50_sigmoid_local.h5', 100, 105, 1, method='VeriNet', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x50_sigmoid_local.h5', 100, 105, 1, method='RobustVerifier', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x50_tanh_local.h5', 100, 105, 1, method='NeWise', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x50_tanh_local.h5', 100, 105, 1, method='DeepCert', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x50_tanh_local.h5', 100, 105, 1, method='VeriNet', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x50_tanh_local.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x100_sigmoid_local.h5', 100, 105, 1, method='NeWise', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x100_sigmoid_local.h5', 100, 105, 1, method='DeepCert', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x100_sigmoid_local.h5', 100, 105, 1, method='VeriNet', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x100_sigmoid_local.h5', 100, 105, 1, method='RobustVerifier', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x100_tanh_local.h5', 100, 105, 1, method='NeWise', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x100_tanh_local.h5', 100, 105, 1, method='DeepCert', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x100_tanh_local.h5', 100, 105, 1, method='VeriNet', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x100_tanh_local.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x150_sigmoid_local.h5', 100, 105, 1, method='NeWise', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x150_sigmoid_local.h5', 100, 105, 1, method='DeepCert', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x150_sigmoid_local.h5', 100, 105, 1, method='VeriNet', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x150_sigmoid_local.h5', 100, 105, 1, method='RobustVerifier', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x150_tanh_local.h5', 100, 105, 1, method='NeWise', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x150_tanh_local.h5', 100, 105, 1, method='DeepCert', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x150_tanh_local.h5', 100, 105, 1, method='VeriNet', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x150_tanh_local.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x200_sigmoid_local.h5', 100, 105, 1, method='NeWise', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x200_sigmoid_local.h5', 100, 105, 1, method='DeepCert', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x200_sigmoid_local.h5', 100, 105, 1, method='VeriNet', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x200_sigmoid_local.h5', 100, 105, 1, method='RobustVerifier', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x200_tanh_local.h5', 100, 105, 1, method='NeWise', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x200_tanh_local.h5', 100, 105, 1, method='DeepCert', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x200_tanh_local.h5', 100, 105, 1, method='VeriNet', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x200_tanh_local.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x250_sigmoid_local.h5', 100, 105, 1, method='NeWise', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x250_sigmoid_local.h5', 100, 105, 1, method='DeepCert', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x250_sigmoid_local.h5', 100, 105, 1, method='VeriNet', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x250_sigmoid_local.h5', 100, 105, 1, method='RobustVerifier', fashion_mnist=True, log_name=log_name)
    
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x250_tanh_local.h5', 100, 105, 1, method='NeWise', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x250_tanh_local.h5', 100, 105, 1, method='DeepCert', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x250_tanh_local.h5', 100, 105, 1, method='VeriNet', activation='tanh', fashion_mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x250_tanh_local.h5', 100, 105, 1, method='RobustVerifier', activation='tanh', fashion_mnist=True, log_name=log_name)
    
    printlog('generat the log of table 9 finished!!!', log_name)
    