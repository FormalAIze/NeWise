import numpy as np
from output_middel_layer_data import run_certified_bounds

import time as timing
import datetime

def printlog(s, log_name):
    print(s, file=open("logs/"+log_name+".txt", "a"))

if __name__ == '__main__':
    
    path_prefix = "models/mixed_models/"
    log_name = "verinet_than_deepcert"
    
    # FNN
    run_certified_bounds(path_prefix + 'ffnnSIGMOID__Point_6x200.h5', 1, 105, 1, data_from_local=False, method='DeepCert', eran_fnn=True, mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'ffnnSIGMOID__Point_6x200.h5', 1, 105, 1, data_from_local=False, method='VeriNet', eran_fnn=True, mnist=True, log_name=log_name)
    
    printlog('generat middle interval finished!!!', log_name)
    
    log_name = "deepcert_than_verinet"
    
    run_certified_bounds(path_prefix + 'ffnnSIGMOID__Point_6_500.h5', 1, 105, 1, data_from_local=False, method='DeepCert', eran_fnn=True, mnist=True, log_name=log_name)
    run_certified_bounds(path_prefix + 'ffnnSIGMOID__Point_6_500.h5', 1, 105, 1, data_from_local=False, method='VeriNet', eran_fnn=True, mnist=True, log_name=log_name)
    
    printlog('generat middle interval finished!!!', log_name)