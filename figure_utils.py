import numpy as np

def loadData(filePath, neuron):
    fr = open(filePath, 'r+')
    lines = fr.readlines()
    length = len(lines)
    
    deepcert_hidden_layer_lb = np.zeros((6,neuron), dtype = np.float32)
    deepcert_hidden_layer_ub = np.zeros((6,neuron), dtype = np.float32)
    deepcert_output_layer_lb = np.zeros((1,10), dtype = np.float32)
    deepcert_output_layer_ub = np.zeros((1,10), dtype = np.float32)
    
    verinet_hidden_layer_lb = np.zeros((6,neuron), dtype = np.float32)
    verinet_hidden_layer_ub = np.zeros((6,neuron), dtype = np.float32)
    verinet_output_layer_lb = np.zeros((1,10), dtype = np.float32)
    verinet_output_layer_ub = np.zeros((1,10), dtype = np.float32)
    
    count = 0
    node_count = 0
    flag_1_layer = False
    flag_2_layer = False
    flag_3_layer = False
    flag_4_layer = False
    flag_5_layer = False
    flag_6_layer = False
    flag_7_layer = False
    for line in lines:
        if "Overapproximation" in line:
            count += 1
            node_count = 0
        elif "1 layer" in line and "11 layer" not in line:
            flag_1_layer = True
            flag_2_layer = False
            flag_3_layer = False
            flag_4_layer = False
            flag_5_layer = False
            flag_6_layer = False
            flag_7_layer = False
        elif "2 layer" in line and "12 layer" not in line:
            flag_1_layer = False
        elif "3 layer" in line and "13 layer" not in line:
            flag_1_layer = False
            flag_2_layer = True
            flag_3_layer = False
            flag_4_layer = False
            flag_5_layer = False
            flag_6_layer = False
            flag_7_layer = False
        elif "4 layer" in line:
            flag_2_layer = False
        elif "5 layer" in line:
            flag_1_layer = False
            flag_2_layer = False
            flag_3_layer = True
            flag_4_layer = False
            flag_5_layer = False
            flag_6_layer = False
            flag_7_layer = False
        elif "6 layer" in line:
            flag_3_layer = False
        elif "7 layer" in line:
            flag_1_layer = False
            flag_2_layer = False
            flag_3_layer = False
            flag_4_layer = True
            flag_5_layer = False
            flag_6_layer = False
            flag_7_layer = False
        elif "8 layer" in line:
            flag_4_layer = False
        elif "9 layer" in line:
            flag_1_layer = False
            flag_2_layer = False
            flag_3_layer = False
            flag_4_layer = False
            flag_5_layer = True
            flag_6_layer = False
            flag_7_layer = False
        elif "10 layer" in line:
            flag_5_layer = False
        elif "11 layer" in line:
            flag_1_layer = False
            flag_2_layer = False
            flag_3_layer = False
            flag_4_layer = False
            flag_5_layer = False
            flag_6_layer = True
            flag_7_layer = False
        elif "12 layer" in line:
            flag_6_layer = False
        elif "13 layer" in line:
            flag_1_layer = False
            flag_2_layer = False
            flag_3_layer = False
            flag_4_layer = False
            flag_5_layer = False
            flag_6_layer = False
            flag_7_layer = True
        elif "[L0]" in line:
            flag_1_layer = False
            flag_2_layer = False
            flag_3_layer = False
            flag_4_layer = False
            flag_5_layer = False
            flag_6_layer = False
            flag_7_layer = False
        else:   
            if flag_1_layer:
                if count == 1:
                    lb, ub = line.split(',')
                    lb, ub = float(lb), float(ub)
                    deepcert_hidden_layer_lb[0][node_count%neuron] = lb
                    deepcert_hidden_layer_ub[0][node_count%neuron] = ub
                    node_count += 1
                elif count == 2:
                    lb, ub = line.split(',')
                    lb, ub = float(lb), float(ub)
                    verinet_hidden_layer_lb[0][node_count%neuron] = lb
                    verinet_hidden_layer_ub[0][node_count%neuron] = ub
                    node_count += 1
            elif flag_2_layer:
                if count == 1:
                    lb, ub = line.split(',')
                    lb, ub = float(lb), float(ub)
                    deepcert_hidden_layer_lb[1][node_count%neuron] = lb
                    deepcert_hidden_layer_ub[1][node_count%neuron] = ub
                    node_count += 1
                elif count == 2:
                    lb, ub = line.split(',')
                    lb, ub = float(lb), float(ub)
                    verinet_hidden_layer_lb[1][node_count%neuron] = lb
                    verinet_hidden_layer_ub[1][node_count%neuron] = ub
                    node_count += 1
            elif flag_3_layer:
                if count == 1:
                    lb, ub = line.split(',')
                    lb, ub = float(lb), float(ub)
                    deepcert_hidden_layer_lb[2][node_count%neuron] = lb
                    deepcert_hidden_layer_ub[2][node_count%neuron] = ub
                    node_count += 1
                elif count == 2:
                    lb, ub = line.split(',')
                    lb, ub = float(lb), float(ub)
                    verinet_hidden_layer_lb[2][node_count%neuron] = lb
                    verinet_hidden_layer_ub[2][node_count%neuron] = ub
                    node_count += 1
            elif flag_4_layer:
                if count == 1:
                    lb, ub = line.split(',')
                    lb, ub = float(lb), float(ub)
                    deepcert_hidden_layer_lb[3][node_count%neuron] = lb
                    deepcert_hidden_layer_ub[3][node_count%neuron] = ub
                    node_count += 1
                elif count == 2:
                    lb, ub = line.split(',')
                    lb, ub = float(lb), float(ub)
                    verinet_hidden_layer_lb[3][node_count%neuron] = lb
                    verinet_hidden_layer_ub[3][node_count%neuron] = ub
                    node_count += 1
            elif flag_5_layer:
                if count == 1:
                    lb, ub = line.split(',')
                    lb, ub = float(lb), float(ub)
                    deepcert_hidden_layer_lb[4][node_count%neuron] = lb
                    deepcert_hidden_layer_ub[4][node_count%neuron] = ub
                    node_count += 1
                elif count == 2:
                    lb, ub = line.split(',')
                    lb, ub = float(lb), float(ub)
                    verinet_hidden_layer_lb[4][node_count%neuron] = lb
                    verinet_hidden_layer_ub[4][node_count%neuron] = ub
                    node_count += 1
            elif flag_6_layer:
                if count == 1:
                    lb, ub = line.split(',')
                    lb, ub = float(lb), float(ub)
                    deepcert_hidden_layer_lb[5][node_count%neuron] = lb
                    deepcert_hidden_layer_ub[5][node_count%neuron] = ub
                    node_count += 1
                elif count == 2:
                    lb, ub = line.split(',')
                    lb, ub = float(lb), float(ub)
                    verinet_hidden_layer_lb[5][node_count%neuron] = lb
                    verinet_hidden_layer_ub[5][node_count%neuron] = ub
                    node_count += 1
            elif flag_7_layer:
                if count == 1:
                    lb, ub = line.split(',')
                    lb, ub = float(lb), float(ub)
                    deepcert_output_layer_lb[0][node_count%neuron] = lb
                    deepcert_output_layer_ub[0][node_count%neuron] = ub
                    node_count += 1
                elif count == 2:
                    lb, ub = line.split(',')
                    lb, ub = float(lb), float(ub)
                    verinet_output_layer_lb[0][node_count%neuron] = lb
                    verinet_output_layer_ub[0][node_count%neuron] = ub
                    node_count += 1
        
        
        
        
    return deepcert_hidden_layer_lb, deepcert_hidden_layer_ub, deepcert_output_layer_lb, deepcert_output_layer_ub, verinet_hidden_layer_lb, verinet_hidden_layer_ub, verinet_output_layer_lb, verinet_output_layer_ub
