import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
from figure_utils import *

if __name__ == '__main__':
    filePath = 'logs/verinet_than_deepcert.txt'
    save_name = 'figs/verinet_than_deepcert_'
    neuron = 200
    deepcert_hidden_layer_lb, deepcert_hidden_layer_ub, deepcert_output_layer_lb, deepcert_output_layer_ub, verinet_hidden_layer_lb, verinet_hidden_layer_ub, verinet_output_layer_lb, verinet_output_layer_ub = loadData(filePath, neuron)
    
    x = list(range(1, neuron+1))
    deepcert_interval = np.subtract(deepcert_hidden_layer_ub[2], deepcert_hidden_layer_lb[2])
    verinet_interval = np.subtract(verinet_hidden_layer_ub[2], verinet_hidden_layer_lb[2])
    interval = np.subtract(verinet_interval, deepcert_interval)
    plt.scatter(x, np.divide(interval, deepcert_interval), color="red", marker='o',  s=40)
    plt.scatter(x, np.divide(np.subtract(verinet_hidden_layer_lb[2], deepcert_hidden_layer_lb[2]), deepcert_hidden_layer_lb[2]),color="blue", marker='o',  s=40)
    plt.scatter(x, np.divide(np.subtract(verinet_hidden_layer_ub[2], deepcert_hidden_layer_ub[2]), deepcert_hidden_layer_ub[2]), color="green", marker='o',  s=40)
    plt.xticks(size=20)
    plt.yticks(size=20)
    x_major_locator=MultipleLocator(50)
    y_major_locator=MultipleLocator(0.4)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.savefig(save_name+"3_layer.pdf", dpi=500, bbox_inches='tight')

    # plt.show()
    print('generate figures 4(b) finished!')