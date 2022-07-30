#!/bin/bash

mkdir logs
mkdir results
mkdir figs

# NW DC VN RV 
python table_one.py
python table_two.py
python table_three.py
python table_four.py
python table_five.py
python table_six.py
python table_seven.py
python table_eight.py
python table_nine.py

# Alg1. Table 4
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_4 --batch_size 100 --model_dir models/one_layer_models/ --model_name mnist_cnn_as_mlp_2layer_1_5_sigmoid_local.pth --num_neurons 576 --num_layers 2 --activation sigmoid --dataset mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_4 --batch_size 100 --model_dir models/one_layer_models/ --model_name mnist_cnn_as_mlp_2layer_2_5_sigmoid_local.pth --num_neurons 1152 --num_layers 2 --activation sigmoid --dataset mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_4 --batch_size 100 --model_dir models/one_layer_models/ --model_name mnist_cnn_as_mlp_2layer_3_5_sigmoid_local.pth --num_neurons 1728 --num_layers 2 --activation sigmoid --dataset mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_4 --batch_size 100 --model_dir models/one_layer_models/ --model_name mnist_cnn_as_mlp_2layer_4_5_sigmoid_local.pth --num_neurons 2304 --num_layers 2 --activation sigmoid --dataset mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_4 --batch_size 100 --model_dir models/one_layer_models/ --model_name mnist_cnn_as_mlp_2layer_5_3_sigmoid.pth --num_neurons 3380 --num_layers 2 --activation sigmoid --dataset mnist --neuronwise_optimize

python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_4 --batch_size 100 --model_dir models/one_layer_models/ --model_name fashion_mnist_cnn_as_mlp_2layer_1_5_sigmoid_local.pth --num_neurons 576 --num_layers 2 --activation sigmoid --dataset fashion_mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_4 --batch_size 100 --model_dir models/one_layer_models/ --model_name fashion_mnist_cnn_as_mlp_2layer_2_5_sigmoid_local.pth --num_neurons 1152 --num_layers 2 --activation sigmoid --dataset fashion_mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_4 --batch_size 100 --model_dir models/one_layer_models/ --model_name fashion_mnist_cnn_as_mlp_2layer_3_5_sigmoid_local.pth --num_neurons 1728 --num_layers 2 --activation sigmoid --dataset fashion_mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_4 --batch_size 100 --model_dir models/one_layer_models/ --model_name fashion_mnist_cnn_as_mlp_2layer_4_5_sigmoid_local.pth --num_neurons 2304 --num_layers 2 --activation sigmoid --dataset fashion_mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_4 --batch_size 100 --model_dir models/one_layer_models/ --model_name fashion_mnist_cnn_as_mlp_2layer_5_3_sigmoid.pth --num_neurons 3380 --num_layers 2 --activation sigmoid --dataset fashion_mnist --neuronwise_optimize

python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_4 --batch_size 100 --model_dir models/one_layer_models/ --model_name mnist_fnn_1x50_sigmoid_local.pth --num_neurons 50 --num_layers 2 --activation sigmoid --dataset mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_4 --batch_size 100 --model_dir models/one_layer_models/ --model_name mnist_fnn_1x100_sigmoid_local.pth --num_neurons 100 --num_layers 2 --activation sigmoid --dataset mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_4 --batch_size 100 --model_dir models/one_layer_models/ --model_name mnist_fnn_1x150_sigmoid_local.pth --num_neurons 150 --num_layers 2 --activation sigmoid --dataset mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_4 --batch_size 100 --model_dir models/one_layer_models/ --model_name mnist_fnn_1x200_sigmoid_local.pth --num_neurons 200 --num_layers 2 --activation sigmoid --dataset mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_4 --batch_size 100 --model_dir models/one_layer_models/ --model_name mnist_fnn_1x250_sigmoid_local.pth --num_neurons 250 --num_layers 2 --activation sigmoid --dataset mnist --neuronwise_optimize

python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_4 --batch_size 100 --model_dir models/one_layer_models/ --model_name fashion_mnist_fnn_1x50_sigmoid_local.pth --num_neurons 50 --num_layers 2 --activation sigmoid --dataset fashion_mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_4 --batch_size 100 --model_dir models/one_layer_models/ --model_name fashion_mnist_fnn_1x100_sigmoid_local.pth --num_neurons 100 --num_layers 2 --activation sigmoid --dataset fashion_mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_4 --batch_size 100 --model_dir models/one_layer_models/ --model_name fashion_mnist_fnn_1x150_sigmoid_local.pth --num_neurons 150 --num_layers 2 --activation sigmoid --dataset fashion_mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_4 --batch_size 100 --model_dir models/one_layer_models/ --model_name fashion_mnist_fnn_1x200_sigmoid_local.pth --num_neurons 200 --num_layers 2 --activation sigmoid --dataset fashion_mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_4 --batch_size 100 --model_dir models/one_layer_models/ --model_name fashion_mnist_fnn_1x250_sigmoid_local.pth --num_neurons 250 --num_layers 2 --activation sigmoid --dataset fashion_mnist --neuronwise_optimize

# # Alg1. Table 9
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name mnist_cnn_as_mlp_2layer_1_5_sigmoid_local.pth --num_neurons 576 --num_layers 2 --activation sigmoid --dataset mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name mnist_cnn_as_mlp_2layer_1_5_tanh_local.pth --num_neurons 576 --num_layers 2 --activation tanh --dataset mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name mnist_cnn_as_mlp_2layer_2_5_sigmoid_local.pth --num_neurons 1152 --num_layers 2 --activation sigmoid --dataset mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name mnist_cnn_as_mlp_2layer_2_5_tanh_local.pth --num_neurons 1152 --num_layers 2 --activation tanh --dataset mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name mnist_cnn_as_mlp_2layer_3_5_sigmoid_local.pth --num_neurons 1728 --num_layers 2 --activation sigmoid --dataset mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name mnist_cnn_as_mlp_2layer_3_5_tanh_local.pth --num_neurons 1728 --num_layers 2 --activation tanh --dataset mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name mnist_cnn_as_mlp_2layer_4_5_sigmoid_local.pth --num_neurons 2304 --num_layers 2 --activation sigmoid --dataset mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name mnist_cnn_as_mlp_2layer_4_5_tanh_local.pth --num_neurons 2304 --num_layers 2 --activation tanh --dataset mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name mnist_cnn_as_mlp_2layer_5_3_sigmoid.pth --num_neurons 3380 --num_layers 2 --activation sigmoid --dataset mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name mnist_cnn_as_mlp_2layer_5_3_tanh.pth --num_neurons 3380 --num_layers 2 --activation tanh --dataset mnist --neuronwise_optimize

python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name fashion_mnist_cnn_as_mlp_2layer_1_5_sigmoid_local.pth --num_neurons 576 --num_layers 2 --activation sigmoid --dataset fashion_mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name fashion_mnist_cnn_as_mlp_2layer_1_5_tanh_local.pth --num_neurons 576 --num_layers 2 --activation tanh --dataset fashion_mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name fashion_mnist_cnn_as_mlp_2layer_2_5_sigmoid_local.pth --num_neurons 1152 --num_layers 2 --activation sigmoid --dataset fashion_mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name fashion_mnist_cnn_as_mlp_2layer_2_5_tanh_local.pth --num_neurons 1152 --num_layers 2 --activation tanh --dataset fashion_mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name fashion_mnist_cnn_as_mlp_2layer_3_5_sigmoid_local.pth --num_neurons 1728 --num_layers 2 --activation sigmoid --dataset fashion_mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name fashion_mnist_cnn_as_mlp_2layer_3_5_tanh_local.pth --num_neurons 1728 --num_layers 2 --activation tanh --dataset fashion_mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name fashion_mnist_cnn_as_mlp_2layer_4_5_sigmoid_local.pth --num_neurons 2304 --num_layers 2 --activation sigmoid --dataset fashion_mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name fashion_mnist_cnn_as_mlp_2layer_4_5_tanh_local.pth --num_neurons 2304 --num_layers 2 --activation tanh --dataset fashion_mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name fashion_mnist_cnn_as_mlp_2layer_5_3_sigmoid.pth --num_neurons 3380 --num_layers 2 --activation sigmoid --dataset fashion_mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name fashion_mnist_cnn_as_mlp_2layer_5_3_tanh.pth --num_neurons 3380 --num_layers 2 --activation tanh --dataset fashion_mnist --neuronwise_optimize

python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name mnist_fnn_1x50_sigmoid_local.pth --num_neurons 50 --num_layers 2 --activation sigmoid --dataset mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name mnist_fnn_1x50_tanh_local.pth --num_neurons 50 --num_layers 2 --activation tanh --dataset mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name mnist_fnn_1x100_sigmoid_local.pth --num_neurons 100 --num_layers 2 --activation sigmoid --dataset mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name mnist_fnn_1x100_tanh_local.pth --num_neurons 100 --num_layers 2 --activation tanh --dataset mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name mnist_fnn_1x150_sigmoid_local.pth --num_neurons 150 --num_layers 2 --activation sigmoid --dataset mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name mnist_fnn_1x150_tanh_local.pth --num_neurons 150 --num_layers 2 --activation tanh --dataset mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name mnist_fnn_1x200_sigmoid_local.pth --num_neurons 200 --num_layers 2 --activation sigmoid --dataset mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name mnist_fnn_1x200_tanh_local.pth --num_neurons 200 --num_layers 2 --activation tanh --dataset mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name mnist_fnn_1x250_sigmoid_local.pth --num_neurons 250 --num_layers 2 --activation sigmoid --dataset mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name mnist_fnn_1x250_tanh_local.pth --num_neurons 250 --num_layers 2 --activation tanh --dataset mnist --neuronwise_optimize

python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name fashion_mnist_fnn_1x50_sigmoid_local.pth --num_neurons 50 --num_layers 2 --activation sigmoid --dataset fashion_mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name fashion_mnist_fnn_1x50_tanh_local.pth --num_neurons 50 --num_layers 2 --activation tanh --dataset fashion_mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name fashion_mnist_fnn_1x100_sigmoid_local.pth --num_neurons 100 --num_layers 2 --activation sigmoid --dataset fashion_mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name fashion_mnist_fnn_1x100_tanh_local.pth --num_neurons 100 --num_layers 2 --activation tanh --dataset fashion_mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name fashion_mnist_fnn_1x150_sigmoid_local.pth --num_neurons 150 --num_layers 2 --activation sigmoid --dataset fashion_mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name fashion_mnist_fnn_1x150_tanh_local.pth --num_neurons 150 --num_layers 2 --activation tanh --dataset fashion_mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name fashion_mnist_fnn_1x200_sigmoid_local.pth --num_neurons 200 --num_layers 2 --activation sigmoid --dataset fashion_mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name fashion_mnist_fnn_1x200_tanh_local.pth --num_neurons 200 --num_layers 2 --activation tanh --dataset fashion_mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name fashion_mnist_fnn_1x250_sigmoid_local.pth --num_neurons 250 --num_layers 2 --activation sigmoid --dataset fashion_mnist --neuronwise_optimize
python alg1/main.py --cuda -1 --p_norm 200 --eps0 0.2 --acc 0.01 --log_name table_9 --batch_size 100 --model_dir models/one_layer_models/ --model_name fashion_mnist_fnn_1x250_tanh_local.pth --num_neurons 250 --num_layers 2 --activation tanh --dataset fashion_mnist --neuronwise_optimize

# generate total table results
python table_results.py

# generate figure logs
python figure_four_logs.py

# generate figure 4
python figure_4_a.py
python figure_4_b.py
python figure_4_c.py
python figure_4_d.py
python figure_4_e.py
python figure_4_f.py
python figure_4_g.py
python figure_4_h.py
python figure_4_i.py
python figure_4_j.py
python figure_4_k.py
python figure_4_l.py