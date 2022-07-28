# NeWise
A prototype verification tool.
## Installation
***
We first provide scripts that will install all the necessary dependencies.
```
. install.sh
```

The dependency also can be installed step by step as follows (sudo rights might be required):

Install dependencies:
```
sudo apt update
sudo apt upgrade -y
sudo apt install build-essential zlib1g-dev libbz2-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev
sudo apt-get install -y libgl1-mesa-dev
```
Install python 3.7.5
```
wget https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz
tar -xzvf Python-3.7.5.tgz
cd Python-3.7.5
./configure --prefix=/usr/local/src/python37
sudo make
sudo make install
sudo ln -s /usr/local/src/python37/bin/python3.7 /usr/bin/python3.7
sudo ln -s /usr/local/src/python37/bin/pip3.7 /usr/bin/pip3.7
```
Install virtualenv and enter virtualenv:
```
sudo apt-get install python-virtualenv
virtualenv -p python3.7 venv
source venv/bin/activate
```

Install the remaining python dependencies (such as numpy and tensorflow), type:
```
pip install -r requirements.txt
```
Modify one file of tensorflow package:
```
python modify_file.py
```

## How to Run
***

```
python main.py 
```
or
```
. run.sh
```

Results will be saved in 'logs/'. The result of FNNs will be saved in 'logs/cnn_bounds_full_with_LP_xxx.txt', and that of CNNs will be saved in 'logs/cnn_bounds_full_core_with_LP_xxx.txt'.

Note that we just submit some models due to the limit of supplementary material. All the pre-trained models used in the paper can be downloaded from <https://drive.google.com/drive/folders/1Fa3ASB7uHwKll76AuwPComoCLldx0YqR?usp=sharing>.