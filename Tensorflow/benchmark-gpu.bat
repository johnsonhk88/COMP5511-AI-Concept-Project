python -m cProfile 401_CNN.py > 401_CNN-GPU.pstat
RMDIR /Q/S model
mkdir model
mprof run 401_CNN.py
move mprofile*.dat 401_CNN-GPU.dat
RMDIR /Q/S model
mkdir model

python -m cProfile 402_RNN_classification.py > 402_RNN_classification-GPU.pstat
RMDIR /Q/S model
mkdir model
mprof run 402_RNN_classification.py
move mprofile*.dat 402_RNN_classification-GPU.dat
RMDIR /Q/S model
mkdir model