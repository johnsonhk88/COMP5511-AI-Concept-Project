python -m cProfile 401_CNN.py > 401_CNN-CPU.pstat
RMDIR /Q/S model
mkdir model
mprof run 401_CNN.py
move mprofile*.dat 401_CNN-CPU.dat
RMDIR /Q/S model
mkdir model

python -m cProfile 402_RNN_classification.py > 402_RNN_classification-CPU.pstat
RMDIR /Q/S model
mkdir model
mprof run 402_RNN_classification.py
move mprofile*.dat 402_RNN_classification-CPU.dat
RMDIR /Q/S model
mkdir model