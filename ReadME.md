This repository contain the code for matrix multoplication using tiling.
The python file is used to verify the result. Flops for the tiling algorithm
and numpy will be printed out after running the bash script.

Next step:
	Add in multithreading for gemm.cpp. Numpy uses multiply core to multiply
	a matrix but for comparison purpose numpy is limited to just using a single
	core. Following work will focus on running the tiling algorithm on all the
	cores.



Machine Info:
	CPU: Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz 
