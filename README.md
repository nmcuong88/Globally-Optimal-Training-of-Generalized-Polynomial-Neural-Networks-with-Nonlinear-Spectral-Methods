## Optimal Training of Polynomial Nets with Nonlinear Spectral Methods
This contains Matlab code for our Nonlinear Spectral Methods presented at NIPS 2016.

For citation of our paper  
@inproceedings{AQM2016,  
	title={Globally Optimal Training of Generalized Polynomial Neural Networks with Nonlinear Spectral Methods},  
	author={A. Gautier and Q. Nguyen and M. Hein},  
	booktitle={Advances in Neural Information Processing Systems (NIPS)},  
	year={2016}  
}  

The following version presents our general theory for a certain class of non-convex optimization problems:  
@inproceedings{QAM2016,  
	title={Nonlinear Spectral Methods for Nonconvex Optimization with Global Optimality},  
	author={Q. Nguyen and A. Gautier and M. Hein},  
	booktitle={NIPS Workshop on Optimization for Machine Learning},  
	year={2016}  
}  

Installation:  
	Our code require cvx which can be obtained from: http://cvxr.com/cvx/download/

Guideline:  
Please see the following files to run our experiments  
	1. main_NLSM.m: testing our Nonlinear Spectral Methods  
	2. main_ReLU1.m: testing one-hidden-layer ReLU nets by Batch-SGD  
	3. main_ReLU2.m: testing two-hidden-layer ReLU nets by Batch-SGD  

In all experiments, we use UCI-datasets obtained from:  
https://archive.ics.uci.edu/ml/datasets.html  

