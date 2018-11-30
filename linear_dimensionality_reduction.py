## Input format: code input_image
from tqdm import tqdm
import numpy as np 
import cv2
import sys

if __name__ == "__main__":

	Y = cv2.imread(sys.argv[1],0)
	Y = cv2.resize(Y,(64*4,64*4))

	D = Y.shape[0]
	N = Y.shape[1]
	M = int(64*4)

	MAXITER = 20 #How many times will be repeted

	#Hyper Parameters
	Sigma_omega = np.identity(M)*0.1
	Sigma_mu = np.identity(D)

	sigma_y = 31

	#Initial State
	m_mu = np.random.rand(D).reshape((D,1))
	m_omega = np.zeros((M,D))
	mu_x = np.random.rand(M,N)

	Sigma_mu_hat = np.random.rand(D,D)
	Sigma_oemga_hat = np.random.rand(M,M)
	Sigma_x_hat = np.random.rand(M,M)

	for i in tqdm(range(MAXITER)):

		#mu is reloated

		sum1 = np.zeros((D,1))
		for n in range(N):
			sum1 = sum1 + Y[:,n].reshape((D,1)) - np.dot(m_omega.T,mu_x[:,n].reshape((M,1)))

		m_mu = sigma_y**(-2) * np.dot(Sigma_mu_hat , sum1)
		Sigma_mu_hat = np.linalg.inv(N * sigma_y**(-2) * np.identity(D) + np.linalg.inv(Sigma_mu))

		#W is reloated

		for d in range(D):
			sum2 = np.zeros((M,1))
			sum3 = np.zeros((M,M))
			for n in range(N):

				sum2 = sum2 + (Y[d,n] - m_mu[d])*mu_x[:,n].reshape((M,1))
				sum3 = sum3 + np.dot(mu_x[:,n].reshape((M,1)),mu_x[:,n].reshape((1,M))) + Sigma_x_hat
			
			m_omega[:,d] = sigma_y**(-2) * np.dot(Sigma_oemga_hat , sum2).reshape((1,M))

		Sigma_oemga_hat = np.linalg.inv(sigma_y**(-2) * sum3 * np.linalg.inv(Sigma_omega))

		#X is reloated
	     
		sum4 = np.zeros((M,M))
		for d in range(D):
			sum4 = sum4 + np.dot(m_omega[:,d].reshape((M,1)),m_omega[:,d].reshape((1,M))) + Sigma_oemga_hat
		Sigma_x_hat = np.linalg.inv(sigma_y**(-2)*sum4 + np.identity(M))

		for n in range(N):		
			mu_x[:,n] = ( sigma_y**(-2)*Sigma_x_hat @ m_omega @ (Y[:,n].reshape((D,1)) - m_mu) ).reshape((1,M))
		


	reductioned_Y = m_omega.T @ mu_x + m_mu
	cv2.imwrite("reductioned_img.jpg",reductioned_Y)
