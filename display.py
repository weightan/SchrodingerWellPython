

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

e_vec = np.load('data_E_vectors_circle200x200e120.npy')
N = 200

print('done load')

for i in range(80, 99):
    figure(num = None, figsize=(6, 6), dpi=300)
    plt.axis('off') 
    plot = plt.imshow( pow( np.absolute( e_vec[:,i].reshape(N,N) ) ,2), cmap = 'nipy_spectral', interpolation='lanczos') 
    plt.savefig( str(i) + 'e.png', bbox_inches = 'tight') 
    plt.show()


print('done saving')
