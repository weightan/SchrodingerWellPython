

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

e_vec = np.load('data_E_vectors_circle100x100e100.npy')
N = 100

print('done load')

#cmap1 = pal.ListedColormap(palettable.colorbrewer.qualitative.Dark2_7.mpl_colors)


for i in range(80, 99):
    #figi, axi = plt.subplots(1, 1)
    figure(num = None, figsize=(6, 6), dpi=300)
    plt.axis('off') 
    plot = plt.imshow( pow( np.absolute( e_vec[:,i].reshape(N,N) ) ,2), cmap = 'nipy_spectral', interpolation='lanczos') 
    #plot((6, 6), 200)
    #plt.set_prop_cycle('color', palettable.colorbrewer.qualitative.Dark2_8.mpl_colors)
    #plt.setp(axi, xticks=[], yticks=[])
    
    plt.savefig( str(i) + '5.png', bbox_inches = 'tight') 
       
    #plt.show()


print('done saving')
