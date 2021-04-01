import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


e_vec = np.load('data_E_vectors_circle200x200e120.npy')
N = 300

print('done load')



def cMap1():

    v = 10
    k = 256

    vals = np.ones((k, 4))
    vals[:, 0] = np.array([(i % v)/v for i in range(k)])
    vals[:, 1] = np.array([((i + 5) % v)/v for i in range(k)])
    vals[:, 2] = np.array([((i + 7) % v)/v for i in range(k)])
    newcmp = ListedColormap(vals)

    return newcmp

def cMap2():
    colors = [(234/255, 230/255, 202/255),
              (114/255, 0, 0),
              (234/255, 230/255, 202/255),
              (114/255, 0, 0),
              (234/255, 230/255, 202/255),
              (114/255, 0, 0),
              (30/255, 23/255, 20/255),
              (234/255, 230/255, 202/255),
              (114/255, 0, 0),
              (30/255, 23/255, 20/255),
              (234/255, 230/255, 202/255),
              (30/255, 23/255, 20/255),
              (114/255, 0, 0)]  # R -> G -> B

    cmap = LinearSegmentedColormap.from_list('my_list', colors, N=30)
    return cmap

for i in range(100, 120):

    figure(num = None, figsize=(6, 6), dpi=300)

    plt.axis('off') 

    temp = pow( np.absolute( e_vec[:,i].reshape(N,N) ) ,2) 

    newcmp = cMap2()

    #newcmp = 'flag_r'

    plot = plt.imshow(temp, cmap = newcmp, interpolation='lanczos') 

    plt.savefig( 'A' + str(i) + 'test' + '.png', bbox_inches = 'tight')
    
    print(' + ' + str(i))

    plt.show()

    plt.close()


print('done saving')
