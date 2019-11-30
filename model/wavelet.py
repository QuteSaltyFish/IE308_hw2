import torch as t
import torchvision as tv
import numpy as np
from torchvision import transforms
from PIL import Image
import json

class wavelet():
    '''
    Used to stored the mateix of wavelet 
    '''
    def __init__(self):
        self.M_dict = {}
        self.iM_dict = {}
    
    def dwt(self, data):
        size = data.shape[-1]
        idx = (int)(np.log2(size))
        try: 
            # TO test if have the weight stored in the dict
            self.M_dict['{}_{}'.format(size, idx)]
            print('Exist In the Dict')
        except KeyError:
            loop = idx
            print('Not Exist In the Dict')
            self.M_dict['{}_{}'.format(size, idx)] = []
            while(loop != 0):
                w = np.identity(size, dtype=float)
                w[:2*loop, :2*loop] = 0
                for i in range(loop):
                    w[2*i, i] = 0.5
                    w[2*i+1, i] = 0.5
                    w[2*i, i+loop] = 0.5
                    w[2*i+1, i+loop] = -0.5
                self.M_dict['{}_{}'.format(size, idx)].append(w)
                print(w)
                loop -= 1
        finally:
            for w in self.M_dict['{}_{}'.format(size, idx)]:
                data = data.dot(w)
            print(data)
            return data
            
    def idwt(self, data):
        size = data.shape[-1]
        idx = (int)(np.log2(size))
        try: 
            # TO test if have the weight stored in the dict
            self.iM_dict['{}_{}'.format(size, idx)]
            print('Exist In the Dict')
        except KeyError:
            loop = idx
            print('Not Exist In the Dict')
            try:
                self.iM_dict['{}_{}'.format(size, idx)] = [np.linalg.inv(item) for item in self.M_dict['{}_{}'.format(size, idx)]]
                self.iM_dict['{}_{}'.format(size, idx)].reverse()
            except:
                self.iM_dict['{}_{}'.format(size, idx)] = []
                while(loop != 0):
                    w = np.identity(size, dtype=float)
                    w[:2*loop, :2*loop] = 0
                    for i in range(loop):
                        w[2*i, i] = 0.5
                        w[2*i+1, i] = 0.5
                        w[2*i, i+loop] = 0.5
                        w[2*i+1, i+loop] = -0.5
                    self.iM_dict['{}_{}'.format(size, idx)].append(np.linalg.inv(w))
                    print(w)
                    loop -= 1
                self.iM_dict['{}_{}'.format(size, idx)].reverse()
        finally:
            for w in self.iM_dict['{}_{}'.format(size, idx)]:
                data = data.dot(w)
            print(data)
            return data

if __name__ == "__main__":
    a = np.array([9, 7, 3, 5])
    wavelet = wavelet()
    out = wavelet.dwt(a)
    out = wavelet.idwt(out)