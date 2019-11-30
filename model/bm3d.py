import torch as t
import torchvision as tv
import numpy as np
from torchvision import transforms
from PIL import Image
import json

class wavelet():
    
class BM3D():
    def __init__(self, pic_dir='src/img.jpg'):
        super().__init__()
        self.config = json.load(open('config.json'))
        self.PIC = np.array(Image.open(pic_dir))
        # Define some parameters
        self.BLOCK_SIZE = 5
        self.BLK_STRIDE = 3
        self.WINDOW_SIZE = 25
        self.TH = 2500
        self.MAX_COUNT = 400

    def get_window(self, x, y):
        if x-self.WINDOW_SIZE//2 < 0:
            xl = 0
            xr = self.WINDOW_SIZE
        else:
            xl = x-self.WINDOW_SIZE//2
            xr = x+self.WINDOW_SIZE//2

        if y-self.WINDOW_SIZE//2 < 0:
            yl = 0
            yr = self.WINDOW_SIZE
        else:
            yl = x-self.WINDOW_SIZE//2
            yr = x+self.WINDOW_SIZE//2

        window = self.PIC[:, xl:xr, yl:yr]
        [M, N] = window = 
        window = np.pad(window, self.BLOCK_SIZE//2, mode='reflect')
        for i in range(0, self.BLK_STRIDE, self.BLK_STRIDE):
            for j in range(0, self.BLK_STRIDE, self.BLK_STRIDE):
                if i+

    def DFT(self, X):
        [M, N] = X.shape
        tmp1 = np.arange(M).reshape(-1, 1)
        tmp2 = np.arange(N).reshape(-1, 1)
        x1 = tmp1.T*tmp1
        x2 = tmp2.T*tmp2
        Wr = np.exp(-2*np.pi*1j*x1/M)
        Wc = np.exp(-2*np.pi*1j*x2/N)
        F = Wr.dot(X).dot(Wc)
        return F

    def IDFT(self, X):
        [M, N] = X.shape
        tmp1 = np.arange(M).reshape(-1, 1)
        tmp2 = np.arange(N).reshape(-1, 1)
        x1 = tmp1.T*tmp1
        x2 = tmp2.T*tmp2
        Wr = np.exp(2*np.pi*1j*x1/M)
        Wc = np.exp(2*np.pi*1j*x2/N)
        F = Wr.dot(X).dot(Wc)
        return F

    def Stage_One(self):
        for c in range(self.PIC.shape[0]):
            for i in range(self.PIC.shape[1]):
                for j in range(self.PIC.shape[2]):
                    pad_size = self.BLOCK_SIZE//2
                    self.PIC = np.pad(self.PIC, pad_size, mode='reflect')
                    # select the window
                    # window = self.get_window()


if __name__ == "__main__":
    model = BM3D()
    a = np.random.randn(3, 14)
    print(model.DFT(a), model.DFT(a).shape)
    result = model.IDFT(model.DFT(a))
    print(np.sum((a-result)**2))