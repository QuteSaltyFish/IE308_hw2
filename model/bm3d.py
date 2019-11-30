import torch as t
import torchvision as tv
import numpy as np
from torchvision import transforms
from PIL import Image
import json
from wavelet import wavelet


class BM3D():
    def __init__(self, pic_dir='src/img.jpg', data=None):
        super().__init__()
        if data is None:
            self.config = json.load(open('config.json'))
            self.PIC = np.array(Image.open(pic_dir)).transpose(2, 0, 1)
        else:
            # Only for debug

            self.PIC = np.expand_dims(data, axis=0)
        # Define some parameters
        self.BLOCK_SIZE = 5
        self.BLK_STRIDE = 3
        self.WINDOW_SIZE = 25
        self.TH = 2500
        self.MAX_COUNT = 400
        self.WAVELET = wavelet()

        self.ASSEMBLE_DICT = {}

    def find_simier_blk(self, x, y):
        # Creat dict to store the assemble maps
        try:
            self.ASSEMBLE_DICT['{}_{}'.format(x, y)]
        except:
            self.ASSEMBLE_DICT['{}_{}'.format(x, y)] = []

        # Adjust the windows location
        if x-self.WINDOW_SIZE//2 < 0:
            xl = 0
            xr = self.WINDOW_SIZE
        elif x-self.WINDOW_SIZE//2 > self.PIC.shape[-2]:
            xl = self.PIC.shape[-2]-1-self.WINDOW_SIZE
            xr = self.PIC.shape[-2]-1
        else:
            xl = x-self.WINDOW_SIZE//2
            xr = x+self.WINDOW_SIZE//2

        if y-self.WINDOW_SIZE//2 < 0:
            yl = 0
            yr = self.WINDOW_SIZE
        elif x-self.WINDOW_SIZE//2 > self.PIC.shape[-1]:
            yl = self.PIC.shape[-1]-1-self.WINDOW_SIZE
            yr = self.PIC.shape[-1]-1
        else:
            yl = x-self.WINDOW_SIZE//2
            yr = x+self.WINDOW_SIZE//2

        # Build the compare window
        window = self.PIC[:, xl:xr, yl:yr]
        [M, N] = window.shape[-2:]
        window = np.pad(window, self.BLOCK_SIZE//2, mode='reflect')

        # add the origin block into the dict
        ori_blk = window[x-self.BLOCK_SIZE//2:x-self.BLOCK_SIZE//2+self.BLOCK_SIZE,
                         y-self.BLOCK_SIZE//2:y-self.BLOCK_SIZE//2+self.BLOCK_SIZE]
        ori_blk_DWT = self.DWT2D(ori_blk)
        self.ASSEMBLE_DICT['{}_{}'.format(x, y)].append(ori_blk_DWT)

        # Start to move the block
        stride = self.BLK_STRIDE
        Bsize = self.BLOCK_SIZE
        for iw in range(0, self.BLK_STRIDE, self.BLK_STRIDE):
            for jw in range(0, self.BLK_STRIDE, self.BLK_STRIDE):
                # slide and record the DWT map of them
                for ib in range(M):
                    for jb in range(N):
                        try:
                            tmp_block = window[ib*stride:ib*stride +
                                               Bsize, jb*stride:jb*stride+Bsize]
                            self.ASSEMBLE_DICT['{}_{}'.format(x, y)].append(
                                self.DWT2D(tmp_block))
                        except:
                            # if the block not fit, continue
                            continue

        # Start to compare the distance
        LIST = self.ASSEMBLE_DICT['{}_{}'.format(x, y)]
        LIST = sorted(LIST, key=lambda x: np.sum((x-ori_blk_DWT)**2))
        LIST = LIST[:self.MAX_COUNT]
        self.ASSEMBLE_DICT['{}_{}'.format(x, y)] = LIST


    def DWT2D(self, X):
        out = self.WAVELET.dwt(X)
        out = self.WAVELET.dwt(out.T).T
        return out

    def iDWT2D(self, X):
        out = self.WAVELET.idwt(X)
        out = self.WAVELET.idwt(out.T).T
        return out

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
        # for c in range(self.PIC.shape[0]):
        for i in range(self.PIC.shape[1]):
            for j in range(self.PIC.shape[2]):
                self.find_simier_blk()


if __name__ == "__main__":

    a = np.random.randn(64, 64)
    model = BM3D(data=a)
    print(model.DFT(a), model.DFT(a).shape)
    result = model.IDFT(model.DFT(a))
    print(np.sum((a-result)**2))
