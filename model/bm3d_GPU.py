import torch as t
import torchvision as tv

from torchvision import transforms
from PIL import Image
import json
from wavelet_GPU import wavelet_GPU
from time import time


class BM3D_GPU():
    def __init__(self, pic_dir='src/img.jpg', data=None):
        super().__init__()
        self.config = json.load(open('config.json'))
        self.DEVICE = self.config["DEVICE"]
        if data is None:
            self.PIC = transforms.ToTensor()(Image.open(pic_dir)).to(self.DEVICE)
        else:
            # Only for debug

            self.PIC = data.unsqueeze(0)
        # Define some parameters
        self.BLOCK_SIZE = 5
        self.BLK_STRIDE = 3
        self.WINDOW_SIZE = 25
        self.TH = 2500
        self.MAX_COUNT = 400
        self.WAVELET = wavelet_GPU()

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
        elif x+self.WINDOW_SIZE//2 >= self.PIC.shape[-2]:
            xl = self.PIC.shape[-2]-self.WINDOW_SIZE
            xr = self.PIC.shape[-2]
        else:
            xl = x-self.WINDOW_SIZE//2
            xr = x+self.WINDOW_SIZE//2+1

        if y-self.WINDOW_SIZE//2 < 0:
            yl = 0
            yr = self.WINDOW_SIZE
        elif y+self.WINDOW_SIZE//2 >= self.PIC.shape[-1]:
            yl = self.PIC.shape[-1]-self.WINDOW_SIZE
            yr = self.PIC.shape[-1]
        else:
            yl = y-self.WINDOW_SIZE//2
            yr = y+self.WINDOW_SIZE//2+1

        # Build the compare window
        window = self.PIC[:, xl:xr, yl:yr]
        assert window.shape[-2] == 25
        assert window.shape[-1] == 25
        [M, N] = window.shape[-2:]
        try:
            window = window.unsqueeze(0)
            window = t.nn.functional.pad(window, (self.BLOCK_SIZE//2,self.BLOCK_SIZE//2,self.BLOCK_SIZE//2,self.BLOCK_SIZE//2), 'reflect').squeeze(0)
        except:
            print("windows padding error")
        assert window.shape[-2] == 29
        assert window.shape[-1] == 29
        # add the origin block into the dict
        ori_blk = window[:, x-xl:x-xl +
                         self.BLOCK_SIZE, y-yl:y-yl+self.BLOCK_SIZE]
        assert ori_blk.shape[-2] == self.BLOCK_SIZE
        assert ori_blk.shape[-1] == self.BLOCK_SIZE
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
                            tmp_block = window[:, ib*stride:ib*stride +
                                               Bsize, jb*stride:jb*stride+Bsize]
                            if tmp_block.shape[-1] != Bsize or tmp_block.shape[-2] != Bsize:
                                break
                            self.ASSEMBLE_DICT['{}_{}'.format(x, y)].append(
                                self.DWT2D(tmp_block))
                        except:
                            # if the block not fit, continue
                            break

        # Start to compare the distance
        LIST = self.ASSEMBLE_DICT['{}_{}'.format(x, y)]
        LIST = sorted(LIST, key=lambda x: t.sum((x-ori_blk_DWT)**2))
        LIST = LIST[:self.MAX_COUNT]
        self.ASSEMBLE_DICT['{}_{}'.format(x, y)] = LIST

    def DWT2D(self, X):
        out = self.WAVELET.dwt(X)
        out = self.WAVELET.dwt(out.permute(0, 2, 1)).permute(0, 2, 1)
        return out

    def iDWT2D(self, X):
        out = self.WAVELET.idwt(X)
        out = self.WAVELET.idwt(out.permute(0, 2, 1)).permute(0, 2, 1)
        return out

    def DFT(self, X):
        [M, N] = X.shape
        tmp1 = t.arange(M).reshape(-1, 1)
        tmp2 = t.arange(N).reshape(-1, 1)
        x1 = tmp1.T*tmp1
        x2 = tmp2.T*tmp2
        Wr = t.exp(-2*t.pi*1j*x1/M)
        Wc = t.exp(-2*t.pi*1j*x2/N)
        F = Wr.dot(X).dot(Wc)
        return F

    def IDFT(self, X):
        [M, N] = X.shape
        tmp1 = t.arange(M).reshape(-1, 1)
        tmp2 = t.arange(N).reshape(-1, 1)
        x1 = tmp1.T*tmp1
        x2 = tmp2.T*tmp2
        Wr = t.exp(2*t.pi*1j*x1/M)
        Wc = t.exp(2*t.pi*1j*x2/N)
        F = Wr.dot(X).dot(Wc)
        return F

    def Stage_One(self):
        # for c in range(self.PIC.shape[0]):
        for i in range(self.PIC.shape[1]):
            for j in range(self.PIC.shape[2]):
                self.find_simier_blk(i, j)
                print('{}_{} done.'.format(i, j))
        print('completed')


if __name__ == "__main__":
    starttime = time()
    a = t.randn(64, 64).to('cuda')
    model = BM3D_GPU(data=a)
    model.Stage_One()
    endtime = time()
    print('\n\n\n Cost time:{}'.format(endtime-starttime))
    # print(model.DFT(a), model.DFT(a).shape)
    # result = model.IDFT(model.DFT(a))
    # print(t.sum((a-result)**2))
