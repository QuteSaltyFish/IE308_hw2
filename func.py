import numpy as np
import cupy
import cv2
from PIL import Image


class BM3D():
    def __init__(self):
        # parameters
        self.sigma = 25
        # Threshold for Hard Thresholding
        self.Threshold_Hard3D = 2.7*self.sigma
        self.First_Match_threshold = 2500             # 用于计算block之间相似度的阈值
        self.Step1_max_matched_cnt = 16              # 组最大匹配的块数
        self.Step1_Blk_Size = 8                     # block_Size即块的大小，8*8
        # Rather than sliding by one pixel to every next reference block, use a step of Nstep pixels in both horizontal and vertical directions.
        self.Step1_Blk_Step = 3
        self.Step1_Search_Step = 3                   # 块的搜索step
        # Search for candidate matching blocks in a local neighborhood of restricted size NS*NS centered
        self.Step1_Search_Window = 39

        self.Second_Match_threshold = 400           # 用于计算block之间相似度的阈值
        self.Step2_max_matched_cnt = 32
        self.Step2_Blk_Size = 8
        self.Step2_Blk_Step = 3
        self.Step2_Search_Step = 3
        self.Step2_Search_Window = 39

        self.Beta_Kaiser = 2.0

        self.wavelet_matrix = np.array([
            [1 / 8, 1 / 8, 1 / 4, 0, 1 / 2, 0, 0, 0],
            [1 / 8, 1 / 8, 1 / 4, 0, -1 / 2, 0, 0, 0],
            [1 / 8, 1 / 8, -1 / 4, 0, 0, 1 / 2, 0, 0],
            [1 / 8, 1 / 8, -1 / 4, 0, 0, -1 / 2, 0, 0],
            [1 / 8, -1 / 8, 0, 1 / 4, 0, 0, 1 / 2, 0],
            [1 / 8, -1 / 8, 0, 1 / 4, 0, 0, -1 / 2, 0],
            [1 / 8, -1 / 8, 0, -1 / 4, 0, 0, 0, 1 / 2],
            [1 / 8, -1 / 8, 0, -1 / 4, 0, 0, 0, -1 / 2]
        ])

    def init_matrix(self, img, _blk_size, _Beta_Kaiser):
        '''
        used to init the matrix used inthe process and the kaiser window.
        '''
        m_shape = img.shape
        self.m_img = np.zeros(m_shape, dtype=np.float)
        self.m_wight = np.zeros(m_shape, dtype=np.float)
        self.m_Kaiser = np.kaiser(_blk_size, _Beta_Kaiser)
        return self.m_img, self.m_wight, self.m_Kaiser

    def Locate_blk(self, i, j, blk_step, block_Size, width, height):
        '''
        Check the blk to ensure it will not be out of the limit.
        '''
        if i*blk_step+block_Size < width:
            point_x = i*blk_step
        else:
            point_x = width - i*blk_step

        if j*blk_step+block_Size < height:
            point_y = j*blk_step
        else:
            point_y = height - i*blk_step

        # The vertice of the reference point
        return np.array([point_x, point_y], dtype=int)

    def Search_Window(self, _noisyImg, _BlockPoint, _WindowSize, Blk_Size):
        '''
        Return an array [x, y] contains the vertice of the Search Window
        '''
        point_x = _BlockPoint[0]
        point_y = _BlockPoint[1]

        # Obtain the four points of the window
        LX = point_x + Blk_Size//2 - _WindowSize//2
        LY = point_y + Blk_Size//2 - _WindowSize//2
        RX = LX + _WindowSize
        RY = LY + _WindowSize

        # Check if the window over the limit
        if LX < 0:
            LX = 0
        elif RX > _noisyImg.shape[0]:
            LX = _noisyImg.shape[0]-_WindowSize
        if LY < 0:
            LY = 0
        elif RY > _noisyImg.shape[1]:
            LY = _noisyImg.shape[1]-_WindowSize

        return np.array([LX, LY], dtype=int)

    def wavelet(self, img): 
        return self.wavelet_matrix.T.dot(img).dot(self.wavelet_matrix)

    def Step_1_Fast_Match(self, _noisyImg, _BlockPoint):
        """Fast Matching"""
        '''
        Return a array contains the most alike blocks including itself.
        '''
        present_x, present_y = _BlockPoint
        Blk_Size = self.Step1_Blk_Size
        Search_Step = self.Step1_Search_Step
        Threhold = self.First_Match_threshold
        max_match = self.Step1_max_matched_cnt
        window_size = self.Step1_Search_Window

        # Save the position of the alike blocks
        blk_positions = np.zeros([max_match, 2], dtype=int)
        Final_Similar_blocks = np.zeros(
            [max_match, Blk_Size, Blk_Size], dtype=float)

        img = _noisyImg[present_x:present_x +
                        Blk_Size, present_y:present_y+Blk_Size]
        if img.shape[0]==Blk_Size or img.shape[1]==Blk_Size:
            return None, None, None
        dct_img = np.fft.fft2(img)

        Final_Similar_blocks[0] = dct_img
        blk_positions[0] = _BlockPoint

        Window_location = self.Search_Window(
            _noisyImg, _BlockPoint, window_size, Blk_Size)
        blk_num = int((window_size - Blk_Size)/Search_Step)
        present_x, present_y = Window_location

        Similar_blocks = np.zeros(
            [blk_num**2, Blk_Size, Blk_Size], dtype=float)
        m_Blkpositions = np.zeros([blk_num**2, 2], dtype=int)
        Distance = np.zeros(blk_num**2, dtype=float)

        # start search
        match_cnt = 0
        for i in range(blk_num):
            for j in range(blk_num):
                tmp_img = _noisyImg[present_x:present_x +
                                    Blk_Size, present_y:present_y+Blk_Size]
                dct_tmp_img = np.fft.fft2(tmp_img)
                m_Distance = np.linalg.norm(
                    (dct_tmp_img - dct_img)**2/(Blk_Size**2))

                if m_Distance < Threhold and m_Distance > 0:
                    Similar_blocks[match_cnt] = dct_tmp_img
                    m_Blkpositions[match_cnt] = [present_x, present_y]
                    Distance[match_cnt] = m_Distance
                    match_cnt += 1
                present_y += Search_Step

            present_x += Search_Step
            present_y = Window_location[1]
        Distance = Distance[:match_cnt]
        Sort = Distance.argsort()

        # 统计一下找到了多少相似的blk
        if match_cnt < max_match:
            Count = match_cnt + 1
        else:
            Count = max_match

        if Count > 0:
            for i in range(1, Count):
                Similar_blocks[i] = [Sort[i - 1]]
                blk_positions[i] = m_Blkpositions[Sort[i - 1]]
        return Final_Similar_blocks, blk_positions, Count

    def Step_1_3DFiltering(self, _similar_blocks):
        statis_nonzero = 0  # 非零元素个数
        m_Shape = _similar_blocks.shape
        for i in range(m_Shape[1]):
            for j in range(m_Shape[2]):
                tem_Vct_Trans = np.fft.fft(_similar_blocks[:, i, j])
                tem_Vct_Trans[np.abs(tem_Vct_Trans[:]) <
                              self.Threshold_Hard3D] = 0.
                statis_nonzero += tem_Vct_Trans.nonzero()[0].size
                _similar_blocks[:, i, j] = np.fft.fft(tem_Vct_Trans)[0]
        return _similar_blocks, statis_nonzero

    def Aggregation_hardthreshold(self, _similar_blocks, blk_positions, m_basic_img, m_wight_img, _nonzero_num, Count, Kaiser):
        _shape = _similar_blocks.shape
        if _nonzero_num < 1:
            _nonzero_num = 1
        block_wight = (1./_nonzero_num) * Kaiser
        for i in range(Count):
            point = blk_positions[i, :]
            tem_img = (1./_nonzero_num)*cv2.idct(_similar_blocks[i, :, :]) * Kaiser
            m_basic_img[point[0]:point[0]+_shape[1], point[1]:point[1]+_shape[2]] += tem_img
            m_wight_img[point[0]:point[0]+_shape[1], point[1]:point[1]+_shape[2]] += block_wight
        return m_basic_img, m_wight_img

    def BM3D_1st_step(self, _noisyImg):
        """第一步,基本去噪"""
        # 初始化一些参数：
        (width, height) = _noisyImg.shape   # 得到图像的长宽
        block_Size = self.Step1_Blk_Size         # 块大小
        blk_step = self.Step1_Blk_Step           # N块步长滑动
        Width_num = (width - block_Size)/blk_step
        Height_num = (height - block_Size)/blk_step

        # 初始化几个数组
        Basic_img, m_Wight, m_Kaiser = self.init_matrix(_noisyImg, self.Step1_Blk_Size, self.Beta_Kaiser)

        # 开始逐block的处理,+2是为了避免边缘上不够
        for i in range(int(Width_num+2)):
            for j in range(int(Height_num+2)):
                # m_blockPoint当前参考图像的顶点
                m_blockPoint = self.Locate_blk(i, j, blk_step, block_Size, width, height)       # 该函数用于保证当前的blk不超出图像范围
                Similar_Blks, Positions, Count = self.Step_1_Fast_Match(_noisyImg, m_blockPoint)
                if Similar_Blks is None:
                    continue
                Similar_Blks, statis_nonzero = self.Step_1_3DFiltering(Similar_Blks)
                Basic_img, m_wight = self.Aggregation_hardthreshold(Similar_Blks, Positions, Basic_img, m_Wight, statis_nonzero, Count, m_Kaiser)
        Basic_img[:, :] /= m_Wight[:, :]
        basic = np.matrix(Basic_img, dtype=int)
        basic.astype(np.uint8)

        return basic
if __name__ == '__main__':
    img = np.array(Image.open('src/img.jpg').convert('L'))
    model = BM3D()
    basic = model.BM3D_1st_step(img)
    print(basic)
