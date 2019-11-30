import numpy as np
import pywt
a = np.array([9, 7, 3, 5])

w = np.array([
    [0.5, 0, 0.5, 0],
    [0.5, 0, -0.5, 0],
    [0, 0.5, 0, 0.5],
    [0, 0.5, 0, -0.5]
])
w2 = w.T
w2 = 2*w2
w2[w2 == 2] = 1

M_dict = {}
iM_dict = {}
def dwt(data):
    size = np.max(data.shape)
    idx = (int)(np.log2(size))
    try: 
        # TO test if have the weight stored in the dict
        M_dict['{}_{}'.format(size, idx)]
        print('Exist In the Dict')
    except KeyError:
        loop = idx
        print('Not Exist In the Dict')
        M_dict['{}_{}'.format(size, idx)] = []
        while(loop != 0):
            w = np.identity(size, dtype=float)
            w[:2*loop, :2*loop] = 0
            for i in range(loop):
                w[2*i, i] = 0.5
                w[2*i+1, i] = 0.5
                w[2*i, i+loop] = 0.5
                w[2*i+1, i+loop] = -0.5
            M_dict['{}_{}'.format(size, idx)].append(w)
            print(w)
            loop -= 1
    finally:
        for w in M_dict['{}_{}'.format(size, idx)]:
            data = data.dot(w)
        print(data)
        return data
def idwt(data):
    size = np.max(data.shape)
    idx = (int)(np.log2(size))
    try: 
        # TO test if have the weight stored in the dict
        iM_dict['{}_{}'.format(size, idx)]
        print('Exist In the Dict')
    except KeyError:
        loop = idx
        print('Not Exist In the Dict')
        iM_dict['{}_{}'.format(size, idx)] = []
        while(loop != 0):
            w = np.identity(size, dtype=float)
            w[:2*loop, :2*loop] = 0
            for i in range(loop):
                w[2*i, i] = 0.5
                w[2*i+1, i] = 0.5
                w[2*i, i+loop] = 0.5
                w[2*i+1, i+loop] = -0.5
            iM_dict['{}_{}'.format(size, idx)].append(np.linalg.inv(w))
            print(w)
            loop -= 1
        iM_dict['{}_{}'.format(size, idx)].reverse()
    finally:
        for w in iM_dict['{}_{}'.format(size, idx)]:
            data = data.dot(w)
        print(data)
        return data

out = dwt(a)
aa = idwt(out)
print(pywt.dwt(a, 'haar'))
print(np.linalg.inv(w))
print(w2)
print(a.dot(w).dot(np.linalg.inv(w)))
# print(a.dot(w))
