import cv2
import numpy as np
from scipy import signal
import math
import matplotlib.pyplot as plt
from PIL import ImageEnhance

def getClosenessWeight(sigma_g,H,W):
    r,c=np.mgrid[0:H:1,0:W:1]
    r-=(H-1)//2
    c-=int(W-1)//2
    closeWeight=np.exp(-0.5*(np.power(r,2)+np.power(c,2))/math.pow(sigma_g,2))
    return closeWeight
def bfltGray(I,H,W,sigma_g,sigma_d):
    closenessWeight=getClosenessWeight(sigma_g,H,W)
    cH=(H-1)//2
    cW=(W-1)//2
    rows,cols=I.shape
    bfltGrayImage=np.zeros(I.shape,np.float32)
    for r in range(rows):
        for c in range(cols):
            pixel=I[r][c]
            rTop=0 if r-cH<0 else r-cH
            rBottom=rows-1 if r+cH>rows-1 else r+cH
            cLeft=0 if c-cW<0 else c-cW
            cRight=cols-1 if c+cW>cols-1 else c+cW
            region=I[rTop:rBottom+1,cLeft:cRight+1]
            similarityWeightTemp=np.exp(-0.5*np.power(region-pixel,2.0)/math.pow(sigma_d,2))
            closenessWeightTemp=closenessWeight[rTop-r+cH:rBottom-r+cH+1,cLeft-c+cW:cRight-c+cW+1]
            weightTemp=similarityWeightTemp*closenessWeightTemp
            weightTemp=weightTemp/np.sum(weightTemp)
            bfltGrayImage[r][c]=np.sum(region*weightTemp)
    return bfltGrayImage

def convolve(filter,mat,padding,strides):
    '''
    :param filter:卷积核，必须为二维(2 x 1也算二维) 否则返回None
    :param mat:图片
    :param padding:对齐
    :param strides:移动步长
    :return:返回卷积后的图片。(灰度图，彩图都适用)
    '''
    result = None
    filter_size = filter.shape
    mat_size = mat.shape
    if len(filter_size) == 2:
        if len(mat_size) == 3:
            channel = []
            for i in range(mat_size[-1]):
                pad_mat = np.pad(mat[:,:,i], ((padding[0], padding[1]), (padding[2], padding[3])), 'constant')
                temp = []
                for j in range(0,mat_size[0],strides[1]):
                    temp.append([])
                    for k in range(0,mat_size[1],strides[0]):
                        val = (filter*pad_mat[j:j+filter_size[0],k:k+filter_size[1]]).sum()
                        temp[-1].append(val)
                channel.append(np.array(temp))

            channel = tuple(channel)
            result = np.dstack(channel)
        elif len(mat_size) == 2:
            channel = []
            pad_mat = np.pad(mat, ((padding[0], padding[1]), (padding[2], padding[3])), 'constant')
            for j in range(0, mat_size[0], strides[1]):
                channel.append([])
                for k in range(0, mat_size[1], strides[0]):
                    val = (filter * pad_mat[j:j + filter_size[0],k:k + filter_size[1]]).sum()
                    channel[-1].append(val)


            result = np.array(channel)


    return result

def linear_convolve(filter,mat,padding=None,strides=[1,1]):
    '''
    :param filter:线性卷积核
    :param mat:图片
    :param padding:对齐
    :param strides:移动步长
    :return:返回卷积后的图片。(灰度图，彩图都适用) 若不是线性卷积核，返回None
    '''
    result = None
    filter_size = filter.shape
    if len(filter_size) == 2 and 1 in filter_size:
        if padding == None or len(padding) < 2:
            if filter_size[1] == 1:
                padding = [filter_size[0]//2,filter_size[0]//2]
            elif filter_size[0] == 1:
                padding = [filter_size[1]//2,filter_size[1]//2]
        if filter_size[0] == 1:
            result = convolve(filter,mat,[0,0,padding[0],padding[1]],strides)
        elif filter_size[1] == 1:
            result = convolve(filter, mat, [padding[0],padding[1],0,0], strides)

    return result

def _2_dim_divided_convolve(filter,mat):
    '''

    :param filter: 线性卷积核,必须为二维(2 x 1也算二维) 否则返回None
    :param mat: 图片
    :return: 卷积后的图片,(灰度图，彩图都适用) 若不是线性卷积核，返回None
    '''
    result = None
    if 1 in filter.shape:
        result = linear_convolve(filter,mat)
        result = linear_convolve(filter.T,result)

    return result

def judgeConnect(m2,threshold):
    e = 0.01
    s = []
    cood = []
    for i in range(m2.shape[0]):
        cood.append([])
        for j in range(m2.shape[1]):
            cood[-1].append([i,j])
            if abs(m2[i,j] - 255) < e:
                s.append([i,j])
    cood = np.array(cood)

    while not len(s) == 0:
        index = s.pop()
        jud = m2[max(0, index[0] - 1):min(index[0] + 2, m2.shape[1]), max(0, index[1] - 1):min(index[1] + 2, m2.shape[0])]
        jud_i = cood[max(0, index[0] - 1):min(index[0] + 2, cood.shape[1]), max(0, index[1] - 1):min(index[1] + 2, cood.shape[0])]
        jud = (jud > threshold[0])&(jud < threshold[1])
        jud_i = jud_i[jud]
        for i in range(jud_i.shape[0]):
            s.append(list(jud_i[i]))
            m2[jud_i[i][0],jud_i[i][1]] = 255

    return m2



def DecideAndConnectEdge(g_l,g_t,threshold = None):
    if threshold == None:
        lower_boundary = g_l.mean()*0.5
        threshold = [lower_boundary,lower_boundary*3]
        print(threshold)
        # threshold = [20, 50]
    result = np.zeros(g_l.shape)

    for i in range(g_l.shape[0]):
        for j in range(g_l.shape[1]):
            isLocalExtreme = True
            eight_neiborhood = g_l[max(0,i-1):min(i+2,g_l.shape[0]),max(0,j-1):min(j+2,g_l.shape[1])]
            if eight_neiborhood.shape == (3,3):
                if g_t[i,j] <= -1:
                    x = 1/g_t[i,j]
                    first = eight_neiborhood[0,1] + (eight_neiborhood[0,1] - eight_neiborhood[0,0])*x
                    x = -x
                    second = eight_neiborhood[2,1] + (eight_neiborhood[2,2] - eight_neiborhood[2,1])*x
                    if not (g_l[i,j] > first and g_l[i,j] > second):
                        isLocalExtreme = False
                elif g_t[i,j] >= 1:
                    x = 1 / g_t[i, j]
                    first = eight_neiborhood[0, 1] + (eight_neiborhood[0, 2] - eight_neiborhood[0, 1]) * x
                    x = -x
                    second = eight_neiborhood[2, 1] + (eight_neiborhood[2, 1] - eight_neiborhood[2, 0]) * x
                    if not (g_l[i, j] > first and g_l[i, j] > second):
                        isLocalExtreme = False
                elif g_t[i,j] >= 0 and g_t[i,j] < 1:
                    y = g_t[i, j]
                    first = eight_neiborhood[1, 2] + (eight_neiborhood[0, 2] - eight_neiborhood[1, 2]) * y
                    y = -y
                    second = eight_neiborhood[1, 0] + (eight_neiborhood[1, 0] - eight_neiborhood[2, 0]) * y
                    if not (g_l[i, j] > first and g_l[i, j] > second):
                        isLocalExtreme = False
                elif g_t[i,j] < 0 and g_t[i,j] > -1:
                    y = g_t[i, j]
                    first = eight_neiborhood[1, 2] + (eight_neiborhood[1, 2] - eight_neiborhood[2, 2]) * y
                    y = -y
                    second = eight_neiborhood[1, 0] + (eight_neiborhood[0, 0] - eight_neiborhood[1, 0]) * y
                    if not (g_l[i, j] > first and g_l[i, j] > second):
                        isLocalExtreme = False
            if isLocalExtreme:
                result[i,j] = g_l[i,j]       #非极大值抑制

    result[result>=threshold[1]] = 255
    result[result<=threshold[0]] = 0


    result = judgeConnect(result,threshold)
    result[result!=255] = 0
    return result

def OneDimensionStandardNormalDistribution(x,sigma):
    E = -0.5/(sigma*sigma)
    return 1/(math.sqrt(2*math.pi)*sigma)*math.exp(x*x*E)

def lines_detector_hough(edge, ThetaDim=None, DistStep=None, threshold=100, halfThetaWindowSize=2,
                         halfDistWindowSize=None):
    '''
    :param edge: 经过边缘检测得到的二值图
    :param ThetaDim: hough空间中theta轴的刻度数量(将[0,pi)均分为多少份),反应theta轴的粒度,越大粒度越细
    :param DistStep: hough空间中dist轴的划分粒度,即dist轴的最小单位长度
    :param threshold: 投票表决认定存在直线的起始阈值
    :return: 返回检测出的所有直线的参数(theta,dist)

    '''
    imgsize = edge.shape
    if ThetaDim == None:
        ThetaDim = 90
    if DistStep == None:
        DistStep = 1
    MaxDist = np.sqrt(imgsize[0] ** 2 + imgsize[1] ** 2)
    DistDim = int(np.ceil(MaxDist / DistStep))

    if halfDistWindowSize == None:
        halfDistWindowSize = int(DistDim / 50)
    accumulator = np.zeros((ThetaDim, DistDim))  # theta的范围是[0,pi). 在这里将[0,pi)进行了线性映射.类似的,也对Dist轴进行了线性映射

    sinTheta = [np.sin(t * np.pi / ThetaDim) for t in range(ThetaDim)]
    cosTheta = [np.cos(t * np.pi / ThetaDim) for t in range(ThetaDim)]

    for i in range(imgsize[0]):
        for j in range(imgsize[1]):
            if not edge[i, j] == 0:
                for k in range(ThetaDim):
                    accumulator[k][int(round((i * cosTheta[k] + j * sinTheta[k]) * DistDim / MaxDist))] += 1

    M = accumulator.max()

    if threshold == None:
        threshold = int(M * 2.3875 / 17)
        print(threshold)
    result = np.array(np.where(accumulator > threshold))  # 阈值化
    temp = [[], []]
    for i in range(result.shape[1]):
        eight_neiborhood = accumulator[
                           max(0, result[0, i] - halfThetaWindowSize + 1):min(result[0, i] + halfThetaWindowSize,
                                                                              accumulator.shape[0]),
                           max(0, result[1, i] - halfDistWindowSize + 1):min(result[1, i] + halfDistWindowSize,
                                                                             accumulator.shape[1])]
        if (accumulator[result[0, i], result[1, i]] >= eight_neiborhood).all():
            temp[0].append(result[0, i])
            temp[1].append(result[1, i])

    result = np.array(temp)  # 非极大值抑制

    result = result.astype(np.float64)
    result[0] = result[0] * np.pi / ThetaDim
    result[1] = result[1] * MaxDist / DistDim

    return result


def drawLines(lines, edge, color=(255, 0, 0), err=3):
    if len(edge.shape) == 2:
        result = np.dstack((edge, edge, edge))
    else:
        result = edge
    Cos = np.cos(lines[0])
    Sin = np.sin(lines[0])

    for i in range(edge.shape[0]):
        for j in range(edge.shape[1]):
            e = np.abs(lines[1] - i * Cos - j * Sin)
            if (e < err).any():
                result[i, j] = color

    return result


if __name__ == '__main__':

    sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # filename = 'image.png'
    # img = plt.imread(filename)

    # img = img * 255
    # img = img.mean(axis=-1)  # this is a way to get a gray image.

    # sigma = 1.52
    # dim = int(np.round(6 * sigma + 1))
    # if dim % 2 == 0:
    #     dim += 1
    # linear_Gaussian_filter = [np.abs(t - (dim // 2)) for t in range(dim)]
    #
    # linear_Gaussian_filter = np.array(
    #     [[OneDimensionStandardNormalDistribution(t, sigma) for t in linear_Gaussian_filter]])
    # linear_Gaussian_filter = linear_Gaussian_filter / linear_Gaussian_filter.sum()
    #
    # img2 = _2_dim_divided_convolve(linear_Gaussian_filter, img)
    # img2 = convolve(Gaussian_filter_5, img, [2, 2, 2, 2], [1, 1])

    # plt.imshow(img2.astype(np.uint8), cmap='gray')
    # plt.axis('off')
    # plt.show()
    filename = 'image05.png'

    img = plt.imread(filename).copy()

    image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # image = cv2.equalizeHist(image)
    # cv2.imshow("yuantu", image)
    image = image / 255.0
    bfltImage = bfltGray(image, 5, 5, 19, 0.2)
    # cv2.imshow("hou", bfltImage)


    img3 = convolve(sobel_kernel_x, bfltImage, [1, 1, 1, 1], [1, 1])
    img4 = convolve(sobel_kernel_y, bfltImage, [1, 1, 1, 1], [1, 1])

    gradiant_length = (img3 ** 2 + img4 ** 2) ** (1.0 / 2)

    img3 = img3.astype(np.float64)
    img4 = img4.astype(np.float64)
    img3[img3 == 0] = 0.00000001
    gradiant_tangent = img4 / img3

    # lower_boundary = 50
    final_img = DecideAndConnectEdge(gradiant_length, gradiant_tangent)
    print(final_img)
    cv2.imshow('edge', final_img.astype(np.uint8))


    lines = lines_detector_hough(final_img)
    final_img = drawLines(lines, img)
    plt.imshow(final_img, cmap='gray')
    plt.axis('off')
    plt.show()





