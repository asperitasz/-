import sys
import numpy as np
from PyQt5.QtCore import *
from PyQt5 import QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from scipy.signal import wiener
from system import Ui_MainWindow
from scipy import signal
import cv2
from pylab import *
import random
from PIL import Image
import matplotlib.pyplot as plt
font=cv2.FONT_HERSHEY_COMPLEX
tmp_path="F:\\image\\result.png"
demo_path="F:\\image\\image\\lena.png"
save_path="F:\\image\\save.png"
class Demo(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Demo, self).__init__()
        self.setupUi(self)

        # 图像读入
        self.example.triggered.connect(self.demo_file)
        self.open.triggered.connect(self.open_file)
        self.save.triggered.connect(self.save_file)

        # 图像显示
        self.grayscale.triggered.connect(self.grayscale_func)
        self.binarization.triggered.connect(self.binarization_func)
        self.stretch.triggered.connect(self.stretch_func)
            # 灰度变换
        self.linear.triggered.connect(self.linear_func)
        self.logarithm.triggered.connect(self.logarithm_func)
        self.power.triggered.connect(self.power_func)
        self.bitmap.triggered.connect(self.bitmap_func)
            # 直方图
        self.display.triggered.connect(self.display_func)
        self.equalization.triggered.connect(self.equalization_func)
        self.regularization.triggered.connect(self.regularization_func)
            # 噪声
        self.gaussian.triggered.connect(self.gaussian_func)
        self.salt.triggered.connect(self.salt_func)
        self.poisson.triggered.connect(self.poisson_func)
        self.speckle.triggered.connect(self.speckle_func)
        # 图像增强
            # 平滑
        self.mean.triggered.connect(self.mean_func)
        self.median.triggered.connect(self.median_func)
        self.gaussian_f.triggered.connect(self.gaussian_f_func)
        self.wiener.triggered.connect(self.wiener_func)
        self.lowpass.triggered.connect(self.lowpass_func)
            # 锐化

        self.sobel.triggered.connect(self.sobel_func)
        self.laplace.triggered.connect(self.laplace_func)

        self.highpass.triggered.connect(self.highpass_func)
        # 图像分割
        self.threshold.triggered.connect(self.threshold_func)
        self.regiongrow.triggered.connect(self.regiongrow_func)

        self.actionCanny.triggered.connect(self.actionCanny_func)
        self.actionKirsch.triggered.connect(self.actionKirsch_func)
    #     self.actionRANSAC.triggered.connect(self.actionRANSAC_func)
   #     self.actionHough.triggered.connect(self.actionHough_func)

        # 特征提取
   #     self.parameters.triggered.connect(self.parameters_func)
     #   self.mark.triggered.connect(self.mark_func)
  #      self.extraction.triggered.connect(self.extraction_func)
        self.inversion.triggered.connect(self.inversion_func)
        self.action_5.triggered.connect(self.action5_5_func)
        # 形态
        self.action.triggered.connect(self.action_func1)
        self.action_2.triggered.connect(self.action_2_func)
        self.action_3.triggered.connect(self.action_3_func)
        self.action_4.triggered.connect(self.action_4_func)

    def demo_file(self):
        # 打开示例图像
        img = cv2.imread(demo_path)
        self.img_src.setPixmap(QPixmap(demo_path))

    def open_file(self):
        # 打开自选取图像
        filename, filetype = QFileDialog.getOpenFileName(self,
                                                         "打开文件", "F://image//image", "All Files(*);;Text Files(*.png);;Text Files(*.jpg)")
        if filetype == '':  # 若路径无效或打开文件失败，则什么也不做，让用户回到主界面
            return
        img = cv2.imread(filename)  # 用cv库打开文件
        cv2.imwrite(tmp_path, img)  # 将处理后的图像保存在临时路径中
        # 这里虽然只是打开图像而没有进行处理，但是也会拷贝一份作为缓存，供后续的图像处理函数读取
        self.img_src.setPixmap(QPixmap(tmp_path))  # 读取原始图像并显示

    def save_file(self):
        # 保存当前图像
        img = cv2.imread(tmp_path)
        cv2.imwrite(save_path, img)  # 将处理后的图像保存在临时路径中

    def grayscale_func(self):
        img = cv2.imread(tmp_path, 0)  # 读取临时路径中的图像，用cv库进行灰度化处理
        cv2.imwrite(tmp_path, img)  # 将处理后的图像保存在临时路径中
        self.img_demo.setPixmap(QPixmap(tmp_path))  # 显示处理后的图像

    def binarization_func(self):

        img = cv2.imread(tmp_path)  # 读取临时路径中的图像
        try:  # 如果输入的参数有效，则以该参数处理图像
            p1 = int(self.para1.toPlainText())
            if p1 < 0 or p1 > 255:
                raise ValueError
            img[img > p1] = 255
            img[img <= p1] = 0
        except Exception:  # 否则，以默认的方式处理图像
            img[img > 128] = 255
            img[img <= 128] = 0
        cv2.imwrite(tmp_path, img)  # 将处理后的图像保存在临时路径中
        self.img_demo.setPixmap(QPixmap(tmp_path))  # 显示处理后的图像
    def stretch_func(self):
        img = cv2.imread(tmp_path)  # 读取临时路径中的图像
        try:  # 如果输入的参s数有效，则以该参数处理图像
            p1 = int(self.para1.toPlainText())
            p2 = int(self.para2.toPlainText())
            img = cv2.resize(img, (p1, p2))
        except Exception:  # 否则，以默认的方式处理图像
            img = cv2.resize(img, (512, 512))

        cv2.imwrite(tmp_path, img)  # 将处理后的图像保存在临时路径中
        self.img_demo.setPixmap(QPixmap(tmp_path))  # 显示处理后的图像

    def linear_func(self):
        # 线性变换

        img = cv2.imread(tmp_path)
        isParaReady = True
        try:
            p1 = float(self.para1.toPlainText())
            c = p1
            p2 = float(self.para2.toPlainText())
            d = p2
        except Exception:
            isParaReady = False
        if not isParaReady:
            c = 1
            d = 50
        # 图像灰度转换
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 获取图像高度和宽度
        height = grayImage.shape[0]
        width = grayImage.shape[1]
        # 创建一幅图像
        result = np.zeros((height, width), np.uint8)
        # 图像灰度上移变换 DB=DA+50
        for i in range(height):
            for j in range(width):
                if (int(grayImage[i, j] + d) > 255):
                    gray = 255
                else:
                    gray = int(c * grayImage[i, j] + d)
                result[i, j] = np.uint8(gray)

        # 显示图像
   #      cv2.imshow("Gray Image", grayImage)
       #  cv2.imshow("Result", result)
        cv2.imwrite(tmp_path, result)  # 将处理后的图像保存在临时路径中
        self.img_demo.setPixmap(QPixmap(tmp_path))
    def logarithm_func(self):
        # 对数变换
        img = cv2.imread(tmp_path)  # 读取临时路径中的图像

        isParaReady = True
        try:  # 如果输入的参数有效，则以该参数处理图像
            p1 = int(self.para1.toPlainText())
            c = p1
        except Exception:  # 否则，以默认的方式处理图像
            isParaReady = False
        if not isParaReady:
            c = 42  # 如果不能成功识别到参数，那么对数变换的参数c默认为1
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        des = c * np.log(1.0 + grayImage)
        des = np.uint8(des + 0.5)  # 四舍五入转成uint8的像素数据格式类型
        cv2.imwrite(tmp_path, des)  # 将处理后的图像保存在临时路径中
        self.img_demo.setPixmap(QPixmap(tmp_path))  # 显示处理后的图像
    def power_func(self):
        # 幂
        img=cv2.imread(tmp_path) # 读取临时路径中的图像
        isParaReady=True
        try: # 如果输入的参数有效，则以该参数处理图像
            p1=float(self.para1.toPlainText())
            c=p1
            p2=float(self.para2.toPlainText())
            lamda=p2
        except Exception: # 否则，以默认的方式处理图像
            isParaReady=False
        if not isParaReady:
            c=1 # 如果不能成功识别到参数，那么幂律变换的参数c默认为1
            lamda=0.9 # 如果不能成功识别到参数，那么幂律变换的参数lamda默认为0.9，以暴露更多暗处细节
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = grayImage/255**lamda*c*255
        cv2.imwrite(tmp_path,img) # 将处理后的图像保存在临时路径中
        self.img_demo.setPixmap(QPixmap(tmp_path)) # 显示处理后的图像
    def bitmap_func(self):
        # 位图切割
        img=cv2.imread(tmp_path,0) # 读取临时路径中的图像
        isParaReady=True
        try: # 如果输入的参数有效，则以该参数处理图像
            p1=int(self.para1.toPlainText())
            layer=p1
        except Exception: # 否则，以默认的方式处理图像
            isParaReady=False
        if not isParaReady:
            layer=7 # 如果不能成功识别到参数，那么默认切第7个比特位
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                tmpvalue=img[i][j]
                bits = bin(tmpvalue)[2:].rjust(8, '0')
                fill = int(bits[-layer - 1])
                img[i][j] = 255 if fill else 0
        cv2.imwrite(tmp_path,img) # 将处理后的图像保存在临时路径中
        self.img_demo.setPixmap(QPixmap(tmp_path)) # 显示处理后的图像
    def display_func(self):
        # 直方图显示
        img = cv2.imread(tmp_path, 0)  # 读取临时路径中的图像
        hist = cv2.calcHist([img], [0], None, [256], [0, 255])
        plt.plot(hist,color='b')
        plt.show()
    def equalization_func(self):
        # 直方图均衡
        img = cv2.imread(tmp_path, 0)  # 读取临时路径中的图像
        equ = cv2.equalizeHist(img)

        plt.subplot(221)
        plt.imshow(img,plt.cm.gray)
        plt.subplot(222)
        plt.hist(img.ravel(), 256)

        plt.subplot(223)
        plt.imshow(equ, plt.cm.gray)
        plt.axis('off')

        plt.subplot(224)
        plt.hist(equ.ravel(), 256)

        plt.savefig("equalization.jpg")
        plt.show()

    def regularization_func(self):
        # 直方图规则
        img = cv2.imread(tmp_path, 0)  # 读取临时路径中的图像
        img_hist = cv2.calcHist([img], [0], None, [256], [0, 256])

        equ = cv2.equalizeHist(img)
        equ_hist = cv2.calcHist([equ], [0], None, [256], [0, 256])

        plt.subplot(211), plt.plot(img_hist), plt.xlim([0, 255])
        plt.subplot(212), plt.plot(equ_hist), plt.xlim([0, 255])
        plt.show()
    def gaussian_func(self):

        # 高斯噪声
        img = cv2.imread(tmp_path, 0)  # 读取临时路径中的图像
        src=img.copy()
        des = np.empty(src.shape, dtype=np.uint8)
        try: # 如果输入的参数有效，则以该参数处理图像
            p1=float(self.para1.toPlainText())
            p2=float(self.para2.toPlainText())
            src = img.copy()
            src = np.array(src/255, dtype=float)
            noise = np.random.normal(p1, p2 ** 0.5, src.shape)
            des = src + noise
            if des.min() < 0:
                low_clip = -1.
            else:
                low_clip = 0.
            des = np.clip(des, low_clip, 1.0)
            des = np.uint8(des*255)
        except Exception: # 否则，以默认的方式处理图像
            src = img.copy()
            src = np.array(src/255, dtype=float)
            noise = np.random.normal(0, 0.01 ** 0.5, src.shape)
            des = src + noise
            if des.min() < 0:
                low_clip = -1.
            else:
                low_clip = 0.
            des = np.clip(des, low_clip, 1.0)
            des = np.uint8(des*255)
        cv2.imwrite(tmp_path,des) # 将处理后的图像保存在临时路径中
        self.img_demo.setPixmap(QPixmap(tmp_path))  # 显示处理后的图像


    def salt_func(self):
        # 椒盐噪声
        img = cv2.imread(tmp_path, 0)  # 读取临时路径中的图像
        isParaReady=True
        try: # 如果输入的参数有效，则以该参数处理图像
            p1=float(self.para1.toPlainText())
            prob=p1

        except Exception: # 否则，以默认的方式处理图像
            isParaReady=False
        if not isParaReady:
            prob = 0.02
        def sp_noise(image, prob):

            '''

            添加椒盐噪声

            prob:噪声比例

            '''

            output = np.zeros(image.shape, np.uint8)

            thres = 1 - prob

            for i in range(image.shape[0]):

                for j in range(image.shape[1]):

                    rdn = random.random()

                    if rdn < prob:

                        output[i][j] = 0

                    elif rdn > thres:

                        output[i][j] = 255

                    else:

                        output[i][j] = image[i][j]

            return output

        noisy_img = sp_noise(img, prob)
        cv2.imwrite(tmp_path, noisy_img)
        self.img_demo.setPixmap(QPixmap(tmp_path))  # 显示处理后的图像


    def poisson_func(self):
        # 泊松噪声
        img = cv2.imread(tmp_path, 0)  # 读取临时路径中的图像
        # 计算图像像素的分布范围
        isParaReady=True
        try: # 如果输入的参数有效，则以该参数处理图像
            p1=float(self.para1.toPlainText())
            lam=p1

        except Exception: # 否则，以默认的方式处理图像
            isParaReady=False
        if not isParaReady:
            lam = 1000
        noise_type = np.random.poisson(lam, size=img.shape).astype(
            dtype='uint8')  # lam>=0 值越小，噪声频率就越少，size为图像尺寸
        noise_image = noise_type + img  # 将原图与噪声叠加

        # 保存图片
        cv2.imwrite(tmp_path, noise_image)
        self.img_demo.setPixmap(QPixmap(tmp_path))  # 显示处理后的图像

    def speckle_func(self):
        # 斑点噪声
        img = cv2.imread(tmp_path, 0)  # 读取临时路径中的图像
        gauss = np.random.random(img.shape)
        # 给图片添加speckle噪声
        noisy_img = img + img * gauss
        # 归一化图像的像素值
        noisy_img = np.clip(noisy_img, a_min=0, a_max=255)
        cv2.imwrite(tmp_path, noisy_img)
        self.img_demo.setPixmap(QPixmap(tmp_path))  # 显示处理后的图像

    def mean_func(self):
        # 均值滤波
        img = cv2.imread(tmp_path, 0)  # 读取临时路径中的图像
        src = img.copy()
        des = np.empty(src.shape, dtype=np.uint8)

        isParaReady = True
        try:  # 如果输入的参数有效，则以该参数处理图像
            p1 = int(self.para1.toPlainText())
            if p1 < 1 or p1 > 5:
                raise ValueError
            r = p1
        except Exception:  # 否则，以默认的方式处理图像
            isParaReady = False
        if not isParaReady:
            r = 1  # 如果不能成功识别到参数，那么默认模板的r为1，即核的边长为3

        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                count = 0
                local_sum = 0
                for ii in range(-r, r + 1):
                    for jj in range(-r, r + 1):
                        if (i + ii > 0 and j + jj > 0 and i + ii < src.shape[0] and j + jj < src.shape[1]):
                            local_sum += src[i + ii][j + jj]
                            count += 1
                des[i][j] = local_sum // count

        cv2.imwrite(tmp_path, des)  # 将处理后的图像保存在临时路径中
        self.img_demo.setPixmap(QPixmap(tmp_path))  # 显示处理后的图像

    def median_func(self):
        # 中值滤波
        img = cv2.imread(tmp_path, 0)  # 读取临时路径中的图像
        src = img.copy()

        isParaReady = True
        try:  # 如果输入的参数有效，则以该参数处理图像
            p1 = int(self.para1.toPlainText())
            if p1 < 1 or p1 > 5:
                raise ValueError
            size = p1
        except Exception:  # 否则，以默认的方式处理图像
            isParaReady = False
        if not isParaReady:
            size = 3  # 如果不能成功识别到参数，那么默认模板的r为1，即核的边长为3

        img = cv2.imread(tmp_path, 0)  # 读取临时路径中的图像
        src = img.copy()
        des = np.empty(src.shape, dtype=np.uint8)
        height = img.shape[0]
        wide = img.shape[1]
        img1 = np.zeros((height, wide), np.uint8)  # 用于存放新的图像
        for i in range(int(size / 2), height - int(size / 2)):
            for j in range(int(size / 2), wide - int(size / 2)):
                Adjacent_pixels = np.zeros(size * size, np.uint8)
                s = 0
                for k in range(-1 * int(size / 2), int(size / 2) + 1):
                    for l in range(-1 * int(size / 2), int(size / 2) + 1):
                        Adjacent_pixels[s] = img[i + k, j + l]
                        s += 1
                Adjacent_pixels.sort()  # 寻找中值
                median = Adjacent_pixels[int((size * size - 1) / 2)]  # 将中值代替原来的中心值
                img1[i, j] = median

        cv2.imwrite(tmp_path, img1)  # 将处理后的图像保存在临时路径中
        self.img_demo.setPixmap(QPixmap(tmp_path))  # 显示处理后的图像

    def gaussian_f_func(self):
        # 高斯滤波
        img = cv2.imread(tmp_path, 0)  # 读取临时路径中的图像
        src=img.copy()
        des = np.empty(src.shape, dtype=np.uint8)

        weights=np.array([[0.09474,0.11832,0.09474],[0.11832,0.14776,0.11832],[0.09474,0.11832,0.09474]])
        r=1
        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                count=0
                local_sum=0
                for ii in range(-r,r+1):
                    for jj in range(-r,r+1):
                        if (i+ii>0 and j+jj>0 and i+ii<src.shape[0] and j+jj<src.shape[1]):
                            local_sum+=weights[1+ii][1+jj]*src[i+ii][j+jj]
                            count+=1
                des[i][j]=local_sum

        cv2.imwrite(tmp_path,des) # 将处理后的图像保存在临时路径中
        #self.img_des.setPixmap(QPixmap(tmp_path)) # 显示处理后的图像
        self.img_demo.setPixmap(QPixmap(tmp_path))  # 显示处理后的图像

    def wiener_func(self):
        # wiener滤波
        img = cv2.imread(tmp_path, 0)  # 读取临时路径中的图像
        imgNoise = img.astype('float64')
        imgWiener = wiener(imgNoise, [3, 3])
        imgWiener = np.uint8(imgWiener / imgWiener.max() * 255)
        cv2.imwrite(tmp_path, imgWiener)  # 将处理后的图像保存在临时路径中
        self.img_demo.setPixmap(QPixmap(tmp_path))  # 显示处理后的图像


    def lowpass_func(self):
            # 低通滤波器
            img=cv2.imread(tmp_path,0) # 读取临时路径中的图像
            src=img.copy()
            des = np.empty(src.shape, dtype=np.uint8)
            isParaReady=True
            try: # 如果输入的参数有效，则以该参数处理图像
                p1=int(self.para1.toPlainText())
                d0=p1
            except Exception: # 否则，以默认的方式处理图像
                isParaReady=False

            if not isParaReady:
                d0=100 # 如果不能成功识别到参数，那么截止频率默认为100

            r_ext = np.zeros((src.shape[0] * 2, src.shape[1] * 2))
            for i in range(src.shape[0]):
                for j in range(src.shape[1]):
                    r_ext[i][j] = src[i][j]

            # 频域变换相关操作
            r_ext_fu = np.fft.fft2(r_ext)
            r_ext_fu = np.fft.fftshift(r_ext_fu)
            # 频率域中心坐标
            center = (r_ext_fu.shape[0] // 2, r_ext_fu.shape[1] // 2)
            h = np.empty(r_ext_fu.shape)
            # 绘制滤波器 H(u, v)
            for u in range(h.shape[0]):
                for v in range(h.shape[1]):
                    duv = ((u - center[0]) ** 2 + (v - center[1]) ** 2) ** 0.5
                    h[u][v] = duv < d0

            s_ext_fu = r_ext_fu * h
            s_ext = np.fft.ifft2(np.fft.ifftshift(s_ext_fu))
            s_ext = np.abs(s_ext)
            des = s_ext[0:src.shape[0], 0:src.shape[1]]

            for i in range(des.shape[0]):
                for j in range(des.shape[1]):
                    des[i][j] = min(max(des[i][j], 0), 255)
            cv2.imwrite(tmp_path,des) # 将处理后的图像保存在临时路径中
            self.img_demo.setPixmap(QPixmap(tmp_path)) # 显示处理后的图像



    def sobel_func(self):
        # 索伯尔
        img = cv2.imread(tmp_path, 0)  # 读取临时路径中的图像
        # 缩放
        img1 = cv2.resize(img, None, fx=0.7, fy=0.7)
        # cv2.imshow('img1', img1)

        # 计算X方向的梯度
        sobelx = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
        # 将复数转化为整数【绝对值函数】
        sobelx = cv2.convertScaleAbs(sobelx)
        # cv2.imshow('sobelx',sobelx)

        # 计算y方向的梯度
        sobely = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
        # 将复数转化为整数【绝对值函数】
        sobely = cv2.convertScaleAbs(sobely)
        # cv2.imshow('sobely',sobely)

        # 融合
        sobelxy1 = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        # cv2.imshow('sobelxy1',sobelxy1)

        # 直接计算融合的X和Y梯度
        sobelxy2 = cv2.Sobel(img1, cv2.CV_64F, 1, 1, ksize=3)
        # 将复数转化为整数【绝对值函数】
        sobelxy2 = cv2.convertScaleAbs(sobelxy2)
        # cv2.imshow('sobelxy2',sobelxy2)
        imgs = np.hstack([img1, sobelx, sobely, sobelxy1, sobelxy2])
        cv2.imshow('multi pic', imgs)
        sobel_img = cv2.resize(sobelxy1, None, fx=1.4, fy=1.4)
        cv2.imwrite(tmp_path, sobel_img)  # 将处理后的图像保存在临时路径中
        self.img_demo.setPixmap(QPixmap(tmp_path))  # 显示处理后的图像

    def laplace_func(self):
        # 拉普拉斯
        img = cv2.imread(tmp_path, 0)  # 读取临时路径中的图像
        img1 = cv2.resize(img, None, fx=1, fy=1)
        # cv2.imshow('img1', img1)
        new_img = cv2.Laplacian(img1, cv2.CV_64F)
        new_img = cv2.convertScaleAbs(new_img)
        imgs = np.hstack([img1, new_img])

        cv2.imwrite(tmp_path, new_img)  # 将处理后的图像保存在临时路径中
        self.img_demo.setPixmap(QPixmap(tmp_path))  # 显示处理后的图像
    def highpass_func(self):
        # 高通滤波器
        img = cv2.imread(tmp_path, 0)  # 读取临时路径中的图像
        src = img.copy()
        des = np.empty(src.shape, dtype=np.uint8)
        isParaReady = True
        try:  # 如果输入的参数有效，则以该参数处理图像
            p1 = int(self.para1.toPlainText())
            d0 = p1
        except Exception:  # 否则，以默认的方式处理图像
            isParaReady = False

        if not isParaReady:
            d0 = 100  # 如果不能成功识别到参数，那么截止频率默认为100

        r_ext = np.zeros((src.shape[0] * 2, src.shape[1] * 2))
        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                r_ext[i][j] = src[i][j]

        # 频域变换相关操作
        r_ext_fu = np.fft.fft2(r_ext)
        r_ext_fu = np.fft.fftshift(r_ext_fu)
        # 频率域中心坐标
        center = (r_ext_fu.shape[0] // 2, r_ext_fu.shape[1] // 2)
        h = np.empty(r_ext_fu.shape)
        # 绘制滤波器 H(u, v)
        for u in range(h.shape[0]):
            for v in range(h.shape[1]):
                duv = ((u - center[0]) ** 2 + (v - center[1]) ** 2) ** 0.5
                h[u][v] = np.e ** (-duv ** 2 / d0 ** 2)

        s_ext_fu = r_ext_fu * h
        s_ext = np.fft.ifft2(np.fft.ifftshift(s_ext_fu))
        s_ext = np.abs(s_ext)
        des = s_ext[0:src.shape[0], 0:src.shape[1]]

        for i in range(des.shape[0]):
            for j in range(des.shape[1]):
                des[i][j] = min(max(des[i][j], 0), 255)
        cv2.imwrite(tmp_path, des)  # 将处理后的图像保存在临时路径中
        self.img_demo.setPixmap(QPixmap(tmp_path))  # 显示处理后的图像

    def threshold_func(self):
        # 阈值法
        img = cv2.imread(tmp_path, 0)  # 读取临时路径中的图像
        img_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)

        cv2.imwrite(tmp_path, img_thresh)  # 将处理后的图像保存在临时路径中
        self.img_demo.setPixmap(QPixmap(tmp_path))  # 显示处理后的图像

    def regiongrow_func(self):
        #img = cv2.imread(tmp_path, 0)  # 读取临时路径中的图像
        im = cv2.imread(tmp_path, 0)  # 读取图片

        def region_growth():
     #       print("循环增长，直到len(region_points) = 0")
            count = 0
            x = [-1, 0, 1, -1, 1, -1, 0, 1]
            y = [-1, -1, -1, 0, 0, 1, 1, 1]
            while len(region_points) > 0:
                if count == 0:
                    point = region_points.pop(0)
                    i = point[0]
                    j = point[1]
         #        print("len = ", len(region_points))
                p_val = input_arr[i][j]
                # 像素强度差异范围 + - 8
                lt = p_val - 8
                ht = p_val + 8
                for k in range(8):
                    if seg_img[i + x[k]][j + y[k]] != 1:
                        try:
                            if lt < input_arr[i + x[k]][j + y[k]] < ht:
                                seg_img[i + x[k]][j + y[k]] = 1
                                p = [0, 0]
                                p[0] = i + x[k]
                                p[1] = j + y[k]
                                if p not in region_points:
                                    if 0 < p[0] < rows and 0 < p[1] < columns:
                                        # 满足条件的点
                                        region_points.append([i + x[k], j + y[k]])
                            else:
                                seg_img[i + x[k]][j + y[k]] = 0
                        except IndexError:
                            continue

                point = region_points.pop(0)
                i = point[0]
                j = point[1]
                count = count + 1

        input_img = Image.open(tmp_path).convert("L")  # 读取图片
        input_arr = np.asarray(input_img)
        rows, columns = np.shape(input_arr)

        plt.figure()
        plt.imshow(input_img)
        plt.gray()

      #  print("请选择初始点...")
        p_seed = plt.ginput(1)
        print(p_seed[0][0], p_seed[0][1])

        # 可以手动设置种子点
        # x = int(120)
        # y = int(160)
        x = int(p_seed[0][0])
        y = int(p_seed[0][1])
        seed_pixel = [x, y]

       # print("选择的点为：", seed_pixel)

        plt.close()
        seg_img = np.zeros((rows + 1, columns + 1))
        seg_img[seed_pixel[0]][seed_pixel[1]] = 255.0
        img_display = np.zeros((rows, columns))

        region_points = [[x, y]]
        region_growth()

        plt.imsave("result.jpg", seg_img)
        plt.figure()
#        plt.imshow(seg_img)
        #plt.colorbar()
        plt.show()
        cv2.imwrite(tmp_path, seg_img)  # 将处理后的图像保存在临时路径中
        self.img_demo.setPixmap(QPixmap("E:\imageprocess\\result.jpg"))  # 显示处理后的图像


    def actionCanny_func(self):
        # canny算子
        img = cv2.imread(tmp_path, 0)  # 读取临时路径中的图像
        edges = cv2.Canny(img, 100, 200)
        cv2.imwrite(tmp_path, edges)  # 将处理后的图像保存在临时路径中
        self.img_demo.setPixmap(QPixmap(tmp_path))  # 显示处理后的图像

    def actionKirsch_func(self):
        # Kirsch算子
        image = cv2.imread(tmp_path, 0)

        # 图片的高度和宽度
        h, w = image.shape[:2]
        print('imagesize={}-{}'.format(w, h))

        # 显示原图
   #     cv2.imshow("Image", image)

        # 定义Kirsch 卷积模板
        m1 = np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]])
        m2 = np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])
        m3 = np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]])
        m4 = np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])
        m5 = np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]])
        m6 = np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]])
        m7 = np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]])
        m8 = np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])
        list_m = [m1, m2, m3, m4, m5, m6, m7, m8]
        #
        list_e = []
        count = 1

        for m in list_m:
            imgk = signal.convolve2d(image, m, boundary='symm')
            list_e.append(np.abs(imgk))
            out = imgk.astype(np.uint8)
            # cv2.imshow("out{}".format(count),out)
            count += 1
        # 求最大值
        e = list_e[0]
        for i in range(len(list_e)):
            e = e * (e >= list_e[i]) + list_e[i] * (e < list_e[i])
        #
        e[e > 255] = 255
        e = e.astype(np.uint8)
        # cv2.imshow('e', e)
        cv2.imwrite(tmp_path, e)  # 将处理后的图像保存在临时路径中
        self.img_demo.setPixmap(QPixmap(tmp_path))  # 显示处理后的图像

    def inversion_func(self):
        # 图像反转

        img = cv2.imread(tmp_path)  # 读取临时路径中的图像
        img = 255 - img
        cv2.imwrite(tmp_path, img)  # 将处理后的图像保存在临时路径中
        self.img_demo.setPixmap(QPixmap(tmp_path))  # 显示处理后的图像

    def action_func1(self):
        #腐蚀
        img = cv2.imread(tmp_path)  # 读取临时路径中的图像
        kernel = np.ones((5, 5), np.uint8)
        # 腐蚀运算，iteration=1,迭代腐蚀1次
        erosion = cv2.erode(img, kernel, iterations=1)
        cv2.imwrite(tmp_path, erosion)  # 将处理后的图像保存在临时路径中
        self.img_demo.setPixmap(QPixmap(tmp_path))  # 显示处理后的图像
    def action_2_func(self):
        #膨胀
        img = cv2.imread(tmp_path)  # 读取临时路径中的图像
        kernel = np.ones((5, 5), np.uint8)
        # iteration=1,迭代膨胀1次
        dilation = cv2.dilate(img, kernel, iterations=1)
        cv2.imwrite(tmp_path, dilation)  # 将处理后的图像保存在临时路径中
        self.img_demo.setPixmap(QPixmap(tmp_path))  # 显示处理后的图像
    def action_3_func(self):
        #开运算
        img = cv2.imread(tmp_path)  # 读取临时路径中的图像
        kernel = np.ones((7, 7), np.uint8)
        """开运算"""
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        cv2.imwrite(tmp_path, opening)

        self.img_demo.setPixmap(QPixmap(tmp_path))  # 显示处理后的图像
    def action_4_func(self):
        #闭运算
        img = cv2.imread(tmp_path)  # 读取临时路径中的图像
        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        cv2.imwrite(tmp_path, closing)  # 将处理后的图像保存在临时路径中
        self.img_demo.setPixmap(QPixmap(tmp_path))  # 显示处理后的图像
    def action5_5_func(self):
        img = cv2.imread(tmp_path)  # 读取临时路径中的图像
        kernel = np.ones((7, 7), np.uint8)
   #     img = cv2.imread(r'F:\image\image\corn1.jpg')
  #      cv2.imshow('原图', img)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度处理
   #     cv2.imshow('gray_img', gray_img)
        ret, th1 = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY)
    #    cv2.imshow('th1', th1)
        erosion = cv2.erode(th1, kernel, iterations=1)  # 腐蚀
     #   cv2.imshow('erosion', erosion)
        dist_img = cv2.distanceTransform(erosion, cv2.DIST_L1, cv2.DIST_MASK_3)  # 距离变换
    #    cv2.imshow('距离变换', dist_img)
        dist_output = cv2.normalize(dist_img, 0, 1.0, cv2.NORM_MINMAX)  # 归一化
    #    cv2.imshow('dist_output', dist_output * 80)

        ret, th2 = cv2.threshold(dist_output * 80, 0.3, 255, cv2.THRESH_BINARY)
     #   cv2.imshow('th2', th2)

        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)
    #    cv2.imshow('opening', opening)
        opening = np.array(opening, np.uint8)
        contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓提取
        count = 0
        for cnt in contours:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            circle_img = cv2.circle(opening, center, radius, (255, 255, 255), 1)
            area = cv2.contourArea(cnt)
            area_circle = 3.14 * radius * radius
            # print(area/area_circle)
            if area / area_circle <= 0.1:
                img = cv2.drawContours(img, cnt, -1, (0,0,255), 5)#差（红色）
              #  img = cv2.putText(img, '', center, font, 0.5, (0, 0, 255))
            elif area / area_circle >= 0.2:
                img = cv2.drawContours(img, cnt, -1, (0,255,0), 5)#优（绿色）
              #  img = cv2.putText(img, 'big', center, font, 0.5, (0, 0, 255))
            else:
                img = cv2.drawContours(img, cnt, -1, (255,0,0), 5)#良（蓝色）
              #  img = cv2.putText(img, 'normal', center, font, 0.5, (0, 0, 255))
            count += 1
        img = cv2.putText(img, ('sum=' + str(count)), (50, 50), font, 1, (255, 0, 0))
        cv2.imshow('circle_img', img)
        key = 0
        while True:
            key = cv2.waitKey()
            if key == ord('a'):
                break
        # 销毁窗口
        cv2.destroyAllWindows()
        print('硬币共有：', count)
        cv2.imwrite(tmp_path, img)  # 将处理后的图像保存在临时路径中
        self.img_demo.setPixmap(QPixmap(tmp_path))  # 显示处理后的图像
if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = Demo()
    demo.show()
    # 销毁窗口
    cv2.destroyAllWindows()
    sys.exit(app.exec_())
