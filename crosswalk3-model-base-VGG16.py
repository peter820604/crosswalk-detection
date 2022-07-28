#!/usr/bin/python3

import sys, os
if hasattr(sys, 'frozen'):
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']
import cv2
from PyQt5 import QtCore, QtWidgets,QtGui
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QDialog,
                             QFileDialog, QGridLayout, QLabel, QPushButton, QInputDialog)
from pylab import *
import matplotlib
matplotlib.use('Qt5Agg')
import sys


class win(QtWidgets.QDialog):
    def __init__(self,parent=None):

        # 初始二個img的ndarray用於存儲圖像
        self.img = np.ndarray(())
        self.img2 = np.ndarray(())
        self.fileName = ''
        self.Name = ''

        super(win, self).__init__(parent)
        self.initUI()

    def initUI(self):
        self.resize(500, 500)
        self.setWindowTitle('AIP 60847089s')
        self.btnOpen = QPushButton('選擇影像', self)
        self.btndetection = QPushButton('偵測', self)

        self.label = QLabel()
        self.label.setFixedWidth(500)
        self.label.setFixedHeight(500)

        self.label2 = QLabel()
        self.label2.setFixedWidth(500)
        self.label2.setFixedHeight(500)

        self.label3 = QLabel()


        # 佈局
        layout = QGridLayout(self)
        layout.addWidget(self.label, 1, 1, 4, 4)
        layout.addWidget(self.label2, 1, 5, 4, 4)
        layout.addWidget(self.label3, 0, 5, 1, 1)
        layout.addWidget(self.btnOpen, 0, 1)
        layout.addWidget(self.btndetection, 0, 2)

        # 連接
        self.btnOpen.clicked.connect(self.openSlot)
        self.btndetection.clicked.connect(self.detection)


    def openSlot(self):

        self.img = np.ndarray(())
        self.img2 = np.ndarray(())

        self.fileName, tmp = QFileDialog.getOpenFileName(self,"打開影像","","*.jpg;;*.bmp;;*.ppm;;All Files (*)")
        if self.fileName is '':
            return
        self.img = cv2.imdecode(np.fromfile(self.fileName, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        self.img = cv2.resize(self.img, (512, 512), interpolation=cv2.INTER_CUBIC)

        self.img2 = self.img.copy()

        self.refreshShow()

        # 每次跑圖前清空
        self.label2.clear()
        self.label3.clear()


    def detection(self):
        from keras.preprocessing import image
        from keras.applications.vgg16 import preprocess_input, decode_predictions
        from keras import backend as K
        import numpy as np
        from keras.models import load_model
        import cv2

        if self.fileName is '':
            return

        model = load_model("crosswalk3-model-base-VGG16.h5")
        img_path = self.fileName
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        self.Name = features[0][0]
        aa2 = round(self.Name,10)
        aa = str(round(self.Name,10))
        self.label3.setText('機率: '+ aa)

        if aa2 >= 0.05:
            african_elephant_output = model.output[:, 0]
            last_conv_layer = model.get_layer('block5_conv3')
            grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
            pooled_grads = K.mean(grads, axis=(0, 1, 2))
            iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
            pooled_grads_value, conv_layer_output_value = iterate([x])

            for i in range(512):
                conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
            heatmap = np.mean(conv_layer_output_value, axis=-1)
            heatmap = np.maximum(heatmap, 0)  # heatmap與0比較，取其大者
            heatmap /= np.max(heatmap)
            img = cv2.imread(img_path)
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = heatmap
            width = superimposed_img.shape[0]  # 获取宽度
            height = superimposed_img.shape[1]

            aa1 = np.ndarray(())
            aa1 = superimposed_img

            for x in range(width):
                for y in range(height):
                    if superimposed_img[x, y][0] == 255 and superimposed_img[x, y][1] >= 200:
                        superimposed_img[x, y] = 255
                    else:
                        superimposed_img[x, y] = 0

            image = aa1

            imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(imgray, 127, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), -1)

            img2 = np.copy(img)

            imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(imgray, 127, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)


            self.img3 = img2
            self.refreshShow2()
        else:
            self.refreshShow3()


    def refreshShow(self):
        #提取圖像的尺寸和通道, 用於將opencv下的image轉換成Qimage
        height, width, channel = self.img2.shape
        bytesPerLine = 3 * width
        self.qImg = QImage(self.img2.data, width, height, bytesPerLine,QImage.Format_RGB888).rgbSwapped()

        # 將Qimage顯示出來
        self.label.setPixmap(QPixmap.fromImage(self.qImg))
        # 將Qimage調整為適合label大小
        self.label.setScaledContents(True)

    def refreshShow2(self):
        #提取圖像的尺寸和通道, 用於將opencv下的image轉換成Qimage
        height, width, channel = self.img3.shape
        bytesPerLine = 3 * width
        self.qImg2 = QImage(self.img3.data, width, height, bytesPerLine,QImage.Format_RGB888).rgbSwapped()

        # 將Qimage顯示出來
        self.label2.setPixmap(QPixmap.fromImage(self.qImg2))
        # 將Qimage調整為適合label大小
        self.label2.setScaledContents(True)

    def refreshShow3(self):
        #提取圖像的尺寸和通道, 用於將opencv下的image轉換成Qimage
        height, width, channel = self.img2.shape
        bytesPerLine = 3 * width
        self.qImg3 = QImage(self.img2.data, width, height, bytesPerLine,QImage.Format_RGB888).rgbSwapped()

        # 將Qimage顯示出來
        self.label2.setPixmap(QPixmap.fromImage(self.qImg3))
        # 將Qimage調整為適合label大小
        self.label2.setScaledContents(True)



if __name__ == '__main__':
    a = QApplication(sys.argv)
    w = win()
    w.show()
    sys.exit(a.exec_())