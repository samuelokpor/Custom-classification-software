


from PyQt5 import QtCore, QtGui, QtWidgets, uic
import cv2
import os
import time
import uuid
from PyQt5.QtCore import Qt
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import numpy as np
import tensorflow as tf
import subprocess
from tensorflow import *
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
from matplotlib import pyplot as plt
from keras.metrics import Precision, Recall, BinaryAccuracy
from time import sleep
from tqdm.gui import trange, tqdm


class Ui_Dialog(QtWidgets.QWidget):

    def __init__(self, parent=None):
    
        super(Ui_Dialog, self).__init__()
        self.setupUi(Dialog)
        self.timer_camera = QtCore.QTimer(self)
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.slot_init()

    def slot_init(self):
        self.timer_camera.timeout.connect(self.show_camera)

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(836, 887)
        self.Img_window_1 = QtWidgets.QLabel(Dialog)
        self.Img_window_1.setGeometry(QtCore.QRect(30, 20, 381, 381))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.Img_window_1.setFont(font)
        self.Img_window_1.setMouseTracking(True)
        self.Img_window_1.setAutoFillBackground(True)
        self.Img_window_1.setAlignment(QtCore.Qt.AlignCenter)
        self.Img_window_1.setObjectName("Img_window_1")
        self.img_window_2 = QtWidgets.QLabel(Dialog)
        self.img_window_2.setGeometry(QtCore.QRect(30, 430, 381, 411))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.img_window_2.setFont(font)
        self.img_window_2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.img_window_2.setMouseTracking(True)
        self.img_window_2.setTabletTracking(True)
        self.img_window_2.setAutoFillBackground(True)
        self.img_window_2.setAlignment(QtCore.Qt.AlignCenter)
        self.img_window_2.setObjectName("img_window_2")
        self.classify_push_button = QtWidgets.QPushButton(Dialog)
        self.classify_push_button.setGeometry(QtCore.QRect(460, 760, 331, 91))
        self.classify_push_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.classify_push_button.setMouseTracking(True)
        self.classify_push_button.setObjectName("classify_push_button")
    
        self.groupBox_Train_function = QtWidgets.QGroupBox(Dialog)
        self.groupBox_Train_function.setGeometry(QtCore.QRect(500, 340, 241, 151))
        self.groupBox_Train_function.setObjectName("groupBox_Train_function")
        self.Load_model_button = QtWidgets.QPushButton(self.groupBox_Train_function)
        self.Load_model_button.setGeometry(QtCore.QRect(30, 30, 181, 23))
        self.Load_model_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.Load_model_button.setMouseTracking(True)
        self.Load_model_button.setObjectName("Load_model_button")
        self.save_model_button = QtWidgets.QPushButton(self.groupBox_Train_function)
        self.save_model_button.setGeometry(QtCore.QRect(30, 80, 181, 23))
        self.save_model_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.save_model_button.setMouseTracking(True)
        self.save_model_button.setTabletTracking(True)
        self.save_model_button.setObjectName("save_model_button")
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(450, 570, 111, 41))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(550, 670, 151, 41))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(Dialog)
        self.pushButton_3.setGeometry(QtCore.QRect(660, 570, 111, 41))
        self.pushButton_3.setObjectName("pushButton_3")
        self.results_label = QtWidgets.QLabel(Dialog)
        self.results_label.setGeometry(QtCore.QRect(450, 130, 351, 141))
        self.results_label.setAutoFillBackground(True)
        self.results_label.setAlignment(QtCore.Qt.AlignCenter)
        self.results_label.setObjectName("results_label")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

        #connecting buttons
        self.pushButton.clicked.connect(self.collect_images_fuction)
        self.pushButton_3.clicked.connect(self.capture)
        #self.pushButton_2.clicked.connect(self.readFile)
        self.Load_model_button.clicked.connect(self.load_data_to_keras)
        
        #self.save_model_button.clicked.connect(self.model_result)
        self.classify_push_button.clicked.connect(self.classify)
        self.pushButton_2.clicked.connect(self.print_accuracy)

    def show_camera(self):
            
        flag, self.imageopened = self.cap.read()
        show = cv2.resize(self.imageopened, (640, 480))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        self.showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.Img_window_1.setPixmap(QtGui.QPixmap.fromImage(self.showImage))

    def collect_images_fuction(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM, cv2.CAP_DSHOW)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"Connect-Camera-Source",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            # if msg==QtGui.QMessageBox.Cancel:
            #                     pass
            else:
                self.timer_camera.start(30)

                self.pushButton.setText(u'Camera Opened')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.Img_window_1.clear()
            self.pushButton.setText(u'Camera Closed')

    global IMG_PATH
    global labels
    global number_imgs
    global model

    model = Sequential()
    IMG_PATH = os.path.join('data')
    labels = ['ES2T_left_laxian', 'ES2T_left_lagan']
    number_imgs = 1


    def capture(self):
    
        if not os.path.exists(IMG_PATH):

            if os.name == 'posix':
                os.makedirs -p ({IMG_PATH})
            if os.name == 'nt':
            
                os.makedirs(IMG_PATH)
        for label in labels:

            path = os.path.join(IMG_PATH, label)
            if not os.path.exists(path):
                os.makedirs(path)

        for label in labels:
            #self.imageopened = self.cap.read()

            #cap = cv2.VideoCapture(0)
            print('Collecting images for {}'.format(label))
            time.sleep(5)
            for imgnum in range(number_imgs):



                print('Collecting image {}'.format(imgnum))
                #ret, frame = cap.read()
                imageopened = self.cap.read()
                print(imageopened)
                #FName = fr"IMG_PATH\cap{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
                imgname = os.path.join(IMG_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
                # print(imgname)
                # cv2.imwrite(imgname,imageopened)
                #cv2.imshow('frame', frame)
                time.sleep(1)
                self.showImage.save(imgname, "JPG", 100)

            self.capturedImage=self.show
            self.img_window_2.setPixmap(QtGui.QPixmap.fromImage(self.showImage))
            # self.showImage.save(imgname + ".jpg", "JPG", 100)


    global val
    global train1
    global test1
    global pre
    global re
    global acc
    global batch
    global hist

    data = tf.keras.utils.image_dataset_from_directory('data')
    #data = subprocess.Popen('data', shell=True)
    
    data_iterator = data.as_numpy_iterator()
    #print(data_iterator)
    batch = data_iterator.next()
    #print(batch[0].max())

    #scaling data into 0-1 
    data_scaled = data.map(lambda x,y: (x/255, y))
    scaled_data_iterator = data_scaled.as_numpy_iterator()
    batch_new = scaled_data_iterator.next()
    #print(batch_new[0].max())

    #splitting data
    #print(len(data_scaled))
    train_size = int(len(data)*.7)
    val_size = int(len(data)*.2)+1
    test_size = int(len(data)*.1)+1

    train1 = data.take(train_size)
    val  = data.skip(train_size).take(val_size)
    test1 = data.skip(train_size+val_size).take(test_size)
    print(test1)
    model = Sequential()

    model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
    model.add(MaxPooling2D())

    model.add(Conv2D(32, (3,3), 1, activation='relu' ))
    model.add(MaxPooling2D())

    model.add(Conv2D(16, (3,3), 1, activation='relu' ))
    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

    print(model.summary())

    logdir='logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    hist = model.fit(train1, epochs=7, validation_data=val, callbacks=[tensorboard_callback])
        # #subprocess.run('python hist', shell=True)
    print(hist.history)


    

    def load_data_to_keras(self):
        
        fig = plt.figure()
        plt.plot(hist.history['loss'], color='teal', label='loss')
        plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
        fig.suptitle('Loss', fontsize=20)
        plt.legend(loc="upper left")
        plt.show()

        fig1 = plt.figure()
        plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
        plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
        fig1.suptitle('Accuracy', fontsize=20)
        plt.legend(loc="upper left")
        plt.show()


    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()


    def print_accuracy(self):
         
        for batch in test1.as_numpy_iterator():
            X, y = batch
            yhat = model.predict(X)
            pre.update_state(y, yhat)
            re.update_state(y, yhat)
            acc.update_state(y, yhat)
        pe = QPalette()
        pe.setColor(QPalette.WindowText, Qt.red)
        self.results_label.setPalette(pe)
        self.results_label.setFont(QFont("Roman times", 20, QFont.Bold))
        self.results_label.setText(f'Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')
        

        #print(f'Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')
        
        #print(yhat)


    def classify(self, yhat):
        img,ftype = QFileDialog.getOpenFileName(self, "Open File", "./", "All Files(*);;Wav(*.wav);;Txt (*.txt)")
            
        self.imageopened=cv2.imread(r''.join(img))
        #self.img_window_2.setPixmap(QPixmap(img))
        #self.img_window_2.setScaledContents(True)
        #IMG_PATH = os.path.join('test','MER-230-168U3M(NG0190080304)_2022-11-25_16_01_03_484-275.jpg')
        #img  = cv2.imread(IMG_PATH)
        #plt.imshow(img)
        #plt.show()
        self.resized = tf.image.resize(self.imageopened, (256,256))
        #plt.imshow(self.resized.numpy().astype(int))
        #plt.show()
        #print(self.resized.shape)
        #print(np.expand_dims(self.resized, 0).shape)
        yhat = model.predict(np.expand_dims(self.resized/255, 0))
        #print(yhat)
        if yhat > 0.5:
            
           cv2.putText(self.imageopened, f'Predicted class is ES2T_left_laxian', (10, 60), 1, 3, (0, 255, 0), 3)
           pass
           showscreen = cv2.resize(self.imageopened, (640, 460))
           showscreen = cv2.cvtColor(self.imageopened, cv2.COLOR_BGR2RGB)
           self.output = QtGui.QImage(showscreen.data, showscreen.shape[1], showscreen.shape[0], QtGui.QImage.Format_RGB888)
           self.img_window_2.setPixmap(QtGui.QPixmap.fromImage(self.output))
           self.img_window_2.setScaledContents(True)
           pe = QPalette()
           pe.setColor(QPalette.WindowText, Qt.red)
           self.results_label.setPalette(pe)
           self.results_label.setFont(QFont("Roman times", 20, QFont.Bold))
           self.results_label.setText('ES2T_left_laxian')
              
          
        elif yhat < 0.5:
            

            cv2.putText(self.imageopened, f'Predicted class is ES2T_left_lagan', (10, 60), 1, 3, (255, 0, 0), 3)
            pass
            showscreen2 = cv2.resize(self.imageopened, (540, 460))
            showscreen2 = cv2.cvtColor(self.imageopened, cv2.COLOR_BGR2RGB)
            self.output2 = QtGui.QImage(showscreen2.data, showscreen2.shape[1], showscreen2.shape[0], QtGui.QImage.Format_RGB888)
            self.img_window_2.setPixmap(QtGui.QPixmap.fromImage(self.output2))
            self.img_window_2.setScaledContents(True)
            pe = QPalette()
            pe.setColor(QPalette.WindowText, Qt.red)
            self.results_label.setPalette(pe)
            self.results_label.setFont(QFont("Roman times", 20, QFont.Bold))
            self.results_label.setText('ES2T_left_lagan')
        else: 
            print(f'Null')
  





    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.Img_window_1.setText(_translate("Dialog", "IMAGE1"))
        self.img_window_2.setText(_translate("Dialog", "IMAGE2"))
        self.classify_push_button.setText(_translate("Dialog", "Classify"))
        self.groupBox_Train_function.setTitle(_translate("Dialog", "Train_Function"))
        self.Load_model_button.setText(_translate("Dialog", "Plot Performance Metrics"))
        self.save_model_button.setText(_translate("Dialog", "Model Results"))
        self.pushButton.setText(_translate("Dialog", "Open Camera"))
        self.pushButton_2.setText(_translate("Dialog", "Open Folder"))
        self.pushButton_3.setText(_translate("Dialog", "Capture"))
        self.results_label.setText(_translate("Dialog", "RESULTS"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
