from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from distutils.util import strtobool
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QCheckBox
from PyQt5.QtCore import QSettings
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QLineEdit
import sys

class Main(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)          
        # self.pushButton.clicked.connect(self.click_button)
        # self.pushButton.clicked.connect(self.clickButton())
        # self.pushButton.clicked.connect(self.algo())

    def setupUi(self, Cryptara):
        if not Cryptara.objectName():
            Cryptara.setObjectName(u"Cryptara")
        Cryptara.resize(714, 518)
        self.widget = QWidget(Cryptara)
        self.widget.setObjectName(u"widget")
        self.pushButton = QPushButton(self.widget)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(70, 260, 121, 51))
        self.pushButton_2 = QPushButton(self.widget)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setGeometry(QRect(220, 260, 121, 51))
        self.pushButton_3 = QPushButton(self.widget)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setGeometry(QRect(370, 260, 121, 51))
        self.pushButton_4 = QPushButton(self.widget)
        self.pushButton_4.setObjectName(u"pushButton_4")
        self.pushButton_4.setGeometry(QRect(520, 260, 121, 51))
        self.pushButton_4.setCursor(QCursor(Qt.ForbiddenCursor))
        self.label = QLabel(self.widget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(-4, -7, 721, 531))
        self.label.setPixmap(QPixmap(u"../../Users/asant/Desktop/TaraUI/IMG_8841.JPG"))
        self.label.setScaledContents(True)
        self.label_2 = QLabel(self.widget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(40, 10, 131, 51))
        self.label_2.setPixmap(QPixmap(u"../../Users/asant/Desktop/TaraUI/unknown.png"))
        self.label_2.setScaledContents(True)
        Cryptara.setCentralWidget(self.widget)
        self.label.raise_()
        self.pushButton.raise_()
        self.pushButton_2.raise_()
        self.pushButton_3.raise_()
        self.pushButton_4.raise_()
        self.label_2.raise_()
        self.menubar = QMenuBar(Cryptara)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 714, 22))
        Cryptara.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(Cryptara)
        self.statusbar.setObjectName(u"statusbar")
        Cryptara.setStatusBar(self.statusbar)

        self.retranslateUi(Cryptara)

        QMetaObject.connectSlotsByName(Cryptara)

    
    # def setupUi(self, MainWindow):
    #     MainWindow.setObjectName("MainWindow")
    #     MainWindow.resize(519, 354)
    #     self.centralwidget = QtWidgets.QWidget(MainWindow)
    #     self.centralwidget.setObjectName("centralwidget")
    #     self.pushButton = QtWidgets.QPushButton(self.centralwidget)
    #     self.pushButton.setGeometry(QtCore.QRect(300, 220, 150, 100))
    #     self.pushButton.setObjectName("pushButton")
    #     MainWindow.setCentralWidget(self.centralwidget)
    #     self.retranslateUi(MainWindow)
    #     QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Run-Websocket"))
        self.pushButton.clicked.connect(self.results)

    
    def algo(self):
        import algo

    
    def results(self):
        import results
        results.print_results()


# class Ui_Cryptara(object):
#     def setupUi(self, Cryptara):
#         if not Cryptara.objectName():
#             Cryptara.setObjectName(u"Cryptara")
#         Cryptara.resize(714, 518)
        self.widget = QWidget(Cryptara)
        self.widget.setObjectName(u"widget")
        self.pushButton = QPushButton(self.widget)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(70, 260, 121, 51))
        self.pushButton_2 = QPushButton(self.widget)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setGeometry(QRect(220, 260, 121, 51))
        self.pushButton_3 = QPushButton(self.widget)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setGeometry(QRect(370, 260, 121, 51))
        self.pushButton_4 = QPushButton(self.widget)
        self.pushButton_4.setObjectName(u"pushButton_4")
        self.pushButton_4.setGeometry(QRect(520, 260, 121, 51))
        self.pushButton_4.setCursor(QCursor(Qt.ForbiddenCursor))
        self.label = QLabel(self.widget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(-4, -7, 721, 531))
        self.label.setPixmap(QPixmap(u"../../Users/asant/Desktop/TaraUI/IMG_8841.JPG"))
        self.label.setScaledContents(True)
        self.label_2 = QLabel(self.widget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(40, 10, 131, 51))
        self.label_2.setPixmap(QPixmap(u"../../Users/asant/Desktop/TaraUI/unknown.png"))
        self.label_2.setScaledContents(True)
        Cryptara.setCentralWidget(self.widget)
        self.label.raise_()
        self.pushButton.raise_()
        self.pushButton_2.raise_()
        self.pushButton_3.raise_()
        self.pushButton_4.raise_()
        self.label_2.raise_()
        self.menubar = QMenuBar(Cryptara)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 714, 22))
        Cryptara.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(Cryptara)
        self.statusbar.setObjectName(u"statusbar")
        Cryptara.setStatusBar(self.statusbar)

        self.retranslateUi(Cryptara)

        QMetaObject.connectSlotsByName(Cryptara)



def main():
    app = QApplication(sys.argv)
    win = Main()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
