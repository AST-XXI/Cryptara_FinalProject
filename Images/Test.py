# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'designerYLdaGL.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_Cryptara(object):
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
    # setupUi

    def retranslateUi(self, Cryptara):
        Cryptara.setWindowTitle(QCoreApplication.translate("Cryptara", u"MainWindow", None))
        self.pushButton.setText(QCoreApplication.translate("Cryptara", u"Cryptara Algo", None))
        self.pushButton_2.setText(QCoreApplication.translate("Cryptara", u"TaraSwap", None))
        self.pushButton_3.setText(QCoreApplication.translate("Cryptara", u"Sentiment Analysis", None))
        self.pushButton_4.setText(QCoreApplication.translate("Cryptara", u"NFT MarketPlace", None))
        self.label.setText("")
        self.label_2.setText("")
    # retranslateUi
