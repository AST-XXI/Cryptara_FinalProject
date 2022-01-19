#Note: Websocket functionality only works with active TradingView & Twitter Developer account.
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from distutils.util import strtobool
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenuBar, QLabel, QPushButton, QStatusBar
from PyQt5.QtCore import QSettings, QMetaObject
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QCursor, QPixmap
from PyQt5.QtCore import QRect
from PyQt5.QtWidgets import QWidget, QLineEdit
import sys

class Main(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)          
    
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(519, 354)
        self.widget = QWidget(MainWindow)
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
        self.label = QLabel(self.widget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(-4, -7, 721, 531))
        self.label.setPixmap(QPixmap(u"graphics/IMG_8841.JPG"))
        self.label.setScaledContents(True)
        self.label_2 = QLabel(self.widget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(40, 10, 131, 51))
        self.label_2.setPixmap(QPixmap(u"graphics/unknown.png"))
        self.label_2.setScaledContents(True)
        MainWindow.setCentralWidget(self.widget)
        self.label.raise_()
        self.pushButton.raise_()
        self.pushButton_2.raise_()
        self.pushButton_3.raise_()
        self.label_2.raise_()
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 714, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("Cryptara", u"MainWindow", None))
        self.pushButton.setText(QCoreApplication.translate("Cryptara", u"Run Algorithm", None))
        self.pushButton.clicked.connect(self.market_analysis)
        self.pushButton_2.setText(QCoreApplication.translate("Cryptara", u"TaraSwap", None))
        self.pushButton_2.clicked.connect(self.blockchain)
        self.pushButton_3.setText(QCoreApplication.translate("Cryptara", u"Run Websocket", None))
        self.label.setText("")
        self.label_2.setText("")


    def market_analysis(self):
        import _results
        _results.print_results()

    def blockchain(self):
        import webbrowser
        url = 'https://ast-xxi.github.io/Cryptara_FinalProject/'
        webbrowser.open_new_tab(url)

def main():
    app = QApplication(sys.argv)
    win = Main()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
