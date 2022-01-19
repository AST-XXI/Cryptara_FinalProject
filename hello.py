from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys



class Main(QMainWindow):
    def __init__(self):
        super().__init__()
class MyWindow(QMainWindow): 
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setGeometry(200,200,300,300)  #(xpos,ypos,width,height)
        self.setWindowTitle('PyQt5 Tutorial')
        self.initUI()
    def initUI(self):
        self.label = QtWidgets.QLabel(self)
        self.label.setText('My First Label')
        self.label.move(50,50) #X and y position
        self.b1 = QtWidgets.QPushButton(self)
        self.b1.setText('Click Me')
        self.b1.clicked.connect(self.clicked)
        
    def clicked(self):
        self.label.setText('You Pressed the Button.')
        self.update()
    def update(self):
        self.label.adjustSize()
def clicked():
    print('clicked')
def window():
    app=QApplication(sys.argv)
    win = MyWindow()
    win.show()
    sys.exit(app.exec_())
window()