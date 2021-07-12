import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, QInputDialog)
from PyQt5.QtCore import QCoreApplication
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
import numpy as np

class MyApp(QWidget):
    result=''
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        id = QPushButton('[ Student ID : 1912062 ]')
        name = QPushButton('[ Name : 변우진 ]')
        btn1 = QPushButton('1. Titanic Survivor Predictor', self)
        btn2 = QPushButton('2. Market Basket Analyzer', self)
        btn3 =  QPushButton('3. Quit', self)

        id.setStyleSheet("border-style: none;"
                             "text-align: left")
        name.setStyleSheet("border-style: none;"
                             "text-align: left")

        btn1.setStyleSheet("border-style: none;"
                             "text-align: left")
        btn2.setStyleSheet("border-style: none;"
                             "text-align: left")
        btn3.setStyleSheet("border-style: none;"
                             "text-align: left")


        vbox1 = QVBoxLayout()
        vbox1.addWidget(id)
        vbox1.addWidget(name)
        vbox1.addWidget(btn1)
        vbox1.addWidget(btn2)
        vbox1.addWidget(btn3)
        

        vbox = QVBoxLayout()
        vbox.addLayout(vbox1)
        self.setLayout(vbox)
        self.setStyleSheet("background-color:white")


        btn1.clicked.connect(self.resizeBig)
        btn2.clicked.connect(self.showDialog)
        btn3.clicked.connect(QCoreApplication.instance().quit)

        self.setWindowTitle('Data Mining')
        self.setGeometry(500, 200, 450, 300)
        self.show()

   
    def resizeBig(self):
        exec(open('lab01.py').read())

    def showDialog(self):
        text, ok = QInputDialog.getText(self, 'Input Dialog', '실행 결과는 터미널 창을 통해 확인하실 수 있습니다.\n\nEnter the mininumm support')
        
        if ok:
            num=float(text)
            dataset =pd.read_csv('Market_Basket_Optimisation.csv',header=None)
            data = []
            for i in range(0,len(dataset)):
                items = [str(dataset.values[i,j]) for j in range(0, len(dataset.columns))]
                items = [item for item in items if str(item) != 'nan']
                data.append(items)

            data = np.array(data)

            te = TransactionEncoder()
            te_ary = te.fit(data).transform(data)
            df = pd.DataFrame(te_ary, columns = te.columns_)
            result=fpgrowth(df, min_support=num, use_colnames=True)
            print("Frequent Set")
            print(result)
           

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
