import sys
import pickle
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QAbstractScrollArea, QHeaderView, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QRect, QSize, QPoint


class MainWindow(QMainWindow):
   def __init__(self):
       super().__init__()
       try:
           self.docHash = {}
           self.metaData = {}
           with open('SamDocHash.pkl', 'rb') as f:
               data = pickle.load(f)
               self.docHash = data['docHash']
               self.metaData = data['metaData']
               docHash_loaded = True
       except Exception as e:
           print(f'Failure to load memory {str(e)}')
           sys.exit(-1)
       self.setWindowTitle("Document Display")
       self.setGeometry(300, 300, 800, 600)

       self.tableWidget = QTableWidget(self)
       size = QSize(800, 600)
       geometry = QRect(QPoint(), size)
       self.tableWidget.setGeometry(geometry)
       self.tableWidget.setColumnCount(4)
       self.tableWidget.setHorizontalHeaderLabels(["ID", "Tags", "Timestamp", "Embedding"])
       self.tableWidget.horizontalHeader().setStretchLastSection(True)
       self.tableWidget.setRowCount(len(self.docHash))
       # Allow window resizing
       self.setWindowFlags(Qt.WindowFlags() | Qt.WindowMaximizeButtonHint) 
       # Set resize mode on table
       self.tableWidget.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
       # Allow column resize
       self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
       self.tableWidget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
       
       idx = 0
       for id in self.docHash.keys():
           item = QTableWidgetItem(str(idx))
           self.tableWidget.setItem(idx, 0, item)

           item = QTableWidgetItem(self.docHash[id])
           self.tableWidget.setItem(idx, 1, item)

           item = QTableWidgetItem(str(self.metaData[id]["tags"]))
           self.tableWidget.setItem(idx, 2, item)

           item = QTableWidgetItem(str(self.metaData[id]["timestamp"]))
           self.tableWidget.setItem(idx, 3, item)

           idx += 1
       self.show()

if __name__ == "__main__":
   app = QApplication(sys.argv)
   window = MainWindow()
   sys.exit(app.exec_())
