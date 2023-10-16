import pickle
import json
from PyQt5.QtWidgets import QApplication, QWidget, QTextEdit, QPushButton, QVBoxLayout

class Editor(QWidget):

    def __init__(self):
        super().__init__()
        self.filename = "Sam.pkl"
        self.load()
        layout = QVBoxLayout()
        self.edit = QTextEdit()
        string_history = [json.dumps(d) for d in self.history]
        self.edit.setPlainText("\n".join(string_history))
        layout.addWidget(self.edit)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save)        
        layout.addWidget(save_btn)

        self.setLayout(layout)

    def load(self):
        with open(self.filename, "rb") as f:
            data = pickle.load(f)
            self.history = data['history']

    def save(self):
        edited_text = self.edit.toPlainText()
        history_strings = edited_text.split("\n")
        # Convert back to dicts
        history_dicts = [json.loads(s) for s in history_strings]
        # Save updated history 
        data = {}
        data['history'] = history_dicts
        with open(self.filename, 'wb') as f:
            pickle.dump(data, f) 

app = QApplication([])
editor = Editor() 
editor.show()
app.exec()
