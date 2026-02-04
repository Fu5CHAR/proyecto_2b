import sys

from PyQt6.QtWidgets import QApplication
from GUI.selector_mapa import SelectorMapa  

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QWidget {
            background-color: #121212;
            color: #e0e0e0;
            font-family: Arial;
        }

        QPushButton {
            background-color: #2962ff;
            color: white;
            border-radius: 10px;
            padding: 8px;
            font-size: 13px;
        }

        QPushButton:hover {
            background-color: #448aff;
        }

        QLineEdit {
            background-color: #1f1f1f;
            border: 1px solid #333;
            border-radius: 6px;
            padding: 5px;
        }

        QComboBox {
            background-color: #1f1f1f;
            border-radius: 6px;
            padding: 5px;
        }
        """)
    selector = SelectorMapa()
    selector.show()
    sys.exit(app.exec())

# AIzaSyAHlWznhkiHXHrfuej9tamoe0G4FftZqZ8
