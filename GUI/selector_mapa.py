import os
from PyQt6.QtWidgets import (
    QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QLabel,
    QPushButton, QSpacerItem, QSizePolicy
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, Qt

from .interfazV import VentanaTerreno  # Asegúrate que el import esté correcto

class Bridge(QObject):
    coordenadas_recibidas = pyqtSignal(float, float)

    @pyqtSlot(float, float)
    def recibirCoordenadas(self, lat, lon):
        self.coordenadas_recibidas.emit(lat, lon)

class SelectorMapa(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Selector de punto en el mapa")
        self.resize(1100, 700)

        main_layout = QHBoxLayout()

        self.webview = QWebEngineView()
        main_layout.addWidget(self.webview, 3)

        panel = QWidget()
        panel_layout = QVBoxLayout()
        panel.setLayout(panel_layout)
        panel.setMaximumWidth(300)
        main_layout.addWidget(panel, 1)

        panel_layout.addWidget(QLabel("📍 Coordenadas seleccionadas:"))
        self.coord_label = QLabel("Lat: --\nLon: --")
        self.coord_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        panel_layout.addWidget(self.coord_label)

        panel_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        self.btn_abrir = QPushButton("Abrir Visualizador 3D")
        self.btn_abrir.setEnabled(False)
        panel_layout.addWidget(self.btn_abrir)

        panel_layout.addSpacerItem(QSpacerItem(20, 200, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.bridge = Bridge()
        self.bridge.coordenadas_recibidas.connect(self.actualizar_coords)

        self.channel = QWebChannel()
        self.channel.registerObject("backend", self.bridge)
        self.webview.page().setWebChannel(self.channel)

        # Tu API key aquí:
        api_key = "AIzaSyAHlWznhkiHXHrfuej9tamoe0G4FftZqZ8"

        # Ruta absoluta del archivo HTML
        ruta_html = os.path.abspath(os.path.join(os.path.dirname(__file__), "mapa_google.html"))

        # Leer el archivo HTML
        with open(ruta_html, 'r', encoding='utf-8') as f:
            html = f.read()

        # Reemplazar la variable {{API_KEY}} por tu api_key
        html = html.replace("{{API_KEY}}", api_key)

        # Cargar el HTML modificado en el QWebEngineView
        self.webview.setHtml(html)

        self.lat_sel = None
        self.lon_sel = None

        self.btn_abrir.clicked.connect(self.abrir_ventana_terreno)

    def actualizar_coords(self, lat, lon):
        self.lat_sel = lat
        self.lon_sel = lon
        self.coord_label.setText(f"Lat: {lat:.6f}\nLon: {lon:.6f}")
        self.btn_abrir.setEnabled(True)

    def abrir_ventana_terreno(self):
        if self.lat_sel is None or self.lon_sel is None:
            self.coord_label.setText("❗ Debes seleccionar un punto en el mapa.")
            return
        self.ventana_terreno = VentanaTerreno(lat=self.lat_sel, lon=self.lon_sel)
        self.ventana_terreno.show()
        self.close()
