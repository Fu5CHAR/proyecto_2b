import sys
import subprocess
from pathlib import Path
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
     QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QHBoxLayout, QLabel, QLineEdit, QGridLayout, QSizePolicy,
    QSpacerItem, QComboBox
)
import pyvista as pv
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from pyvistaqt import QtInteractor
from scipy.ndimage import gaussian_filter

from Motor.etiquetador import etiquetas_en_cuadrante
from Motor.Visualizador import cargar_datos, obtener_pixel, recortar_cuadrante, crear_malla, generar_sombreado

class VentanaTerreno(QMainWindow):

    def __init__(self, lat=-0.226, lon=-78.516, direccion="NORTE", size=1000):
        super().__init__()
        self.setWindowTitle("Visualizador Terreno 3D con PyQt6 y PyVista")
        self.resize(1100, 700)

        self.lat = lat
        self.lon = lon
        self.direccion = direccion
        self.size = size

        self.altura_ojos = 1.7
        self.altura_extra = 10
        self.distancia_focal = 100

        self.angulos = {
            "NORTE": 0,
            "ESTE": np.pi / 2,
            "SUR": np.pi,
            "OESTE": 3 * np.pi / 2
        }
        self.angulo = self.angulos.get(self.direccion, 0)

        self.data, self.transform = cargar_datos()

        self.frame = QWidget()
        self.setCentralWidget(self.frame)
        self.main_layout = QHBoxLayout()
        self.frame.setLayout(self.main_layout)

        self.plotter = QtInteractor(self.frame)
        self.plotter.set_background("black")
        self.plotter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.main_layout.addWidget(self.plotter, stretch=3)

        self.panel = QWidget()
        self.panel.setMaximumWidth(250)
        self.panel_layout = QVBoxLayout()
        self.panel.setLayout(self.panel_layout)
        self.main_layout.addWidget(self.panel, stretch=1)

        self.panel_layout.addWidget(QLabel("Latitud:"))
        self.lat_input = QLineEdit(str(self.lat))
        self.panel_layout.addWidget(self.lat_input)

        self.panel_layout.addWidget(QLabel("Longitud:"))
        self.lon_input = QLineEdit(str(self.lon))
        self.panel_layout.addWidget(self.lon_input)

        self.panel_layout.addWidget(QLabel("Tamaño (size):"))
        self.size_input = QLineEdit(str(self.size))
        
        # ---------------------------
        # Control: exageración vertical
        # ---------------------------
        self.panel_layout.addWidget(QLabel("Exageración vertical (x):"))
        self.exag_input = QLineEdit("1.5")   # valor por defecto 1.5 (ajústalo)
        self.panel_layout.addWidget(self.exag_input)

        # ---------------------------
        # Control: calidad STL (submuestreo / objetivo triángulos)
        # ---------------------------
        self.panel_layout.addWidget(QLabel("Calidad STL:"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["Alta", "Media", "Baja"])
        self.quality_combo.setCurrentText("Media")
        self.panel_layout.addWidget(self.quality_combo)

        self.panel_layout.addWidget(self.size_input)

        # Desplegable de dirección
        self.panel_layout.addWidget(QLabel("Dirección:"))
        self.direccion_combo = QComboBox()
        self.direccion_combo.addItems(["NORTE", "ESTE", "SUR", "OESTE"])
        self.direccion_combo.setCurrentText(self.direccion)
        self.direccion_combo.currentTextChanged.connect(self.cambiar_direccion)
        self.panel_layout.addWidget(self.direccion_combo)

        # Nueva etiqueta de altitud
        self.altitud_label = QLabel("Altitud: --")
        self.panel_layout.addWidget(self.altitud_label)

        self.panel_layout.addSpacing(20)

        cruz_widget = QWidget()
        cruz_layout = QGridLayout()
        cruz_widget.setLayout(cruz_layout)

        self.btn_arriba = QPushButton("↑")
        self.btn_abajo = QPushButton("↓")
        self.btn_izquierda = QPushButton("←")
        self.btn_derecha = QPushButton("→")
        self.btn_reset = QPushButton("⟳")

        for btn in [self.btn_arriba, self.btn_abajo, self.btn_izquierda, self.btn_derecha, self.btn_reset]:
            btn.setFixedSize(40, 40)

        cruz_layout.addWidget(self.btn_arriba, 0, 1)
        cruz_layout.addWidget(self.btn_izquierda, 1, 0)
        cruz_layout.addWidget(self.btn_reset, 1, 1)
        cruz_layout.addWidget(self.btn_derecha, 1, 2)
        cruz_layout.addWidget(self.btn_abajo, 2, 1)

        self.panel_layout.addWidget(cruz_widget)
        self.panel_layout.addSpacing(30)

        self.btn_ir = QPushButton("Ir")
        self.btn_ir.setFixedHeight(40)
        self.panel_layout.addWidget(self.btn_ir)

        # BOTÓN REGRESAR
        self.btn_regresar = QPushButton("Regresar")
        self.btn_regresar.setFixedHeight(40)
        self.panel_layout.addWidget(self.btn_regresar)

        self.panel_layout.addItem(QSpacerItem(20, 40))

        
        self.btn_export = QPushButton("Exportar STL")
        self.btn_export.setFixedHeight(40)
        self.btn_export.setEnabled(False)  # se habilita cuando haya malla lista
        self.panel_layout.addWidget(self.btn_export)

        # Conectar señales
        self.btn_arriba.pressed.connect(lambda: self.start_repeating(self.mover_arriba))
        self.btn_arriba.released.connect(self.stop_repeating)

        self.btn_abajo.pressed.connect(lambda: self.start_repeating(self.mover_abajo))
        self.btn_abajo.released.connect(self.stop_repeating)

        self.btn_izquierda.pressed.connect(lambda: self.start_repeating(self.mover_izquierda))
        self.btn_izquierda.released.connect(self.stop_repeating)

        self.btn_derecha.pressed.connect(lambda: self.start_repeating(self.mover_derecha))
        self.btn_derecha.released.connect(self.stop_repeating)

        self.btn_reset.clicked.connect(self.reset_camara)
        self.btn_ir.clicked.connect(self.regenerar_malla)
        self.btn_regresar.clicked.connect(self.regresar_a_selector)

        self.btn_export.clicked.connect(self.exportar_stl)


        self.repeat_timer = QTimer()
        self.repeat_timer.setInterval(50)
        self.repeat_timer.timeout.connect(self.repeat_action)
        self.action_to_repeat = None

        self.regenerar_malla()


    def build_solid_from_grid(self, z_grid, dx, dy, floor_thickness=None):
        """
        Construye un sólido cerrado (pyvista.PolyData) a partir de z_grid (2D numpy array),
        con top, bottom y paredes. Unidades: metros.
        dx, dy: tamaño real por celda en metros.
        """
        ny, nx = z_grid.shape

        xs = np.arange(nx) * dx
        ys = np.arange(ny) * dy

        # puntos superiores (top)
        top_points = np.zeros((ny * nx, 3), dtype=float)
        idx = 0
        for i in range(ny):
            for j in range(nx):
                top_points[idx, 0] = xs[j]
                top_points[idx, 1] = ys[i]
                top_points[idx, 2] = float(z_grid[i, j])
                idx += 1

        # base (z mínimo menos grosor)
        zmin = float(np.min(z_grid))
        if floor_thickness is None:
            size_x = xs[-1] - xs[0] if nx > 1 else dx
            size_y = ys[-1] - ys[0] if ny > 1 else dy
            max_side = max(size_x, size_y)
            floor_thickness = max(0.001, max_side * 0.01)  # 1% del lado o 1 mm mínimo

        bottom_z = zmin - floor_thickness

        bottom_points = np.zeros_like(top_points)
        bottom_points[:, 0:2] = top_points[:, 0:2]
        bottom_points[:, 2] = bottom_z

        points = np.vstack([top_points, bottom_points])
        Ntop = ny * nx

        faces = []

        # Top faces (dos triángulos por celda)
        for i in range(ny - 1):
            for j in range(nx - 1):
                a = i * nx + j
                b = a + 1
                c = (i + 1) * nx + j
                d = c + 1
                faces.extend([3, a, b, d])
                faces.extend([3, a, d, c])

        # Bottom faces (orientación invertida)
        for i in range(ny - 1):
            for j in range(nx - 1):
                a = i * nx + j + Ntop
                b = a + 1
                c = (i + 1) * nx + j + Ntop
                d = c + 1
                faces.extend([3, a, d, b])
                faces.extend([3, a, c, d])

        # Paredes laterales: recorrer perímetro y añadir quads triangulados
        def add_side(i1, j1, i2, j2):
            t1 = i1 * nx + j1
            t2 = i2 * nx + j2
            b1 = t1 + Ntop
            b2 = t2 + Ntop
            faces.extend([3, t1, t2, b2])
            faces.extend([3, t1, b2, b1])

        # top row left->right
        for j in range(nx - 1):
            add_side(0, j, 0, j + 1)
        # right column top->bottom
        for i in range(ny - 1):
            add_side(i, nx - 1, i + 1, nx - 1)
        # bottom row right->left
        for j in range(nx - 1):
            add_side(ny - 1, nx - 1 - j, ny - 1, nx - 2 - j)
        # left column bottom->top
        for i in range(ny - 1):
            add_side(ny - 1 - i, 0, ny - 2 - i, 0)

        faces = np.array(faces, dtype=np.int64)

        poly = pv.PolyData(points, faces)
        poly = poly.clean(tolerance=1e-8)
        return poly


    def exportar_stl(self):
        """
        Exporta STL escalado uniformemente para caber en Bambu A1 (256 mm)
        y con opción de exageración vertical. No modifica la malla mostrada.
        """
        print("ENTRÉ A exportar_stl")

        if not hasattr(self, "z_cuadrante_suave"):
            QMessageBox.warning(self, "Exportar STL", "Genere la malla primero (presione 'Ir').")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Guardar STL", "", "STL files (*.stl)")
        if not path:
            return
        if not path.lower().endswith(".stl"):
            path += ".stl"

        try:
            # --- Parámetros según calidad seleccionada ---
            quality = self.quality_combo.currentText() if hasattr(self, "quality_combo") else "Media"
            if quality == "Alta":
                factor_stl = 2
            elif quality == "Baja":
                factor_stl = 4
            else:  # Media
                factor_stl = 3

            # Exageración vertical
            try:
                vertical_exaggeration = float(self.exag_input.text())
                if vertical_exaggeration <= 0:
                    vertical_exaggeration = 1.0
            except Exception:
                vertical_exaggeration = 1.0

            max_build_mm = 256.0  # Bambu A1

            # dx/dy en metros
            dx = self.metros_x
            dy = self.metros_y

            # 🔒 Submuestreo FUERTE Y SEGURO
            z_stl = self.z_cuadrante_suave[::factor_stl, ::factor_stl].astype(np.float32)
            dx_stl = dx * factor_stl
            dy_stl = dy * factor_stl

            # Exageración vertical
            if vertical_exaggeration != 1.0:
                zmin = float(np.min(z_stl))
                z_stl = zmin + (z_stl - zmin) * vertical_exaggeration

            # 🔒 Crear sólido cerrado (SOLO una vez)
            solid = self.build_solid_from_grid(z_stl, dx_stl, dy_stl)

            # 🔒 Limpieza mínima (segura)
            solid = solid.clean(tolerance=1e-6)

            # METROS → MILÍMETROS
            solid.points *= 1000.0

            # Escalado uniforme para caber en la impresora
            bounds = solid.bounds
            size_x = bounds[1] - bounds[0]
            size_y = bounds[3] - bounds[2]
            size_z = bounds[5] - bounds[4]
            max_dim = max(size_x, size_y, size_z)

            scale_fit = min(1.0, max_build_mm / max_dim)
            if scale_fit < 1.0:
                solid.points *= scale_fit
                solid = solid.clean()

            # Centrar XY y base en Z=0
            xmin, xmax, ymin, ymax, zmin, zmax = solid.bounds
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            solid.translate([-cx, -cy, -zmin])

            # Guardado final
            solid.save(path)

            QMessageBox.information(
                self,
                "Exportar STL",
                f"STL guardado correctamente:\n{path}\n"
                f"Triángulos: {solid.n_cells}\n"
                f"Escala aplicada: {scale_fit:.4f}\n"
                f"Exageración vertical: {vertical_exaggeration}x"
            )

        except Exception as e:
            QMessageBox.critical(self, "Exportar STL", f"Error al exportar STL:\n{e}")





    def start_repeating(self, func):
        self.action_to_repeat = func
        func()
        self.repeat_timer.start()

    def stop_repeating(self):
        self.repeat_timer.stop()
        self.action_to_repeat = None

    def repeat_action(self):
        if self.action_to_repeat:
            self.action_to_repeat()

    def cambiar_direccion(self, nueva_direccion):
        self.direccion = nueva_direccion
        self.angulo = self.angulos.get(self.direccion, 0)
        self.actualizar_camara()

    def regenerar_malla(self):
        try:
            self.lat = float(self.lat_input.text())
            self.lon = float(self.lon_input.text())
            self.size = int(self.size_input.text())
            self.direccion = self.direccion_combo.currentText()
            self.angulo = self.angulos.get(self.direccion, 0)
        except ValueError:
            return

        scale_x = abs(self.transform.a)
        scale_y = abs(self.transform.e)
        self.metros_x = scale_x * 111000 * np.cos(np.deg2rad(self.lat))
        self.metros_y = scale_y * 111000

        self.row, self.col = obtener_pixel(self.lat, self.lon, self.transform)

        elevacion_punto = self.data[self.row, self.col]
        self.altitud_label.setText(f"📍 Altitud: {elevacion_punto:.2f} m")

        self.z_cuadrante = recortar_cuadrante(self.data, self.row, self.col, self.size)
        self.z_cuadrante_suave = gaussian_filter(self.z_cuadrante, sigma=1.5)

        self.posiciones_etiquetas, self.nombres_etiquetas = etiquetas_en_cuadrante(
            self.z_cuadrante_suave, self.transform, self.metros_x, self.metros_y,
            row_ini=self.row - self.size, col_ini=self.col - self.size, size=self.size
        )

        self.grid = crear_malla(self.z_cuadrante_suave, self.metros_x, self.metros_y)
        self.colors = generar_sombreado(self.z_cuadrante_suave, self.metros_x, self.metros_y)
        self.grid.point_data["colors"] = self.colors
        # habilitar export sólo si la malla fue generada con éxito
        self.btn_export.setEnabled(True)


        self.plotter.clear()
        self.plotter.add_mesh(self.grid, scalars="colors", rgb=True, show_edges=False)

        for pos, texto in zip(self.posiciones_etiquetas, self.nombres_etiquetas):
            self.plotter.add_point_labels([pos], [texto],
                                          point_color="red",
                                          text_color="white",
                                          font_size=10,
                                          point_size=5,
                                          shape_opacity=0.5)

        self.center_x = (self.z_cuadrante.shape[1] - 1) * self.metros_x / 2
        self.center_y = (self.z_cuadrante.shape[0] - 1) * self.metros_y / 2

        alt_terreno_centro = self.z_cuadrante_suave[
            self.z_cuadrante_suave.shape[0] // 2,
            self.z_cuadrante_suave.shape[1] // 2
        ]
        self.altura_cam = alt_terreno_centro + self.altura_ojos + self.altura_extra

        self.reset_camara()

    def calcular_posiciones_camara(self, angulo_rad, altura):
        cam_x = self.center_x + self.distancia_focal * np.cos(angulo_rad + np.pi)
        cam_y = self.center_y + self.distancia_focal * np.sin(angulo_rad + np.pi)
        z_cam = altura

        foco_x = self.center_x + self.distancia_focal * np.cos(angulo_rad)
        foco_y = self.center_y + self.distancia_focal * np.sin(angulo_rad)
        foco_z = altura

        return (cam_x, cam_y, z_cam), (foco_x, foco_y, foco_z)

    def actualizar_camara(self):
        pos, foco = self.calcular_posiciones_camara(self.angulo, self.altura_cam)
        self.plotter.camera_position = [pos, foco, (0, 0, 1)]
        self.plotter.render()

    def mover_arriba(self):
        self.altura_cam += 5
        self.actualizar_camara()

    def mover_abajo(self):
        self.altura_cam -= 5
        self.actualizar_camara()

    def mover_izquierda(self):
        self.angulo += np.radians(5)
        self.actualizar_camara()

    def mover_derecha(self):
        self.angulo -= np.radians(5)
        self.actualizar_camara()

    def reset_camara(self):
        self.angulo = self.angulos.get(self.direccion, 0)
        alt_terreno_centro = self.z_cuadrante_suave[
            self.z_cuadrante_suave.shape[0] // 2,
            self.z_cuadrante_suave.shape[1] // 2
        ]
        self.altura_cam = alt_terreno_centro + self.altura_ojos + self.altura_extra
        self.actualizar_camara()


    def regresar_a_selector(self):
        self.close()

        root = Path(__file__).resolve()
        for parent in root.parents:
            candidate = parent / "main.py"
            if candidate.exists():
                subprocess.Popen([sys.executable, str(candidate)])
                return

        QMessageBox.critical(self, "Error", "No se encontró main.py")
        


