import numpy as np
import pyvista as pv
from pathlib import Path
import os
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, sobel
from Motor.etiquetador import etiquetas_en_cuadrante
from affine import Affine



from pathlib import Path
import numpy as np

def cargar_datos(procesado_dir="procesado"):
    # Directorio base del proyecto (iiB_Proyecto_Final)
    base_dir = Path(__file__).resolve().parent.parent

    procesado_path = base_dir / procesado_dir

    data_path = procesado_path / "elevacion.npy"
    transform_path = procesado_path / "transform.txt"

    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró: {data_path}")

    if not transform_path.exists():
        raise FileNotFoundError(f"No se encontró: {transform_path}")

    data = np.load(data_path)

    with open(transform_path, "r", encoding="utf-8") as f:
        transform = eval(f.read())

    return data, transform



def obtener_pixel(lat, lon, transform):
    col, row = ~transform * (lon, lat)
    return int(row), int(col)


def recortar_cuadrante(data, row, col, size):
    r1, r2 = max(row - size, 0), min(row + size, data.shape[0] - 1)
    c1, c2 = max(col - size, 0), min(col + size, data.shape[1] - 1)
    return data[r1:r2, c1:c2]


def calcular_normales(z_data, metros_x, metros_y):
    dzdx = sobel(z_data, axis=1) / (8 * metros_x)
    dzdy = sobel(z_data, axis=0) / (8 * metros_y)
    normal = np.dstack((-dzdx, -dzdy, np.ones_like(z_data)))
    norm = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= norm
    normal[:, :, 1] /= norm
    normal[:, :, 2] /= norm
    return normal


def crear_malla(z_data, metros_x, metros_y):
    ny, nx = z_data.shape
    x = np.arange(nx) * metros_x
    y = np.arange(ny) * metros_y
    X, Y = np.meshgrid(x, y)
    Z = z_data
    grid = pv.StructuredGrid(X, Y, Z)
    return grid


def generar_sombreado(z_cuadrante, metros_x, metros_y, azimut=315, elevacion=45):
    normales = calcular_normales(z_cuadrante, metros_x, metros_y)

    azimut_rad = np.radians(azimut)
    elevacion_rad = np.radians(elevacion)
    luz_x = np.cos(azimut_rad) * np.cos(elevacion_rad)
    luz_y = np.sin(azimut_rad) * np.cos(elevacion_rad)
    luz_z = np.sin(elevacion_rad)
    luz = np.array([luz_x, luz_y, luz_z])

    sombreado = np.sum(normales * luz, axis=2)
    sombreado = (sombreado + 1) / 2  # Normalizar a 0-1

    ny, nx = z_cuadrante.shape
    x = np.arange(nx) * metros_x
    y = np.arange(ny) * metros_y
    X, Y = np.meshgrid(x, y)

    center_x = x[-1] / 2
    center_y = y[-1] / 2

    dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    dist_norm = dist / dist.max()


    cmap_altitud = LinearSegmentedColormap.from_list("relieve", [
    (0.0, 0.1, 0.3),    # Azul oscuro, cerca del 0 (bajas altitudes)
    (0.2, 0.6, 0.2),    # Verde intermedio (alturas medias)
    (0.9, 0.9, 0.9)     # Gris/blanco claro para picos altos
])


    colors_alt = cmap_altitud(dist_norm)[:, :, :3]

    sombreado_3d = np.repeat(sombreado[:, :, np.newaxis], 3, axis=2)
    colors = colors_alt * sombreado_3d

    return colors.reshape(-1, 3)


import os

def exportar_stl_desde_grid(grid, filepath):
    """
    Exporta una StructuredGrid (pyvista) a STL y verifica la exportación.
    - grid: StructuredGrid retornado por crear_malla()
    - filepath: ruta final terminando en .stl
    """

    # Validaciones básicas
    if filepath is None or filepath == "":
        raise ValueError("Ruta inválida para exportar STL")

    if not filepath.lower().endswith(".stl"):
        raise ValueError("El archivo de salida debe tener extensión .stl")

    if grid is None or grid.n_points == 0:
        raise ValueError("La malla está vacía o no es válida")

    print(">>> EXPORTANDO STL <<<")
    print("Ruta destino:", os.path.abspath(filepath))

    # Extrae superficie y trianguliza
    surf = grid.extract_surface().triangulate()

    if surf.n_points == 0 or surf.n_cells == 0:
        raise RuntimeError("La superficie generada es inválida (sin geometría)")

    # Guardado
    surf.save(filepath)

    # Verificación REAL en disco
    if not os.path.exists(filepath):
        raise IOError("El archivo STL NO se creó en disco")

    size = os.path.getsize(filepath)
    if size == 0:
        raise IOError("El archivo STL se creó pero está vacío (0 bytes)")

    print(f"✅ STL exportado correctamente ({size/1024:.2f} KB)")
    return filepath



def visualizar_terreno(lat=None, lon=None, direccion=None):
    data, transform = cargar_datos()

    # Entrada para latitud, longitud y direccion con validación
    while True:
        try:
            if lat is None:
                lat = float(input("Ingrese latitud decimal: "))
            if lon is None:
                lon = float(input("Ingrese longitud decimal: "))

            if direccion is None:
                direccion = input("Ingrese dirección inicial de cámara (NORTE, SUR, ESTE, OESTE): ").strip().upper()
                if direccion not in {"NORTE", "SUR", "ESTE", "OESTE"}:
                    print("Dirección inválida. Usando NORTE por defecto.")
                    direccion = "NORTE"

            row, col = obtener_pixel(lat, lon, transform)
            altitud_ms_nm = data[row, col]

            if not (0 <= row < data.shape[0]) or not (0 <= col < data.shape[1]):
                print("❌ Coordenadas fuera del rango válido. Intenta nuevamente.")
                lat, lon, direccion = None, None, None
                continue
            break
        except Exception:
            print("❌ Entrada inválida. Intenta nuevamente.")
            lat, lon, direccion = None, None, None

    print(f"Pixel aproximado: fila {row}, columna {col}")
    print(f"Cámara apuntará hacia: {direccion}")

    size = 1000
    z_cuadrante = recortar_cuadrante(data, row, col, size)
    sigma = 1.5
    z_cuadrante_suave = gaussian_filter(z_cuadrante, sigma=sigma)

    scale_x = abs(transform.a)
    scale_y = abs(transform.e)
    metros_x = scale_x * 111000 * np.cos(np.deg2rad(lat))
    metros_y = scale_y * 111000

    print(f"Resolución en metros: x={metros_x:.2f} m, y={metros_y:.2f} m")

    posiciones_etiquetas, nombres_etiquetas = etiquetas_en_cuadrante(
        z_cuadrante_suave, transform, metros_x, metros_y,
        row_ini=row - size, col_ini=col - size, size=size
    )

    grid = crear_malla(z_cuadrante_suave, metros_x, metros_y)
    colors = generar_sombreado(z_cuadrante_suave, metros_x, metros_y)

    center_x = (z_cuadrante.shape[1] - 1) * metros_x / 2
    center_y = (z_cuadrante.shape[0] - 1) * metros_y / 2
    alt_terreno_centro = z_cuadrante_suave[z_cuadrante_suave.shape[0] // 2, z_cuadrante_suave.shape[1] // 2]

    altura_ojos = 1.7
    altura_extra = 10
    distancia_focal = 100  # metros

    # Inicial posición base y altura
    cam_x, cam_y = center_x, center_y
    altura_cam = alt_terreno_centro + altura_ojos + altura_extra


    def calcular_posiciones_camara(angulo_rad, altura):
        # Cámara detrás del foco a distancia_focal
        x_cam = cam_x + distancia_focal * np.cos(angulo_rad + np.pi)
        y_cam = cam_y + distancia_focal * np.sin(angulo_rad + np.pi)
        z_cam = altura
        # Foco delante del centro a distancia_focal
        x_focal = cam_x + distancia_focal * np.cos(angulo_rad)
        y_focal = cam_y + distancia_focal * np.sin(angulo_rad)
        z_focal = altura
        return (x_cam, y_cam, z_cam), (x_focal, y_focal, z_focal)

    cam_pos, focal_pos = calcular_posiciones_camara(altura_cam)

    view_up = (0, 0, 1)

    plotter = pv.Plotter()
    plotter.set_background("black")
    plotter.add_mesh(grid, scalars=colors, rgb=True, show_edges=False)

    for pos, texto in zip(posiciones_etiquetas, nombres_etiquetas):
        plotter.add_point_labels(
            [pos], [texto], point_color="red", text_color="white",
            font_size=10, point_size=5, shape_opacity=0.5
        )

    plotter.camera_position = [cam_pos, focal_pos, view_up]

    # Estado interno mutable para funciones internas
    estado = {
        "altura": altura_cam,
    }

    def on_key_press(key):
        step_ang = np.radians(5)  # 5 grados de paneo
        step_alt = 5.0            # 5 metros de subida/bajada

        angulo = estado["angulo"]
        altura = estado["altura"]

        if key == "Up":
            altura += step_alt
        elif key == "Down":
            altura -= step_alt
        elif key == "Right":
            angulo -= step_ang
        elif key == "Left":
            angulo += step_ang

        estado["angulo"] = angulo
        estado["altura"] = altura

        cam_pos, focal_pos = calcular_posiciones_camara(angulo, altura)
        plotter.camera_position = [cam_pos, focal_pos, view_up]
        plotter.render()

    plotter.add_key_event("Up", lambda: on_key_press("Up"))
    plotter.add_key_event("Down", lambda: on_key_press("Down"))
    plotter.add_key_event("Right", lambda: on_key_press("Right"))
    plotter.add_key_event("Left", lambda: on_key_press("Left"))

    plotter.show()