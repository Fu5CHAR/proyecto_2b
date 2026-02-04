import numpy as np
import matplotlib.pyplot as plt


def cargar_datos(procesado_dir="procesado"):
    try:
        elevacion = np.load(f"{procesado_dir}/elevacion.npy")
        with open(f"{procesado_dir}/transform.txt", "r") as f:
            transform = eval(f.read())  # Carga el objeto Affine guardado como string
        return elevacion, transform
    except FileNotFoundError:
        print("Error: No se encontraron los archivos de elevación o transform.txt en el directorio.")
        exit(1)

def crear_heatmap(elevacion, transform, max_pixels=1000):
    nrows, ncols = elevacion.shape

    factor = max(1, max(nrows, ncols) // max_pixels)
    elevacion_reducida = elevacion[::factor, ::factor]

    lon_inicio = transform.c
    pixel_width = transform.a * factor
    lat_inicio = transform.f
    pixel_height = transform.e * factor

    nrows_r, ncols_r = elevacion_reducida.shape

    lon = lon_inicio + np.arange(ncols_r) * pixel_width
    lat = lat_inicio + np.arange(nrows_r) * pixel_height
    lat = lat[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Mapa de Elevación (Heatmap)")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")

    img = ax.imshow(elevacion_reducida[::-1, :],
                    extent=[lon[0], lon[-1], lat[0], lat[-1]],
                    cmap="terrain",
                    interpolation="bilinear",
                    vmin=0,
                    vmax=6000)

    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("Elevación (metros)")

    coord_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                         verticalalignment='top',
                         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    marker = ax.plot([], [], marker='o', color='red')[0]
    seleccion = {}

    def mouse_move(event):
        if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
            coord_text.set_text(f"Lon: {event.xdata:.5f}\nLat: {event.ydata:.5f}")
            fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
            lon_click = event.xdata
            lat_click = event.ydata
            seleccion['lat'] = lat_click
            seleccion['lon'] = lon_click
            print(f"Coordenadas seleccionadas: Lat {lat_click:.6f}, Lon {lon_click:.6f}")
            marker.set_data([lon_click], [lat_click])
            fig.canvas.draw_idle()
            plt.close(fig)  # Cerrar el mapa tras selección

    fig.canvas.mpl_connect('motion_notify_event', mouse_move)
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()

    return seleccion.get('lat'), seleccion.get('lon')

def main():
    elevacion, transform = cargar_datos()
    lat, lon = crear_heatmap(elevacion, transform)
    if lat and lon:
        print(f"\nCoordenadas finales seleccionadas:\n   ➤ Latitud: {lat:.6f}\n   ➤ Longitud: {lon:.6f}")
        # Aquí podrías lanzar la otra interfaz si deseas:
        # from interfaz_terreno import VentanaTerreno
        # app = QApplication(sys.argv)
        # ventana = VentanaTerreno(lat=lat, lon=lon)
        # ventana.show()
        # app.exec()

if __name__ == "__main__":
    main()
