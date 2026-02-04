import numpy as np


montanas_ecuador = {
    "Antisana": (-0.4811, -78.1402),
    "Atacazo": (-0.347, -78.630),
    "Cayambe": (0.0296, -78.1396),
    "Chimborazo": (-1.4692, -78.8170),
    "Cotopaxi": (-0.6773, -78.4369),
    "El Altar (Capac Urcu)": (-1.708, -78.442),
    "Guagua Pichincha": (-0.162, -78.618),
    "Imbabura": (0.223, -78.210),
    "Iliniza Norte": (-0.659, -78.715),
    "Iliniza Sur": (-0.663, -78.712),
    "Pululahua": (0.0336, -78.4586),
    "Reventador": (-0.077, -77.656),
    "Ruco Pichincha": (-0.1807, -78.6099),
    "Sangay": (-2.005, -78.341),
    "Sincholagua": (-0.616, -78.439),
    "Sumaco": (-0.539, -77.626),
    "Tungurahua": (-1.467, -78.443),
    
    #cerritos 
    "Cerro Cajas": (-2.783, -79.268),
    "Cerro Hayas": (-1.232, -78.630),
    "Cerro Puntas": (-0.210, -78.257),
    "Cerro Mandango": (-4.254, -79.225),
    "Cerro Panecillo": (-0.2264, -78.5133),
    "Loma de Puengasí": (-0.248, -78.492),
    "Yanaurcu de Pasto": (0.831, -77.870),
    
    
    
}


def etiquetas_en_cuadrante(z_data, transform, metros_x, metros_y, row_ini, col_ini, size):
    posiciones = []
    etiquetas = []
    for nombre, (lat, lon) in montanas_ecuador.items():
        try:
            col, row = ~transform * (lon, lat)
            row, col = int(row), int(col)
            if row_ini <= row < row_ini + 2 * size and col_ini <= col < col_ini + 2 * size:
                rel_row = row - row_ini
                rel_col = col - col_ini
                z = z_data[rel_row, rel_col]
                x = rel_col * metros_x
                y = rel_row * metros_y
                posiciones.append([x, y, z + 30])  # Etiqueta un poco encima del terreno
                etiquetas.append(nombre)
        except:
            continue
    return np.array(posiciones), etiquetas