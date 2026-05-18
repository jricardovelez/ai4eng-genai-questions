import pandas as pd

def resumir_ganancias_helados(df_ventas, df_costos):
    merged = df_ventas.merge(df_costos, on='sabor')
    hot = merged[merged['temperatura'] > 25].copy()
    hot['ganancia_neta'] = hot['unidades_vendidas'] * (hot['precio_venta'] - hot['costo_produccion'])
    return hot.groupby('sabor')['ganancia_neta'].sum().sort_values(ascending=False)
