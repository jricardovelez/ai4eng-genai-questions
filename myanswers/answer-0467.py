import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def clasificar_diagnostico_maquina(df_historico, lectura_actual):
    features = ['vib_x', 'vib_y', 'vib_z', 'temp_rodamientos']
    X = df_historico[features]
    y = df_historico['diagnostico']

    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_scaled, y)

    input_df = pd.DataFrame([lectura_actual], columns=features)
    input_scaled = sc.transform(input_df)

    pred = knn.predict(input_scaled)[0]
    probabilidades = knn.predict_proba(input_scaled)[0]

    mapeo = {0: "Normal", 1: "Desalineación", 2: "Desbalanceo"}
    return {
        "diagnostico_detectado": mapeo[pred],
        "confianza": f"{np.max(probabilidades) * 100:.2f}%"
    }
