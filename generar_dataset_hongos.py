# ============================================================
# GENERADOR DE DATASET PROPIO - CLASIFICACIÓN DE HONGOS
# Comestible vs. Venenoso
# ============================================================
# Este dataset es de creación propia (no descargado de UCI).
# Se genera con reglas lógicas + ruido aleatorio, simulando
# relaciones reales entre las características de un hongo y
# su nivel de toxicidad.
# ============================================================

import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

N = 1800  # número de registros a generar

# ------------------------------------------------------------
# Categorías posibles para cada variable (nombres legibles,
# no letras crípticas como en el dataset de UCI)
# ------------------------------------------------------------
olores = ['almendra', 'anís', 'ninguno', 'creosota', 'pescado', 'fétido', 'picante', 'mohoso']
colores_sombrero = ['blanco', 'marrón', 'rojo', 'amarillo', 'gris', 'verde', 'rosa', 'canela']
formas_sombrero = ['campana', 'cónico', 'convexo', 'plano', 'hundido', 'nudoso']
superficies_sombrero = ['fibrosa', 'lisa', 'escamosa', 'estriada']
colores_laminas = ['negro', 'marrón', 'blanco', 'rosa', 'gris', 'verde', 'morado', 'rojo']
tamanos_laminas = ['ancha', 'angosta']
formas_tallo = ['agrandada', 'ahusada']
colores_tallo = ['blanco', 'marrón', 'rosa', 'gris', 'canela']
tipos_anillo = ['ninguno', 'evanescente', 'grande', 'colgante']
numero_anillos = ['ninguno', 'uno', 'dos']
colores_espora = ['blanco', 'marrón', 'negro', 'morado', 'verde', 'café']
poblacion = ['abundante', 'agrupada', 'numerosa', 'dispersa', 'varias', 'solitaria']
habitat = ['bosque', 'pastizal', 'senderos', 'urbano', 'desechos', 'hojas']
moretones = ['sí', 'no']
colores_velo = ['blanco', 'marrón', 'naranja']
texturas = ['seca', 'húmeda', 'viscosa']

# Olores fuertemente asociados a toxicidad (regla base, inspirada
# en la relación real que existe entre olor y toxicidad en hongos)
olores_peligrosos = {'creosota', 'pescado', 'fétido', 'picante', 'mohoso'}
esporas_peligrosas = {'verde', 'morado'}
habitats_riesgo = {'bosque', 'hojas'}

registros = []

for i in range(N):
    odor = np.random.choice(olores)
    spore = np.random.choice(colores_espora)
    hab = np.random.choice(habitat)

    # --- Regla de generación de la clase (probabilística, no 100% determinista) ---
    riesgo = 0
    if odor in olores_peligrosos:
        riesgo += 3
    if spore in esporas_peligrosas:
        riesgo += 2
    if hab in habitats_riesgo:
        riesgo += 1

    prob_venenoso = min(0.95, 0.05 + riesgo * 0.18)  # más riesgo = más probabilidad
    es_venenoso = np.random.rand() < prob_venenoso
    clase = 'venenoso' if es_venenoso else 'comestible'

    registro = {
        'clase': clase,
        'olor': odor,
        'color_sombrero': np.random.choice(colores_sombrero),
        'forma_sombrero': np.random.choice(formas_sombrero),
        'superficie_sombrero': np.random.choice(superficies_sombrero),
        'color_laminas': np.random.choice(colores_laminas),
        'tamano_laminas': np.random.choice(tamanos_laminas),
        'forma_tallo': np.random.choice(formas_tallo),
        'color_tallo': np.random.choice(colores_tallo),
        'tipo_anillo': np.random.choice(tipos_anillo),
        'numero_anillos': np.random.choice(numero_anillos),
        'color_espora': spore,
        'poblacion': np.random.choice(poblacion),
        'habitat': hab,
        'moretones': np.random.choice(moretones),
        'color_velo': np.random.choice(colores_velo),
        'textura': np.random.choice(texturas),
        'diametro_sombrero_cm': round(np.random.normal(6, 2), 1),
        'altura_tallo_cm': round(np.random.normal(7, 2.5), 1),
    }
    registros.append(registro)

df = pd.DataFrame(registros)

# Limitar valores numéricos a rangos realistas (sin negativos)
df['diametro_sombrero_cm'] = df['diametro_sombrero_cm'].clip(lower=1)
df['altura_tallo_cm'] = df['altura_tallo_cm'].clip(lower=1)

# Guardar el dataset
df.to_csv('hongos_dataset_propio.csv', index=False, encoding='utf-8-sig')

print("Dataset generado exitosamente")
print(f"Registros: {df.shape[0]}")
print(f"Variables: {df.shape[1]} (18 predictoras + 1 variable objetivo 'clase')")
print(f"\nDistribución de clases:")
print(df['clase'].value_counts())
print(f"\nPrimeras filas:")
print(df.head())
print(f"\nArchivo guardado como: hongos_dataset_propio.csv")
