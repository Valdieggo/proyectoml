import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


data_test=pd.read_csv('atp_matches_2023.csv')

# Seleccionar las columnas deseadas
selected_columns = ['tourney_id', 'surface', 'tourney_level', 'winner_id', 'winner_entry', 'winner_hand', 'winner_ht', 'winner_age', 'loser_id', 'loser_entry', 'loser_hand', 'loser_ht', 'loser_age', 'winner_rank_points', 'loser_rank_points']
# Se crea un dataframe con las columnas seleccionadas
data_selected_test = data_test[selected_columns]

# Crear una nueva columna llamada player_id_win
data_selected_test['player_id_win'] = data_selected_test['winner_id']

data_selected_test = data_selected_test.rename(columns={
    'winner_id': 'player1_id',
    'winner_entry': 'player1_entry',
    'winner_hand': 'player1_hand',
    'winner_ht': 'player1_ht',
    'winner_age': 'player1_age',
    'winner_rank_points': 'player1_rank_points',
    'loser_id': 'player2_id',
    'loser_entry': 'player2_entry',
    'loser_hand': 'player2_hand',
    'loser_ht': 'player2_ht',
    'loser_age': 'player2_age',
    'loser_rank_points': 'player2_rank_points'
})
# Mapeo para 'surface'
surface_mapping = {'Clay': 0, 'Grass': 1, 'Hard': 2, 'Carpet': 3} 
data_selected_test['surface_encoded'] = data_selected_test['surface'].map(surface_mapping)

# Mapeo para 'winner_hand'
hand_mapping = {'R': 0, 'L': 1, 'U': 2}  # Puedes ajustar esto según tus datos
data_selected_test['player1_hand_encoded'] = data_selected_test['player1_hand'].map(hand_mapping)
data_selected_test['player2_hand_encoded'] = data_selected_test['player2_hand'].map(hand_mapping)
data_selected_test = data_selected_test.drop(['surface', 'player1_hand','player2_hand'], axis=1)

# Reemplazar el guion '-' con una cadena vacía ''
data_selected_test['tourney_id'] = data_selected_test['tourney_id'].str.replace('-', '')

tourney_level_mapping = {'G': 0, 'M': 1, 'A': 2, 'C': 3, 'S': 4, 'F': 5, 'D': 6}  # Puedes ajustar esto según tus datos
data_selected_test['tourney_level_encoded'] = data_selected_test['tourney_level'].map(tourney_level_mapping)

# Eliminar la columna original 'tourney_level'
data_selected_test = data_selected_test.drop(['tourney_level'], axis=1)

winner_entry_mapping = {'WC': 1, 'Q': 2, 'LL': 3, 'PR': 4} 
data_selected_test['player1_entry_encoded'] = data_selected_test['player1_entry'].map(winner_entry_mapping)
data_selected_test['player2_entry_encoded'] = data_selected_test['player2_entry'].map(winner_entry_mapping)

# Eliminar la columna original 'player1_entry' y 'player2_entry'
data_selected_test = data_selected_test.drop(['player1_entry', 'player2_entry'], axis=1)
# Reemplazar NaN con 0 en 'player1_entry_encoded' y 'player2_entry_encoded'
data_selected_test['player1_entry_encoded'].fillna(0, inplace=True)
data_selected_test['player2_entry_encoded'].fillna(0, inplace=True)

# Eliminar la columna 'tourney_id'
data_selected_test = data_selected_test.drop(['tourney_id'], axis=1)
data_selected_test = data_selected_test.dropna()

# Visualizar las primeras filas del DataFrame actualizado
#print(data_selected_test.columns)

y_test= data_selected_test['player_id_win']
x_test = data_selected_test.drop(['player_id_win'], axis=1)
dataset2_test = data_selected_test.copy()
dataset2_test['win'] = 1

# Establecer una semilla para reproducibilidad
np.random.seed(42)
tamano_subset = int(0.55 * len(dataset2_test))

# Seleccionar al azar un subconjunto de filas para intercambiar las instancias
indices_a_cambiar = np.random.choice(dataset2_test.index, size=tamano_subset, replace=False)  # ajusta el tamaño según sea necesario

# Copiar las características y etiquetas de Player 1 en dataset2
features_player1 = dataset2_test.loc[indices_a_cambiar, [
    'player1_id', 'player1_ht', 'player1_age', 'player1_rank_points',
    'player1_hand_encoded', 'player1_entry_encoded'
]]
labels_player1 = 1 - dataset2_test.loc[indices_a_cambiar, 'win']  # Invertir las etiquetas para representar victorias de Player 2

# Copiar las características y etiquetas de Player 2 en dataset2
features_player2 = dataset2_test.loc[indices_a_cambiar, [
    'player2_id', 'player2_ht', 'player2_age', 'player2_rank_points',
    'player2_hand_encoded', 'player2_entry_encoded'
]]
labels_player2 = dataset2_test.loc[indices_a_cambiar, 'win']

# Actualizar las instancias de Player 1 en dataset2 con las de Player 2
dataset2_test.loc[indices_a_cambiar, [
    'player1_id', 'player1_ht', 'player1_age', 'player1_rank_points',
    'player1_hand_encoded', 'player1_entry_encoded'
]] = features_player2.values
dataset2_test.loc[indices_a_cambiar, 'win'] = labels_player2.values  # Invertir las etiquetas en 'win'

# Actualizar las instancias de Player 2 en dataset2 con las de Player 1
dataset2_test.loc[indices_a_cambiar, [
    'player2_id', 'player2_ht', 'player2_age', 'player2_rank_points',
    'player2_hand_encoded', 'player2_entry_encoded'
]] = features_player1.values
dataset2_test.loc[indices_a_cambiar, 'win'] = labels_player1.values

# Verificar los primeros registros del DataFrame actualizado (dataset2_test)
#print(dataset2_test.head())

y_dataset2_test = dataset2_test['win']
x_dataset2_test = dataset2_test.drop(['player_id_win','win'], axis=1)



print("Los modelos seran evaluados con los datos de los partidos 2023, estos modelos fueron entrenados con los resultados de los partidos de tenis desde el año 2000 al 2022 \n")


# Cargar el modelo desde el archivo
modelo1 = load_model('models/keras_model1_relu62.h5')
print("Modelo 1, capas densas activacion, relu")
# Utilizar el modelo cargado para predicciones, evaluación, etc.
modelo1.evaluate(x_dataset2_test,y_dataset2_test)



modelo2 = load_model('models/keras_model2_leaky63.h5')
print("Modelo 2 ,capas densas, activacion LeakyReLU")
# Utilizar el modelo cargado para predicciones, evaluación, etc.
modelo2.evaluate(x_dataset2_test,y_dataset2_test)


# Cargar el modelo desde el archivo
modelo3 = load_model('models/keras_model3_64_relu_leaky.h5')
print("Modelo 4 ,capas densas, activacion LeakyReLU")
# Utilizar el modelo cargado para predicciones, evaluación, etc.
modelo3.evaluate(x_dataset2_test,y_dataset2_test)
print("Modelo 0 ,randomforest")
# Cargar el modelo
modelo_forest = joblib.load('models/random_forest_model27_comp.pkl')
y_pred = modelo_forest.predict(x_test)

# Calcular la precisión del modelo en datos de prueba
precision = accuracy_score(y_test, y_pred)

print(f'Precisión en datos de prueba: {precision}')