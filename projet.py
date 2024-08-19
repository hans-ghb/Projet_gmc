import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 


st.write('''
# WineApp
Cette application vous prédit la qualité de votre vin
''')

st.sidebar.header("Parametres Physico-chimques")

def user_input(): 
    fixed_acidity = st.sidebar.slider("fixed acidity", 3.80, 15.90, 10.00)
    volatile_acidity = st.sidebar.slider("volatile acidity", 0.08, 1.58, 0.29)
    citric_acid = st.sidebar.slider("citric acid", 0.00, 1.66, 0.98)
    residual_sugar = st.sidebar.slider("residual sugar", 0.60, 65.80, 27.00)
    chlorides = st.sidebar.slider("chlorides", 0.00, 0.61, 0.21)
    free_sulfur_dioxide =  st.sidebar.slider("free sulfur dioxide", 1.00, 289.00, 110.08)
    total_sulfur_dioxide = st.sidebar.slider("total sulfur dioxide", 6.00, 440.00, 118.00)
    density = st.sidebar.slider("density", 0.97, 1.04, 0.99)
    pH = st.sidebar.slider("pH", 2.72, 4.01, 3.40)
    sulphates = st.sidebar.slider("sulphates", 0.22, 2.00, 1.10)
    alcohol = st.sidebar.slider("alcohol", 8.00, 14.90, 11.01)
    data={
        "fixed_acidity": fixed_acidity,
        "volatile_acidity": volatile_acidity,
        "citric_acid": citric_acid,
        "residual_sugar": residual_sugar,
        "chlorides": chlorides,
        "free_sulfur_dioxide": free_sulfur_dioxide,
        "total sulfur dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol
    }
    wine_parametre = pd.DataFrame(data, index=[0])
    return wine_parametre

df_ = user_input ()

st.header('On veut déterminer la qualité d’un vin')
st.write(df_)



# Charger la base de données 

df_red = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
df_white = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', sep=';')

df_red['type_vin'] = 1
df_white['type_vin'] = 0

df = pd.concat([df_red, df_white])

# Division du jeu de donnée en données d'entrainement et de test

X = df.drop(['quality', 'type_vin'], axis=1)
y = df['quality']

seed = 33 #Pour s'assurer de la reproductibilité des résultats

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)


# Standardisation du jeu de données

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Modélisation 

rf = RandomForestRegressor(random_state=seed)
rf.fit(X_train_scaled, y_train)



# Prédiction à l'aide du modèle

# prediction = rf.predict(X_test_scaled)
prediction = rf.predict(df_)


st.subheader("Le score de qualité de ce vin est:")

st.write(prediction)