import streamlit as st
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier

@st.cache_data
def load_data():
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['Class'] = wine.target
    return df, wine.target_names

df,target_names=load_data()

model=RandomForestClassifier()
model.fit(df.iloc[:,:-1],df['Class'])

df.rename(columns={'od280/od315_of_diluted_wines': 'od280_od315_diluted_wines'}, inplace=True)


st.sidebar.title("Input Features")
alcohol = st.sidebar.slider( "Alcohol", float(df['alcohol'].min()),float(df['alcohol'].max()))
malic_acid = st.sidebar.slider("Malic acid",float(df['malic_acid'].min()),float(df['malic_acid'].max()))
ash = st.sidebar.slider("Ash",float(df['ash'].min()),float(df['ash'].max()))
color_intensity = st.sidebar.slider("color_intensity",float(df['color_intensity'].min()),float(df['color_intensity'].max()))
hue= st.sidebar.slider("Hue",float(df['hue'].min()),float(df['hue'].max()))
flavanoids = st.sidebar.slider("Flavanoids",float(df['flavanoids'].min()),float(df['flavanoids'].max()))
alcalinity_of_ash = st.sidebar.slider("Alcalinity of ash",float(df['alcalinity_of_ash'].min()),float(df['alcalinity_of_ash'].max()))
magnesium = st.sidebar.slider("Magnesium",float(df['magnesium'].min()),float(df['magnesium'].max()))
nonflavanoid_phenols = st.sidebar.slider("Nonflavanoid_phenols",float(df['nonflavanoid_phenols'].min()),float(df['nonflavanoid_phenols'].max()))
proanthocyanins = st.sidebar.slider("Proanthocyanins",float(df['proanthocyanins'].min()),float(df['proanthocyanins'].max()))
total_phenols = st.sidebar.slider("Total Phenols",float(df['total_phenols'].min()),float(df['total_phenols'].max()))
od280_od315_diluted_wines = st.sidebar.slider("OD280/OD315 of diluted wines",float(df['od280_od315_diluted_wines'].min()),float(df['od280_od315_diluted_wines'].max()))
proline = st.sidebar.slider("Proline",float(df['proline'].min()),float(df['proline'].max()))


st.write(f"Selected Alcohol: {alcohol}")
st.write(f"Selected Malic Acid: {malic_acid}")
st.write(f"Selected Ash: {ash}")
st.write(f"Selected Alcalinity of ash: {alcalinity_of_ash}")
st.write(f"Selected Magnesium: {magnesium}")
st.write(f"Selected Nonflavanoid_phenols: {nonflavanoid_phenols}")
st.write(f"Selected Proanthocyanins: {proanthocyanins}")
st.write(f"Selected Total Phenols: {total_phenols}")
st.write(f"Selected OD280/OD315 of diluted wines: {od280_od315_diluted_wines}")
st.write(f"Selected Proline: {proline}")


input_data = [[alcohol, malic_acid, ash, color_intensity, hue,flavanoids,
               alcalinity_of_ash, magnesium,nonflavanoid_phenols,proanthocyanins,total_phenols, od280_od315_diluted_wines,proline ]]

## Prediction
prediction = model.predict(input_data)
predicted_class = target_names[prediction[0]]

st.write("Prediction")
st.write(f"The predicted class is: {predicted_class}")

