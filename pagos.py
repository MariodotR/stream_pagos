############### STREAMLIT ########################

from PIL import Image
import streamlit as st


#title_image = Image.open('AppTitle.png')
title_image = Image.open('title.jpg')
st.set_page_config(page_title='Stock Clientes AFP Capital', 
                   page_icon=title_image, 
                   layout='wide')

#--------------------------------- ---------------------------------  ---------------------------------
#--------------------------------- SETTING UP THE APP
#--------------------------------- ---------------------------------  ---------------------------------
col1, col2, col3 = st.columns(3)
with col1:
	st.markdown('')
with col2:
	st.image(title_image,use_column_width='auto',width = 800)
with col3:
	st.markdown('')


st.markdown('#### Bienvenidos al Sistema recomendador de Requerimientos')

Texto = st.text_input('Ingrese su Requerimiento', value = '')
st.markdown('##### Su Requerimiento a análizar es')
st.write( Texto )


############### DESARROLLO ########################


import numpy as np
import pandas as pd

#!pip install sklearn
from sklearn.metrics.pairwise import cosine_similarity


#!pip install nltk
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords

# palabras que no se consideran
palabras_basura= stopwords.words('spanish') +["saludo","saludos","gracias","gracia","buena","tarde","buenos","estimado","estimada", "c", "si", "no", "x","n"]

#Carga de Datos
df= pd.read_excel('Pago_Pension.xlsx')
print(df.shape)
RQ_Descripcion = df['Descripcion']
RQ_Respuesta = df['RespuestaBack']
RQ_Fecha = df['Req_Identificador_Id']

!pip install spacy
!python -m spacy download es_core_news_sm

import spacy
import es_core_news_sm
#nlp = spacy.load('es_core_news_md')
nlp = es_core_news_sm.load()

def preprocesamiento(text):
    """
    Lematiza en español y luego quita las stopwords y palabras basura
    """
    results = ""
    doc = nlp(text)  
    #tokens= [word.lemma for sent in doc.sentences for word in sent.words]
    for token in doc:
        if token.pos_!="NUM":
            if str(token.lemma_) not in palabras_basura:
                results+= str(token.lemma_)+" "
        else:
            results+= "/NUM/"+ " "
        
    return results


df['Descripcion_limpio'] = df['Descripcion'].apply(preprocesamiento)

from sklearn.feature_extraction.text import CountVectorizer

import pickle

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

discrete_train = vectorizer.fit_transform(df['Descripcion_limpio']) # aplicar la transformacion a data_train para obtener discrete_train



def Recomendacion(texto):
    
    Emb_Consulta= vectorizer.transform([preprocesamiento(texto)]) 
    Distancias = cosine_similarity(Emb_Consulta, discrete_train)
    top_at=5
    ind = np.argpartition(Distancias[0,:], -top_at)[-top_at:]
    return Distancias[0,ind],ind

##########################STREAMLIT ###########


st.spinner(text="Calculando...") 

Distancias,Top_Indices = Recomendacion(Texto)


st.markdown('##### Los requerimientos ingresados previamente mas similares son: ')


col1, col2 = st.columns(2)
with col1:
    st.markdown('##### Consulta')
    st.metric(label = 'Similaridad ',  value = round(Distancias[0],2))
    st.markdown(RQ_Descripcion.iloc[Top_Indices[0]])

with col2:
    st.markdown('##### Respuesta Back')
    st.metric(label = 'NroReq',  value = RQ_Fecha.iloc[Top_Indices[0]])
    st.markdown(RQ_Respuesta.iloc[Top_Indices[0]])


col1, col2 = st.columns(2)
with col1:
    st.metric(label = 'Similaridad ',  value = round(Distancias[1],2))
    st.markdown(RQ_Descripcion.iloc[Top_Indices[1]])


with col2:
    st.metric(label = 'NroReq',  value = RQ_Fecha.iloc[Top_Indices[1]])
    st.markdown(RQ_Respuesta.iloc[Top_Indices[1]])


col1, col2 = st.columns(2)
with col1:
    st.metric(label = 'Similaridad ',  value = round(Distancias[2],2))
    st.markdown(RQ_Descripcion.iloc[Top_Indices[2]])


with col2:
    st.metric(label = 'NroReq',  value = RQ_Fecha.iloc[Top_Indices[2]])
    st.markdown(RQ_Respuesta.iloc[Top_Indices[2]])