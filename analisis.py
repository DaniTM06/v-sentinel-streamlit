import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud

def analisis():
    st.title("Análisis Visual del Sentimiento")

    # Carga de datos
    df = pd.read_csv("fine_tuning.csv")

    # Convertimos los valores booleanos en enteros para visualización
    emotion_cols = [
        'anger','anticipation','disgust','fear','joy','love',
        'optimism','pessimism','sadness','surprise','trust'
    ]
    df[emotion_cols] = df[emotion_cols].astype(int)

    st.markdown("En este apartado se podrá descubrir **cómo se sienten los usuarios en base a los mensajes analizados en redes sociales**.")

    # Sección 1: Conteo de emociones
    st.subheader("Distribución General de Emociones")
    emotion_counts = df[emotion_cols].sum().sort_values(ascending=False)
    fig_emotions = px.bar(
        emotion_counts,
        x=emotion_counts.index,
        y=emotion_counts.values,
        labels={'x': 'Emoción', 'y': 'Cantidad'},
        title='Cantidad de Mensajes por Emoción',
        color=emotion_counts.values,
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_emotions)

    # Sección 2: Nube de palabras con keywords
    st.subheader("Palabras clave más comunes")
    st.markdown("Visualización de los conceptos más comunes según nuestra IA")
    text_keywords = ' '.join(df['predicted_keyword'].dropna().astype(str))
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='plasma'
    ).generate(text_keywords)

    st.image(wordcloud.to_array(), use_container_width=True)

    # Sección 3: Relación entre sentimientos positivos y negativos
    st.subheader("Comparación de Sentimientos Positivos y Negativos")
    df['positivos'] = df[['joy', 'love', 'trust', 'optimism', 'anticipation']].sum(axis=1)
    df['negativos'] = df[['anger', 'fear', 'disgust', 'sadness', 'pessimism']].sum(axis=1)

    sentiment_df = pd.DataFrame({
        'Tipo': ['Positivos', 'Negativos'],
        'Cantidad': [df['positivos'].sum(), df['negativos'].sum()]
    })

    fig_sentiment = px.pie(
        sentiment_df,
        names='Tipo',
        values='Cantidad',
        title='Proporción de Sentimientos Positivos vs Negativos',
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    st.plotly_chart(fig_sentiment)

    # Sección 4: Gráfico de barras de palabras clave en comentarios negativos
    st.subheader("Palabras clave predominantes en comentarios negativos")
    st.markdown("Aquí se muestran las palabras clave que más aparecen en los comentarios que tienen sentimientos negativos.")

    # Filtrar los comentarios negativos (cuando la columna 'negativos' es mayor que 0)
    negative_comments = df[df['negativos'] > 0]
    text_negative_keywords = ' '.join(negative_comments['predicted_keyword'].dropna().astype(str))

    # Contar las apariciones de las palabras clave
    keyword_counts = Counter(text_negative_keywords.split())

    # Convertir el Counter en un DataFrame
    keyword_df = pd.DataFrame(keyword_counts.items(), columns=['Palabra Clave', 'Apariciones'])
    keyword_df = keyword_df.sort_values(by='Apariciones', ascending=False)

    # Mostrar el gráfico de barras con las palabras clave más frecuentes
    fig_keywords = px.bar(
        keyword_df.head(10),  # Mostrar solo las 10 palabras más frecuentes
        x='Palabra Clave',
        y='Apariciones',
        title='Palabras Clave Más Frecuentes en Comentarios Negativos',
        labels={'Palabra Clave': 'Palabra Clave', 'Apariciones': 'Apariciones'},
        color='Apariciones',
        color_continuous_scale='Blues'
    )

    st.plotly_chart(fig_keywords)

    # Sección 5: Muestra de ejemplos reales
    st.subheader("Ejemplos Reales Analizados")
    st.markdown("Aquí puedes ver algunas palabras clave y los sentimientos que se detectaron en ellas:")

    num_samples = st.slider("Número de ejemplos a mostrar:", 5, 50, 10)
    sample_df = df[['predicted_keyword'] + emotion_cols].dropna().sample(num_samples)

    # Convertir 0 en 'No' y 1 en 'Yes' para las emociones
    sample_df[emotion_cols] = sample_df[emotion_cols].replace({0: 'No', 1: 'Yes'})

    # Función para aplicar el estilo a las celdas
    def color_cells(val):
        color = 'background-color: lightcoral' if val == 'No' else 'background-color: lightgreen'
        return color

    # Estilo de las celdas para mostrar en verde claro si es 'Yes', rojo claro si es 'No'
    styled_df = sample_df.style.applymap(color_cells, subset=emotion_cols)

    st.dataframe(styled_df, use_container_width=True)

    # Sección 6: Conclusión automática
    st.subheader("Conclusiones")
    if sentiment_df.loc[0, 'Cantidad'] > sentiment_df.loc[1, 'Cantidad']:
        st.success("Predominan los sentimientos positivos. Los usuarios se muestran optimistas y confiados.")
    else:
        st.warning("Predominan los sentimientos negativos. Hay una tendencia hacia emociones como miedo o enojo.")

    st.markdown("---")


def main():
    st.sidebar.title("Menú de Navegación")
    option = st.sidebar.selectbox("Selecciona una sección:", ["Informacion del Proyecto", "Analisis", "License"])

    if option == "Informacion del Proyecto":
        st.markdown(
            "<div style='background-color: #007BFF; padding: 20px; border-radius: 10px;'>"
            "<h1 style='color: white; text-align: center;'>Proyecto V-Sentinel</h1>"
            "</div>",
            unsafe_allow_html=True
        )
        st.title("Aplicación para monitorear las opiniones de los usuarios sobre productos, servicios y temas específicos.")
        st.markdown("Haciendo uso de Procesamiento de Lenguaje Natural (NLP) e Inteligencia Artificial (IA), se analizará la información extraída de Twitter (X.com), la cual será mostrada en las distintas secciones.")
        st.title("Contenidos de la sección de análisis:")
        st.markdown("Sección 1: Conteo de emociones" )
        st.markdown("Sección 2: Nube de palabras con keywords" )
        st.markdown("Sección 3: Relación entre sentimientos positivos y negativos" )
        st.markdown( "Sección 4: Gráfico de barras de palabras clave en comentarios negativos" )
        st.markdown("Sección 5: Muestra de ejemplos reales" )
        st.markdown("Sección 6: Conclusión")

    elif option == "Analisis":
        analisis()

    elif option == "License":
        st.subheader("Licencia")
        st.text("""MIT License

Copyright (c) 2025 DaniTM06

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.""")

if __name__ == '__main__':
    main()
