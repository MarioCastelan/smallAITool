import pickle
import pandas as pd
import streamlit as st
from apyori import apriori as Apriori
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator
from mpl_toolkits.mplot3d import Axes3D
FILE_TYPES = ["csv", "xlsx", "xls", "txt"]


# Funcion encargada de la interfaz grafica del algoritmo apriori ademas de
# calcular las reglas de asociacion


def apriori(dataFrame):
    reglas = dataFrame.values.tolist()
    st.title("Reglas de asociación: apriori")
    st.subheader("Datos")
    st.dataframe(dataFrame)
    st.subheader("Parametros")
    min_support = st.slider(
        "Soporte minimo", min_value=0.0000, max_value=1.0000, step=0.0001, value=0.0045, format='%.4f')
    min_confidence = st.slider(
        "Confianza minima", min_value=0.0, max_value=1.0, step=0.0001, value=0.2, format='%.4f')
    min_lift = st.slider("Elevacion minima", min_value=0.0,
                         max_value=2.0, step=0.0001, value=3.0, format='%.4f')
    min_lenght = st.slider("Maximo de elementos",
                           min_value=0, max_value=100, step=1, value=2, format='%i')

    st.subheader("Resumen de parametros")
    st.write("Soporte minimo = ", min_support)
    st.write("Confianza minima = ", min_confidence)
    st.write("Elevacion minima = ", min_lift)
    st.write("Maximo de elementos = ", min_lenght)

    calcular_reglas = st.button("Calcular")
    st.subheader("Reglas")
    try:
        if calcular_reglas:
            # Calculo de las reglas
            reglas = Apriori(reglas, min_support=min_support,
                             min_confidence=min_confidence, min_lift=min_lift, min_lenght=min_lenght)

            for item in reglas:

                # Primer índice de la lista interna
                # Contiene un elemento agrega otro
                pair = item[0]
                items = [x for x in pair]
                st.write(items[0], " ⮕ ", items[1])

                # Segundo índice de la lista interna
                st.write("Soporte = ", item[1])

                # Tercer índice de la lista interna
                st.write("Confianza = ", item[2][0][2])
                st.write("Lift = ", item[2][0][3])
                st.markdown("---")
    except:
        st.warning("Revise los datos de entrada o los parametros")
# Funcion encargada de la interfaz grafica del algoritmo correlacion de pearson


def correlacion_pearson(dataFrame):

    # Funcion encargada de retornar la matriz de correlaciones
    def matriz_correlacion(dataFrame):
        return dataFrame.corr(method='pearson')
    try:
        st.title("Correlación de Pearson")
        st.subheader("Datos")
        st.dataframe(dataFrame)
        if st.checkbox("Matriz de correlación"):
            matriz = matriz_correlacion(dataFrame)
            st.dataframe(matriz.style.highlight_max(axis=0))
            st.subheader("Visualización de correlaciones")

            if st.checkbox("heatmap"):
                state = st.success("Creando heatmap...")
                fig = plt.figure()
                sb.heatmap(matriz, annot=True)

                st.pyplot(fig)

                state.empty()

            st.subheader("Evaluación visual")
            variables = matriz.columns.tolist()
            variable_1 = st.selectbox("Variable 1", variables)
            variable_2 = st.selectbox("Variable 2", variables)

            if st.button("Visualizar"):
                with st.spinner('Creando grafica'):
                    figure, ax = plt.subplots()
                    ax.plot(dataFrame[variable_1], dataFrame[variable_2], 'b*')
                    ax.set_xlabel(variable_1)
                    ax.set_ylabel(variable_2)
                    st.pyplot(figure)
    except:
        st.warning("Revise los datos de entrada")
# Esta funcion retorna un dataFrame de la matriz con alguna metrica de distancia


def distancias(Datos, opcion):
    matriz_distancias = []
    if opcion == "Euclidiana":
        for row in Datos.values:
            matriz_distancias.append(
                [distance.euclidean(row, renglon) for renglon in Datos.values])

    elif opcion == "Chebyshev":
        for row in Datos.values:
            matriz_distancias.append(
                [distance.chebyshev(row, renglon) for renglon in Datos.values])

    elif opcion == "Manhattan":
        for row in Datos.values:
            matriz_distancias.append(
                [distance.cityblock(row, renglon) for renglon in Datos.values])

    elif opcion == "Minkowski":
        for row in Datos.values:
            matriz_distancias.append(
                [distance.minkowski(row, renglon) for renglon in Datos.values])

    return pd.DataFrame(matriz_distancias).style.highlight_min(axis=0)

# Esta funcion retorna on objeto de matplolib para graficar el efecto del codo, ademas retorno
# un numero sugerido de clusters


def elbow_method(k, variables_modelo):
    SSE = []
    for i in range(1, k):
        km = KMeans(n_clusters=i, random_state=0)
        km.fit(variables_modelo)
        SSE.append(km.inertia_)

    figure = plt.figure()
    ax = plt.subplot()
    ax.plot(range(1, k), SSE, marker='o')
    ax.set_xlabel('Cantidad de clusters *k*')
    ax.set_ylabel("SEE")
    ax.set_title('Elbow Method')

    k_sugerido = KneeLocator(
        range(1, k), SSE, curve="convex", direction="decreasing")
    return (figure, k_sugerido.elbow)

# esta funcion grafica en 3d o en 2d los centroides y los datos segun las variables que se escogieron
# en la interfaz


def graficar_clusters(dimension, labels, centroides, datos):
    colores = ['red', 'green', 'blue', 'cyan',
               'yellow', 'black', 'magenta']
    color1 = []
    color2 = []
    i = 0
    for row in labels:
        color1.append(colores[row])
    color2 = colores[-len(centroides):]

    if dimension == '3D':
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(datos.iloc[:, 0], datos.iloc[:, 1],
                   datos.iloc[:, 2], c=color1, s=60)
        ax.scatter(centroides.iloc[:, 0], centroides.iloc[:, 1],
                   centroides.iloc[:, 2], marker='*', c=color2, s=1000)
    else:

        fig = plt.figure()
        ax = plt.subplot()
        ax.scatter(datos.iloc[:, 0], datos.iloc[:, 1], c=color1, s=70)
        ax.scatter(
            centroides.iloc[:, 0], centroides.iloc[:, 1], marker='*', c=color2, s=1000)
    return fig

# Esta funcion crea los clusters de acuerdo a las variables seleccionadas y al numero de clusters especificado


def crear_clusters(k, variables_modelo):
    MParticional = KMeans(n_clusters=k, random_state=0).fit(variables_modelo)
    MParticional.predict(variables_modelo)
    return MParticional

# Funcion encargada de la interfaz grafica de las metricas de similitud


def metricas_similitud(dataFrame):
    st.title("Metricas de similitud")
    st.subheader("Datos")
    st.dataframe(dataFrame)
    if not dataFrame.empty:
        distancia = st.selectbox("Elige una metrica", [
            "Euclidiana", "Chebyshev", "Manhattan", "Minkowski"])

        st.subheader("Matriz de distancias")
        if st.checkbox("Mostrar"):
            # matriz = distancias(dataFrame, distancia)
            try:
                st.dataframe(distancias(dataFrame, distancia))
            except:
                st.warning('Revise los datos de entrada')

# Funcion encargada de la interfaz grafica del clustering particional


def clustering_particional(dataFrame):
    datos = dataFrame
    st.title("Clustering particional")
    st.subheader("Datos")
    st.dataframe(datos.head(50))
    st.subheader("Selección de variables")
    variables = st.multiselect(
        "Variables", options=datos.columns.tolist())
    variables_modelo = datos.loc[:, variables]
    try:
        if st.checkbox("Elbow method"):
            kn = st.slider("Numero de clusters K", min_value=2,
                           max_value=100, step=1, value=15)
            st.write("k=", kn)
            if st.button("Calcular"):
                figure, k_sugerido = elbow_method(kn, variables_modelo)

                st.pyplot(figure)
                st.write("numero de clusters sugerido: k = ", k_sugerido)

        if st.checkbox("Modelo"):
            k = st.slider("k", min_value=2, max_value=100, step=1)
            st.write("k = ", k)
            # if st.button("Calcular", "modelo1-key"):
            Mparticional = crear_clusters(k, variables_modelo)
            datos['ClusterP'] = Mparticional.labels_
            st.dataframe(datos)
            st.subheader("Numero de elementos por cluster")
            st.dataframe(datos.groupby(['ClusterP'])['ClusterP'].count())
            st.subheader("Centroides")
            centroides = pd.DataFrame(
                Mparticional.cluster_centers_.round(4), columns=variables)
            st.dataframe(centroides)
            st.subheader("Elementos mas cercanos a los centroides")
            Cercanos, _ = pairwise_distances_argmin_min(
                centroides, variables_modelo)
            j = 0
            for row in Cercanos:
                st.write('c = ', j, 'e = ', dataFrame.index.values[row])
                j = j + 1
            if len(variables) > 2:
                st.subheader("Visualizar graficamente los clusters")
                dimension = st.selectbox("Dimensión", ['3D', '2D'])
                if dimension == '3D':
                    x_3d = st.selectbox("dimensión x", variables)
                    y_3d = st.selectbox("dimensión y", variables)
                    z_3d = st.selectbox("dimensión z", variables)
                    if st.button("Visualizar", '1'):
                        fig = graficar_clusters('3D', Mparticional.labels_, centroides.loc[:, [
                                                x_3d, y_3d, z_3d]], datos.loc[:, [x_3d, y_3d, z_3d]])
                        st.pyplot(fig)
                elif dimension == '2D':
                    x_2d = st.selectbox("dimensión x", variables)
                    y_2d = st.selectbox("dimensión y", variables)
                    if st.button('Visualizar', '2'):
                        fig = graficar_clusters('2D', Mparticional.labels_, centroides.loc[:, [
                                                x_2d, y_2d]], datos.loc[:, [x_2d, y_2d]])
                        st.pyplot(fig)
    except:
        st.warning("Revise los datos de entrada")
# Funcion encargada de la interfaz grafica de la prediccion a base de un modelo logistico


def regresion_logistica():
    # Funcion encargada de cargar el modelo logistico
    @st.cache
    def load_model():
        loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
        return loaded_model

    loaded_model = load_model()
    st.title("Sistema de inferencia basado en un modelo de regresión logistica")
    if st.checkbox("Información del modelo"):
        st.markdown("**a+bX** = 11.72346938 - 0.1908854*Texture* – 0.0106113  *Area* – 2.27353782 *Compactness* – 3.0783869  *Concavity* – 0.87661688 *Symmetry* – 0.2123106 *FractalDimension*")
        st.write('Exactitud = ', (39+67)/(39+67+2+6))
        st.write('Precisión = ', (39)/(39+2))
        st.write('Tasa de error = ', (2+6)/(39+67+2+6))
        st.write('Sensibilidad = ', (39)/(39+6))
        st.write('Especificidad = ', (67)/(67+2))

    col1, col2 = st.beta_columns(2)
    with col1:
        Texture = st.number_input("Texture", min_value=0.000000, format='%.6f')
        Area = st.number_input("Area", min_value=0.000000, format='%.6f')
        Compactness = st.number_input(
            "Compactness", min_value=0.000000, max_value=1.000000,  format='%.6f')
    with col2:
        Concavity = st.number_input(
            "Concavity", min_value=0.000000, max_value=1.000000, format='%.6f')
        Symmetry = st.number_input(
            "Symmetry", min_value=0.000000, max_value=1.000000, format='%.6f')
        FractalDimension = st.number_input(
            "FractalDimension", min_value=0.000000, format='%.6f')

    nuevo_paciente = pd.DataFrame({'Texture': [Texture], 'Area': [Area], 'Compactness': [Compactness],
                                   'Concavity': [Concavity], 'Symmetry': [Symmetry], 'FractalDimension': [FractalDimension]})

    if st.button('Diagnosis'):
        prediccion = loaded_model.predict(nuevo_paciente)
        if prediccion == 1:
            st.subheader('Benigno')
        if prediccion == 0:
            st.subheader('Maligno')

# Funcion encargada de crear un dataFrame a base del archivo de datos cargado


def load_dataFrame(uploaded_file, header, index, sep):
    def f(x): return 0 if x else None

    if uploaded_file.name[-3:] == "csv":
        dataFrame = pd.read_csv(uploaded_file, header=f(
            header), index_col=f(index), keep_default_na=False)
    elif uploaded_file.name[-3:] == "txt":
        dataFrame = pd.read_table(
            uploaded_file, header=f(header), index_col=f(index), sep=sep, engine='python')
    elif uploaded_file.name[-3:] in ["xlsx", "xls"]:
        dataFrame = pd.read_excel(
            uploaded_file, header=f(header), index_col=f(index))
    else:
        dataFrame = None
    return dataFrame

# Funcion de entrada, encargada de llamar la funcion del algoritmo escogido


def main():
    dataFrame = None
    st.sidebar.header("Algoritmo")

    app_menu = st.sidebar.selectbox(
        "", ["Apriori", "Correlación de Pearson", "Metricas de similitud", "Clustering particional", "Regresión logistica"])

    st.sidebar.subheader("Acceso a los datos")
    uploaded_file = st.sidebar.file_uploader(
        "", type=FILE_TYPES)

    header = st.sidebar.checkbox("Encabezado")
    index = st.sidebar.checkbox("Indice")
    sep = st.sidebar.text_input("Separador", value=r"\t", max_chars=2)

    if uploaded_file is not None:
        dataFrame = load_dataFrame(uploaded_file, header, index, sep)

    if dataFrame is not None:
        if app_menu == "Apriori":
            apriori(dataFrame)
        elif app_menu == "Correlación de Pearson":
            correlacion_pearson(dataFrame)
        elif app_menu == "Metricas de similitud":
            metricas_similitud(dataFrame)
        elif app_menu == "Clustering particional":
            clustering_particional(dataFrame)
        elif app_menu == "Regresión logistica":
            regresion_logistica()
    elif app_menu == "Regresión logistica":
        regresion_logistica()


if __name__ == "__main__":
    main()
