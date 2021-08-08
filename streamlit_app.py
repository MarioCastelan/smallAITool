import pickle
import pandas as pd
import numpy as np   
import streamlit as st
from apyori import apriori as Apriori
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn import model_selection



FILE_TYPES = ["csv", "xlsx", "xls", "txt"]

#  streamlit run .\streamlit_app.py

def analisisExploratorioDatos(dataFrame):
    st.title("Análisis exploratorio de datos (EDA)")
    st.header("Datos")
    st.dataframe(dataFrame)
    st.header("Descripción de la estructura de los datos")
    if st.checkbox("Mostrar", key = "qaz"):
        st.write("Filas: ", dataFrame.shape[0])
        st.write("Columnas: ", dataFrame.shape[1])
        st.subheader("Tipos de datos")
        st.write(pd.DataFrame(dataFrame.dtypes))
    st.markdown("---")
    st.header("Identificación de datos faltantes")
    if st.checkbox("Valores nulos en cada variable"):
        st.write(dataFrame.isnull().sum())
    st.markdown("---")
    st.header("Detección de valores atípicos")
    # if st.checkbox("Mostrar", key = "edc"):
    if st.checkbox("Resumen estadístico de variables numéricas"):
        st.write(dataFrame.describe())
    if st.checkbox("Distribución de variables numéricas"):
        variables_histo = st.multiselect("Seleccione las variables a graficar histograma", options=dataFrame.columns.tolist())
        if st.checkbox("Histogramas"):
            for variable in variables_histo:
                figure = plt.figure()
                plt.hist(dataFrame[variable])
                plt.title(variable)
                plt.grid()
                st.pyplot(figure)
    if st.checkbox("Diagramas para detectar posibles valores atípicos"):
        variables = st.multiselect("Seleccione las variables a graficar", options=dataFrame.columns.tolist())
        if st.checkbox("Boxplot"):
            try:
                for col in variables:
                    fig = plt.figure()
                    sb.boxplot(x = col, data = dataFrame)
                    st.write(fig)
            except:
                st.warning("Elija una variable numérica ")
    st.markdown("---")
    st.header("Identificación de relaciones entre variables")
    if st.checkbox("Matriz de correlación"):
        matriz = dataFrame.corr(method='pearson')
        st.dataframe(matriz.style.highlight_max(axis=0))
        if st.checkbox("heatmap"):
            state = st.success("Creando heatmap...")
            fig = plt.figure()
            sb.heatmap(matriz, cmap='RdBu_r', annot=True, mask=np.triu(matriz))
            st.pyplot(fig)
            state.empty()

    if st.checkbox("Evaluación visual"):
        variables = dataFrame.columns.tolist()
        variable_1 = st.selectbox("Variable 1", variables)
        variable_2 = st.selectbox("Variable 2", variables)
        if st.button("Visualizar"):
            with st.spinner('Creando grafica'):
                figure, ax = plt.subplots()
                ax.plot(dataFrame[variable_1], dataFrame[variable_2], 'b*')
                ax.set_xlabel(variable_1)
                ax.set_ylabel(variable_2)
                st.pyplot(figure)


def analisisComponentesPrincipales(dataFrame):

    st.title("Análisis de componentes principales")
    st.header("Datos")
    st.dataframe(dataFrame)
    st.markdown("---")
    st.header("Estandarización de los datos")
    variables = st.multiselect("Elija las variables a no considerar en el análisis, teniendo en cuenta que también se debe seleccionar las variables que no son numéricas ", 
    options=dataFrame.columns.tolist())
    try:
        normalizar = StandardScaler()                       # Se instancia el objeto StandardScaler 
        matriz = dataFrame.drop(variables, axis=1)      # Se quita la variable dependiente "Y"
        normalizar.fit(matriz)                           # Se calcula la media y desviación para cada dimensión
        MNormalizada = normalizar.transform(matriz)      # Se normalizan los datos 
        MNormalizada = pd.DataFrame(MNormalizada, columns=matriz.columns)
        st.subheader("Matriz normalizada")
        st.write(MNormalizada)
        st.markdown("---")
        st.header("eigen-vectores y eigen-valores")
        pca = PCA(n_components=None)                # Se instancia el objeto PCA           
        pca.fit(MNormalizada)                       # Se obtiene los componentes
        pca.transform(MNormalizada)                  # Se convierte los datos con las nuevas dimensiones
        if st.checkbox("Mostrar eigen-vectores y eigen-valores"):
            st.subheader("eigen-vectores")
            st.write(pd.DataFrame(pca.components_, columns=matriz.columns))
            st.subheader("eigen-valores")
            st.dataframe(pca.explained_variance_)
        st.header("Decisión del número de componentes principales")
        fig = plt.figure()
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Número de componentes')
        plt.ylabel('Varianza acumulada')
        plt.grid()
        st.pyplot(fig)
        Ncomponentes = st.slider("Numero de componentes", min_value=1, max_value=len(matriz.columns), step=1, value=1)
        Varianza = pca.explained_variance_ratio_
        st.write('Varianza acumulada:', sum(Varianza[0:Ncomponentes]))   
        st.header("Examinación de proporción de relevancias –cargas–")
        st.write("Valores absolutos de los componentes principales seleccionados")
        st.write("*Se resaltan las variables con mayor valor en cada componentes como una ayuda al usuario ")
        selectedComponents = pd.DataFrame(abs(pca.components_), columns=matriz.columns).head(Ncomponentes)

        st.write(selectedComponents.style.highlight_max(axis=1))
        # st.write(pd.DataFrame(abs(pca.components_), columns=matriz.columns).head(Ncomponentes))
    except:
        st.warning("Hay componentes no numéricas en los datos de entrada ")
    

def regresion_logistica(dataFrame):
    Clasificacion = linear_model.LogisticRegression() 
    st.title("Regresión logística")
    st.header("Datos")
    st.dataframe(dataFrame)
    st.header("Definición de variables predictoras y variable clase")
    variablesPredictoras = st.multiselect("Variables predictoras", options=dataFrame.columns.tolist())
    X = dataFrame.loc[:, variablesPredictoras]
    variableClase = st.selectbox("Variable clase", [x for x in dataFrame.columns.tolist() if x not in variablesPredictoras])
    Y = dataFrame.loc[:, variableClase]
    validacionPorcentaje = st.slider("Porcentaje del conjunto de datos para validación", min_value= 1, max_value=100, value=20, step =1)
    st.write("Entrenamiento: ", 100-validacionPorcentaje, "%")
    st.write("Validación: ", validacionPorcentaje, "%")
    if len(variablesPredictoras) < 1:
        st.warning("Elija las variables predictoras ")
    else:
        seed = 1234
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validacionPorcentaje/100, random_state=seed, shuffle = True)
        #Se entrena el modelo a partir de los datos de entrada
        Clasificacion.fit(X_train, Y_train)
        #Matriz de clasificación
        PrediccionesNuevas = Clasificacion.predict(X_validation)
        confusion_matrix = pd.crosstab(Y_validation.ravel(), PrediccionesNuevas, rownames=['Real'], colnames=['Clasificación'])
        st.header("Validación del modelo")
        st.subheader("Matriz de confusión")
        st.write(confusion_matrix)
        st.subheader("Evaluación del modelo")
        exactitud = (confusion_matrix[0][0] + confusion_matrix[1][1])/confusion_matrix.values.sum()
        tasaError = (confusion_matrix[0][1] + confusion_matrix[1][0])/confusion_matrix.values.sum()
        precision = confusion_matrix[0][0]/(confusion_matrix[0][0] + confusion_matrix[0][1])
        sensibilidad = confusion_matrix[0][0]/(confusion_matrix[0][0] + confusion_matrix[1][0])
        especifidad =  confusion_matrix[1][1]/(confusion_matrix[1][1] + confusion_matrix[0][1])
        st.write("Exactitud = ", exactitud)
        st.write("Precisión = ", precision)
        st.write("Tasa de error = ", tasaError)
        st.write("Sensibilidad = ", sensibilidad)
        st.write("Especificidad = ", especifidad)
        if st.checkbox("Mostrar intercepto y coeficientes"):
            st.write("Intercepto: ", Clasificacion.intercept_)
            st.write("Coeficientes: ", pd.DataFrame(Clasificacion.coef_, columns=variablesPredictoras))
        st.header("Sistema de inferencia basado en el modelo de regresión logística desarrollado")
        if st.checkbox("Ejecutar"):
            prediccionEntradas = {}
            for variable in variablesPredictoras:
                prediccionEntradas[variable] = [st.number_input(variable, format='%.6f')]
            if st.button("Clasificar"):
                st.subheader(Clasificacion.predict(pd.DataFrame(prediccionEntradas))[0])



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
                sb.heatmap(matriz, cmap='RdBu_r', annot=True, mask=np.triu(matriz))

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


def SistemaDeInferencia():
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
        "", ["Análisis exploratorio de datos (EDA)", "Análisis de componentes principales", "Clustering particional", 
        "Regresión logistica", "Apriori", "Correlación de Pearson", "Metricas de similitud", "Sistema de inferencia"])

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
            regresion_logistica(dataFrame)
        elif app_menu == "Análisis exploratorio de datos (EDA)":
            analisisExploratorioDatos(dataFrame)
        elif app_menu == "Análisis de componentes principales":
            analisisComponentesPrincipales(dataFrame)
        elif app_menu == "Sistema de inferencia":
            SistemaDeInferencia()



if __name__ == "__main__":
    main()
