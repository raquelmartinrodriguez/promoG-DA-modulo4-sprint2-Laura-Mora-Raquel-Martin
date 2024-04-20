#IMPORTACIONES
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
pd.set_option('display.max_columns', None) 
import warnings
warnings.filterwarnings("ignore")
#------
#función para leer los CSV
def leer_csv(carpeta, nombre_archivo, drop_unnamed=True):
    '''
    Abre y lee los documentos CSV. 
        Si el archivo se encuentra, lo abre en un DataFrame de pandas y lo devuelve. 
        Si el archivo no se encuentra, la función imprime un mensaje indicando que el archivo no se encontró y devuelve None.
    
    Esta función toma el nombre de una carpeta y el nombre de archivo como entrada y lee el archivo CSV.
    
    Args:
        carpeta (str): El nombre de la carpeta donde se encuentra el archivo.
        nombre_archivo (str): El nombre del archivo csv sin la extensión.
        drop_unnamed (bool, opcional): Indica si se debe excluir la columna 'Unnamed' al leer el archivo CSV. Por defecto es True.

    Returns:
        pd.DataFrame: Un DataFrame que contiene los datos del archivo csv, o None si el archivo no se encuentra.
    '''
    try:
        df = pd.read_csv(f'{carpeta}/{nombre_archivo}.csv')
        print('El archivo se ha cargado con éxito.')

        if drop_unnamed:
            df = df.drop(columns=["Unnamed: 0"], errors="ignore")  # Ignora si la columna no existe
        
        return df
    
    except FileNotFoundError:
        print('No se ha encontrado el archivo')
        return None
#------
#función pata guardar los CSV
def guardado_csv(dataframe, carpeta, nombre_archivo):
    '''
    Guarda un DataFrame en un archivo CSV.

    Esta función toma un DataFrame, el nombre de una carpeta y un nombre de archivo como entrada y guarda el DataFrame en un archivo CSV.
    
    Args:
        dataframe (pd.DataFrame): El DataFrame que se va a guardar.
        carpeta (str.): El nombre de la carpeta donde se desee guardar
        nombre (str): El nombre del archivo CSV resultante, sin la extensión ".csv".

    Returns:
        None: La función no devuelve ningún valor, pero imprime un mensaje indicando si el guardado fue exitoso o si ocurrió algún error.
    '''
    try:
        dataframe.to_csv(f'{carpeta}/{nombre_archivo}.csv')
        print('El dataframe ha sido guardado con éxito.')
    except Exception as e:
        print(f'Error en el guardado: {e}')
#------
#función para capitalizar el nombre de las columnas:
def col_minuscula(dataframe):
    '''
    Pone en minúscula los nombres de las columnas del DataFrame.
    Esta función toma un DataFrame y cambia todos los nombres de las columnas y los pone en minúscula.

    Args:
        df (pd.DataFrame): El DataFrame que se desea modificar.

    Returns:
        None: La función modifica el DataFrame directamente cambiando el formato de sus nombres de columnas.
    '''
    print("Nombres de las columnas antes del cambio:")
    print(dataframe.columns)

    dataframe.columns = dataframe.columns.str.lower()

    print("----")
    print("\nNombres de las columnas después del cambio:")
    print(dataframe.columns)
#------
#función para vonvertir los valores del DF en minúsculas
#Aplica la función str.lower() a cada celda del DataFrame
def minusculas(dataframe):
    '''
    Convierte todos los valores del DataFrame a minúsculas.

    Args:
        df (pd.DataFrame): El DataFrame que se desea modificar.

    Returns:
        None: La función modifica el DataFrame original inplace.
    '''
    
    # Iterar sobre cada columna
    for col in dataframe.columns:
        # Verificar si la columna contiene strings y convertirlos a minúsculas
        if dataframe[col].dtype == 'object':
            dataframe[col] = dataframe[col].str.lower()
#------
#función para modificar los nombres de las columnas
def modificar_columnas(dataframe):
    '''
    Cambia los nombres de las columnas en un DataFrame para reemplazar los espacios por '_'.

    Args:
        df (pd.DataFrame): El DataFrame cuyas columnas se van a modificar.

    Returns:
        None: La función modifica el DataFrame original inplace.
    '''
    print(f"Las columnas antes de ser cambiadas son: {list(dataframe.columns)}")
    dataframe.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)
    print("---")
    print(f"Las columnas después de ser cambiadas son: {list(dataframe.columns)}")
#------
#función para cambiar el nombre de las columnas
def cambiar_nombres_columnas(dataframe, nombres_nuevos):
    '''
    Cambia los nombres de las columnas en un DataFrame.

    Args:
        df (pd.DataFrame): El DataFrame cuyas columnas se van a renombrar.
    nombres_nuevos (dict): Un diccionario donde las claves son los nombres de las columnas actuales
                        y los valores son los nuevos nombres que deseas asignar.

    Returns:
    None: La función modifica el DataFrame original inplace.
    '''
    dataframe.rename(columns=nombres_nuevos, inplace=True)
#------
#función para cambiar tipo de dato
def cambiar_dato(dataframe, columnas, nuevo_tipo):
    '''
    Cambia el tipo de dato de las columnas en un DataFrame.

    Args:
        df (pd.DataFrame): El DataFrame en el que se van a cambiar los tipos de datos.
        columnas (str o list): Un string o lista de strings que contiene los nombres de las columnas cuyo tipo de dato se va a cambiar.
        nuevo_tipo (str): El nuevo tipo de dato al que se van a convertir las columnas (por ejemplo, 'int', 'float', 'object', etc.).

    Returns:
        None: La función modifica el DataFrame original inplace.
    '''
    print("Antes del cambio de tipo:")
    display(pd.DataFrame(dataframe[columnas].dtypes, columns=["type"]))
    print("----")
    
    for col in columnas:
        if nuevo_tipo in ['int', 'float']:
            if nuevo_tipo == 'int':
                dataframe[col] = pd.to_numeric(dataframe[col].fillna(0), errors='coerce').astype(int)
            else:
                dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
        else:
            if nuevo_tipo == 'object':
                dataframe[col] = dataframe[col].astype(nuevo_tipo)
            else:
                print("El nuevo tipo de dato no es válido. Se mantendrá el tipo original.")
    
    print("Después del cambio de tipo, las columnas quedan:")
    display(pd.DataFrame(dataframe[columnas].dtypes, columns=["type"]))

def cambiar_tipo(dataframe, columnas, nuevo_tipo):
    '''
    Cambia el tipo de dato de las columnas en un DataFrame.

    Args:
        df (pd.DataFrame): El DataFrame en el que se van a cambiar los tipos de datos.
        columnas (str o list): Un string o lista de strings que contiene los nombres de las columnas cuyo tipo de dato se va a cambiar.
        nuevo_tipo (str): El nuevo tipo de dato al que se van a convertir las columnas (por ejemplo, 'int', 'float', 'object', etc.).

    Returns:
        None: La función modifica el DataFrame original inplace.
    '''
    print("Antes del cambio de tipo:")
    display(pd.DataFrame(dataframe[columnas].dtypes, columns=["type"]))
    print("----")
    
    for col in columnas:
        if nuevo_tipo in ['int', 'float']:
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
        elif nuevo_tipo == 'object':
            dataframe[col] = dataframe[col].astype(nuevo_tipo)
        else:
            print("El nuevo tipo de dato no es válido. Se mantendrá el tipo original.")
    
    print("Después del cambio de tipo, las columnas quedan:")
    display(pd.DataFrame(dataframe[columnas].dtypes, columns=["type"]))
#------
#función para ver si hay duplicados:
def explorar_duplicados(dataframe):
    """
    Esta función muestra la cantidad de duplicados en el conjunto de datos.

    Args:
        dataframe (pandas.DataFrame): El DataFrame que se va a explorar en busca de duplicados.

    Returns:
        None
    """
    print(f"Tenemos {dataframe.duplicated().sum()} duplicados en el conjunto de datos.")
#------
#función para eliminar duplicados:
def eliminar_duplicados(dataframe):
    """
    Elimina los duplicados de un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame de pandas.

    Returns:
        None
    """

    print(f"Antes de realizar la eliminación tenemos {dataframe.shape[0]} filas.")
    dataframe.drop_duplicates(inplace=True)
    print(f"Después de la eliminación de duplicados tenemos {dataframe.duplicated().sum()} duplicados en el conjunto de datos.")
    print(f"Después de realizar la eliminación tenemos {dataframe.shape[0]} filas.")
#------
#función para explorar los nulos:
def explorar_nulos(dataframe):
    """
    Esta función presenta un DataFrame que contiene el porcentaje de valores nulos para cada columna.

    Args:
        dataframe (pandas.DataFrame): El DataFrame que se va a explorar en busca de valores nulos.

    Returns: 
        None
    """
    # Nos muestra un DataFrame para los valores nulos
    print("Los nulos que tenemos en el conjunto de datos son:")
    df_nulos = pd.DataFrame(dataframe.isnull().sum() / dataframe.shape[0] * 100, columns=["%_nulos"])
    display(df_nulos[df_nulos["%_nulos"] > 0])
#------
#función para eliminar columnas:
def eliminar_columnas(dataframe, columnas):
    """
    Elimina columnas de un DataFrame.

    Args:
        dataframe (pandas.DataFrame): DataFrame de pandas.
        columnas (str): String o lista de strings de las columnas a eliminar.

    Returns:
        None
    """
    dataframe.drop(columns=columnas, inplace=True)
    print(f"Las columnas actuales son: \n {list(dataframe.columns)}")
#------
#Usar map:
def aplicar_map(dataframe, columna, mapeo):
    """
    Aplica un mapeo a una columna del DataFrame utilizando el método .map().

    Args:
        dataframe (pandas.DataFrame): DataFrame de pandas.
        columna (str): Nombre de la columna en la que se aplicará el mapeo.
        mapeo (dict): Diccionario que contiene el mapeo a aplicar.

    Returns:
        None
    """
    print(f"Valores únicos en la columna '{columna}' antes de aplicar el mapeo: {dataframe[columna].unique()}")
    dataframe[columna] = dataframe[columna].map(mapeo)
    print(f"Valores únicos en la columna '{columna}' después de aplicar el mapeo: {dataframe[columna].unique()}")
#------
def reemplazar(dataframe, columna, mapeo):
    """
    Reemplaza valores en una columna del DataFrame utilizando el método .replace().

    Args:
        dataframe (pandas.DataFrame): DataFrame de pandas.
        columna (str): Nombre de la columna en la que se realizará el reemplazo.
        mapeo (dict): Diccionario que contiene el mapeo de valores a reemplazar.

    Returns:
        None
    """
    # Aplicar el reemplazo usando el método .replace()
    print(f"Valores únicos en la columna '{columna}' antes de aplicar el mapeo: {dataframe[columna].unique()}")
    dataframe[columna] = dataframe[columna].replace(mapeo)
    print(f"Valores únicos en la columna '{columna}' después de aplicar el mapeo: {dataframe[columna].unique()}")
#------
def negativos(df, columna):
    """
    Verifica si hay números negativos en una columna de un DataFrame y, en caso afirmativo, 
    los convierte en su valor absoluto.

    Args:
        df (pandas.DataFrame): DataFrame de pandas.
        columna (str): Nombre de la columna a verificar y transformar.

    Returns:
        None
    """
    if (df[columna] < 0).any():
        print(f"Hay números negativos en la columna {columna}. Se convertirán a su valor absoluto.")
        df[columna] = df[columna].abs()
    else:
        print(f"No hay números negativos en la columna {columna}.")
#------
#función para rellenar nulos:
def rellenar_nulos(dataframe, columnas, valor):
    '''
    Rellena los valores nulos en las columnas especificadas de un DataFrame con un valor específico.

    Args:
        dataframe (pd.DataFrame): El DataFrame en el que se van a rellenar los valores nulos.
        columnas (str o list): Un string o lista de strings que contiene los nombres de las columnas en las que se van a rellenar los valores nulos.
        valor (int, float, str, etc.): El valor con el que se van a rellenar los valores nulos.

    Returns:
        None: La función modifica el DataFrame original inplace.
    '''
    dataframe[columnas] = dataframe[columnas].fillna(valor)
    print(f"El total de nulos en las columnas {columnas} después de aplicar .fillna() es: {dataframe[columnas].isna().sum()}")
#------
def rellenar_mediana(dataframe, columnas):
    """
    Rellena los valores nulos en las columnas especificadas de un DataFrame con la mediana de esas columnas.

    Args:
        dataframe (pd.DataFrame): El DataFrame en el que se van a rellenar los valores nulos.
        columnas (str o list): Un string o lista de strings que contiene los nombres de las columnas en las que se van a rellenar los valores nulos.

    Returns:
        None: La función modifica el DataFrame original inplace.
    """
    mediana = dataframe[columnas].median()  # Calcula la media de las columnas especificadas
    dataframe[columnas] = dataframe[columnas].fillna(mediana)  # Rellena los valores nulos con las medias calculadas
    print(f"Los valores nulos en las columnas {columnas} se han rellenado con la mediana respectiva.")
    print(f"El total de nulos en las columnas {columnas} después de aplicar .fillna() es: {dataframe[columnas].isna().sum()}")
#------
def rellenar_media(dataframe, columnas):
    """
    Rellena los valores nulos en las columnas especificadas de un DataFrame con la mediana de esas columnas.

    Args:
        dataframe (pd.DataFrame): El DataFrame en el que se van a rellenar los valores nulos.
        columnas (str o list): Un string o lista de strings que contiene los nombres de las columnas en las que se van a rellenar los valores nulos.

    Returns:
        None: La función modifica el DataFrame original inplace.
    """
    media = dataframe[columnas].mean()  # Calcula la media de las columnas especificadas
    dataframe[columnas] = dataframe[columnas].fillna(media)  # Rellena los valores nulos con las medias calculadas
    print(f"Los valores nulos en las columnas {columnas} se han rellenado con la media respectiva.")
    print(f"El total de nulos en las columnas {columnas} después de aplicar .fillna() es: {dataframe[columnas].isna().sum()}")
#------
def copiar_columna(dataframe, nombre_columna_original, nombre_nueva_columna):
    """
    Realiza una copia de una columna en un DataFrame de pandas.

    Args:
        dataframe (pd.DataFrame): DataFrame de pandas.
        nombre_columna_original (str): Nombre de la columna que se va a copiar.
        nombre_nueva_columna (str): Nombre de la nueva columna que contendrá la copia.

    Returns:
        None
    """
    dataframe[nombre_nueva_columna] = dataframe[nombre_columna_original].copy()
#------
#función para ver los tipos de las columnas:
def mostrar_tipos(dataframe):
    """
    Muestra el dtype de cada columna de un DataFrame.
    
    Args:
        dataframe (pandas.DataFrame): El DataFrame del que se mostrarán los dtypes de las columnas.
    """
    for columna in dataframe.columns:
        print(f"Columna '{columna}': {dataframe[columna].dtype}")
#------
#función para ver los valores únicos
def valores_unicos(dataframe, columnas):
    '''
    Muestra los valores únicos de las columnas numéricas de un DataFrame.

    Args:
        dataframe (pd.DataFrame): El DataFrame del que se mostrarán los valores únicos.
        columnas (str o list): Un string o lista de strings que contiene los nombres de las columnas numéricas.

    Returns:
        None: La función imprime los valores únicos para cada columna numérica.
    '''
    for col in columnas:
        print(f"La columna {col.upper()} tiene los siguientes valores únicos:")
        display(pd.DataFrame(dataframe[col].value_counts()).head())
        print("----")
#------
#función para ver los estadísticos de las columnas:
def describir_columnas(dataframe, columnas):
    '''
    Muestra las estadísticas descriptivas de las columnas especificadas de un DataFrame.

    Args:
        dataframe (pd.DataFrame): El DataFrame del que se mostrarán las estadísticas descriptivas.
        columnas (str o list): Un string o lista de strings que contiene los nombres de las columnas.

    Returns:
        None: La función imprime las estadísticas descriptivas para cada columna especificada.
    '''
    for col in columnas:
        print(f"Descripción de la columna {col.upper()}:")
        display(dataframe[[col]].describe().T)
        print("\n ----- \n")
#------
