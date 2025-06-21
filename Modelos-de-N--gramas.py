import nltk                  # Biblioteca para procesamiento de lenguaje natural
import string                # Para manejar caracteres especiales y signos de puntuación
import matplotlib.pyplot as plt  # Para crear gráficos y visualizar datos
from sklearn.feature_extraction.text import CountVectorizer  # Para contar palabras y n-gramas en texto
from nltk.corpus import stopwords  # Lista de palabras comunes que no aportan mucho significado
import re                    # Para buscar y reemplazar texto usando patrones (expresiones regulares)

nltk.download('stopwords')

# Abrir el archivo con el corpus de texto y leer todo su contenido
with open("CorpusEducacion.txt", encoding="latin-1") as archivo:
    texto = archivo.read()

# Convertir todo el texto a minúsculas
texto = texto.lower()

# Crear un conjunto con las palabras vacías (stopwords)
stop_words = set(stopwords.words("spanish"))

# Dividir el texto en oraciones usando el punto como separador
oraciones = texto.split(".")

# Función para limpiar cada oración:
def limpiar_texto(texto):
    texto = texto.strip()
    texto = re.sub(f"[{string.punctuation}]", " ", texto)  # Eliminar puntuación
    palabras = texto.split()  # Dividir en palabras
    # Filtrar palabras que no sean stopwords y que sean solo letras
    palabras_limpias = [p for p in palabras if p not in stop_words and p.isalpha()]
    return " ".join(palabras_limpias)  # Unir de nuevo en un string limpio

# Aplicar la función limpiar_texto a cada oración que no esté vacía
oraciones_limpias = [limpiar_texto(ora) for ora in oraciones if len(ora.strip()) > 0]

# Crear un vectorizador para bigramas (pares de palabras) que aparezcan al menos 2 veces
vectorizer2 = CountVectorizer(ngram_range=(2,2), min_df=2)
X2 = vectorizer2.fit_transform(oraciones_limpias)  # Ajustar y transformar el texto limpio
bigrama_frecuencias = X2.toarray().sum(axis=0)     # Sumar frecuencias de cada bigrama
bigrama_vocabulario = vectorizer2.get_feature_names_out()  # Obtener los bigramas

# Igual para trigramas (tripletas de palabras) que aparezcan al menos 2 veces
vectorizer3 = CountVectorizer(ngram_range=(3,3), min_df=2)
X3 = vectorizer3.fit_transform(oraciones_limpias)
trigrama_frecuencias = X3.toarray().sum(axis=0)
trigrama_vocabulario = vectorizer3.get_feature_names_out()

# Imprimir los 10 bigramas más frecuentes con su cantidad
print("Top 10 Bigramas:")
for i in bigrama_frecuencias.argsort()[::-1][:10]:
    print(bigrama_vocabulario[i], "->", bigrama_frecuencias[i])

# Imprimir los 10 trigramas más frecuentes con su cantidad
print("\nTop 10 Trigramas:")
for i in trigrama_frecuencias.argsort()[::-1][:10]:
    print(trigrama_vocabulario[i], "->", trigrama_frecuencias[i])

# Graficar los 10 bigramas más frecuentes con barras horizontales
plt.figure(figsize=(10,5))
plt.barh(bigrama_vocabulario[bigrama_frecuencias.argsort()[::-1][:10]], 
         bigrama_frecuencias[bigrama_frecuencias.argsort()[::-1][:10]],
         color='yellow')
plt.title("Top 10 Bigramas")
plt.xlabel("Frecuencia")
plt.tight_layout()
plt.show()

# Graficar los 10 trigramas más frecuentes con barras horizontales
plt.figure(figsize=(10,5))
plt.barh(trigrama_vocabulario[trigrama_frecuencias.argsort()[::-1][:10]], 
         trigrama_frecuencias[trigrama_frecuencias.argsort()[::-1][:10]],
         color='green')
plt.title("Top 10 Trigramas")
plt.xlabel("Frecuencia")
plt.tight_layout()
plt.show()
