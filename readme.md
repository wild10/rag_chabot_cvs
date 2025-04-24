# Chatbot RAG para Análisis de CVs

Este proyecto implementa un chatbot basado en Retrieval-Augmented Generation (RAG) para analizar y responder preguntas sobre currículums vitae (CVs) de profesionales. Utiliza modelos de lenguaje como LLaMA y DeepSeek junto con herramientas de indexación y recuperación de datos como Pinecone y LangChain.
[Arquitectura](arequitectura.png)](arquitectura_rag.pdf)

## Características
- **Carga de CVs** en formato PDF.
- **Segmentación y procesamiento de texto** usando `langchain-text-splitters`.
- **Indexación eficiente** de documentos con Pinecone.
- **Generación de respuestas inteligentes** con modelos de lenguaje a través de Ollama.
- **Interfaz de usuario** con gradio para interacciones fáciles.

## Instalación
Sigue los siguientes pasos para configurar el entorno:

### 1. Clonar el repositorio
```bash
cd ~/Documents/projects/
git clone <this repository>
cd CVs_caleidos
```

### 2. Crear y activar el entorno con Conda
se debe crear un ambiente virtual con anaconda con elnombre llm_env y ejecutar esto.
```bash
source /home/wilderd/anaconda3/bin/activate
conda activate llm_env
```

### 3. Instalar dependencias
```bash
pip install pinecone langchain streamlit ollama
pip install pymupdf
pip install -qU langchain-text-splitters
pip install -U langchain_ollama
pip install -qU langchain-pinecone
pip install -U langchain-community
```

## Uso
Para ejecutar el chatbot, usa el siguiente comando:
```bash
streamlit run main.py
```

Esto iniciará la interfaz en tu navegador, tipo local host donde puedes hacer preguntas.

## Estructura del Proyecto
```
CVs_caleidos/
│── main.py                 # Aplicación principal con Streamlit
│── aws_bot3.py              # Configuración de API keys y credenciales
│── data/                  # Carpeta para almacenar los CVs
│── README.md              # Documentación del proyecto
```

## Licencia
---
**Autor:** Errol W. Mamani Condori all right reserved.

