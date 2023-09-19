from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader

from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings

#loader 
loader = DirectoryLoader('docs',glob='./*.pdf',loader_cls=PyPDFLoader)

documents = loader.load()
print(len(documents))
#splitting the text into
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

from langchain.embeddings import HuggingFaceInstructEmbeddings

instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
# Embed the texts
embeddings = instructor_embeddings.embed_documents(texts)

# Create a FAISS index
dimension = embeddings.shape[1]
index = FAISS.IndexFlatL2(dimension)

# Add embeddings to the index
index.add(embeddings)

# Persist the index to disk
persist_directory = 'db'

FAISS.write_index(index, persist_directory + '/faiss.index')

# Custom FAISS retriever
def faiss_retriever(query_embedding, k=5):
    _, indices = index.search(query_embedding, k)
    return [texts[i] for i in indices[0]]

# Use custom retriever
retriever = faiss_retriever
