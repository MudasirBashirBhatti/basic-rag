from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

loader = PyPDFLoader("resume.pdf") # Initialize the PDF loader with the path to your PDF file
pages = loader.load() # Load the PDF and split it into pages

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) # Initialize the text splitter with a chunk size of 500 characters and an overlap of 50 characters between chunks
chunks = text_splitter.split_documents(pages) # Split the pages into smaller chunks of text

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # Initialize the embedding model using a pre-trained model from Hugging Face

sample_text = chunks[0].page_content
vector = embedding_model.embed_query(sample_text) # Generate an embedding vector for the sample text chunk

print(f"Text ka pehla hissa: {sample_text[:50]}...")
print(f"Vector ki length: {len(vector)}") # Ye aksar 384 ya 768 hoti hai
print(f"Vector ke pehle 5 numbers: {vector[:5]}")