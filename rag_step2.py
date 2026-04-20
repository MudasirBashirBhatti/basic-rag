from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("resume.pdf") # Initialize the PDF loader with the path to your PDF file
pages = loader.load() # Load the PDF and split it into pages

print(f"Total Pages: {len(pages)}")
print(pages[0].page_content[:500]) # Print the first 500 characters of the first page's content

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
chunks = text_splitter.split_documents(pages)

print(f"Total Chunks: {len(chunks)}")