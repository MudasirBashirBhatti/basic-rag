from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("resume.pdf") # Initialize the PDF loader with the path to your PDF file
pages = loader.load() # Load the PDF and split it into pages

print(f"Total Pages: {len(pages)}")
print(pages[0].page_content[:500]) # Print the first 500 characters of the first page's content