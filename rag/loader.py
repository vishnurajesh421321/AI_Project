from langchain_community.document_loaders import PyMuPDFLoader

FILE_PATH = f"uploads/{}" # need to call this when upload api stores the file to uploades folder
loader = PyMuPDFLoader(FILE_PATH)
