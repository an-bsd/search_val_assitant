import chromadb
from pathlib import Path
from pypdf import PdfReader
from docx import Document
from pptx import Presentation
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Connect to ChromaDB
client = chromadb.PersistentClient(path="data/chroma")

# 2. Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# 3. Build Search and Validation collections
for category in ["search", "validation"]:

    collection_name = f"{category}_guidelines"

    # Remove old collection before rebuilding
    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(collection_name)

    collection = client.create_collection(name=collection_name)

    # 4. Read documents
    for file in Path(f"guidelines/{category}").iterdir():

        if file.suffix.lower() == ".pdf":
            text = "\n".join(
                page.extract_text() or ""
                for page in PdfReader(file).pages
            )

        elif file.suffix.lower() == ".docx":
            text = "\n".join(
                p.text for p in Document(file).paragraphs
            )

        elif file.suffix.lower() == ".pptx":
            text = "\n".join(
                shape.text
                for slide in Presentation(file).slides
                for shape in slide.shapes
                if shape.has_text_frame
            )

        elif file.suffix.lower() == ".txt":
            text = file.read_text(encoding="utf-8")

        else:
            continue

        # 5. Split document into overlapping chunks
        chunks = text_splitter.split_text(text)

        # 6. Store chunks and embeddings
        for i in range(0, len(chunks), 100):

            batch = chunks[i:i + 100]

            collection.add(
                ids=[f"{file.name}_{i+j}" for j in range(len(batch))],
                documents=batch,
                metadatas=[{"source": file.name}] * len(batch)
            )