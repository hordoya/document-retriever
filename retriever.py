import os
import pickle
from typing import List

import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

class Retriever:


    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.embeddings = []

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks

    def _load_file(self, filepath: str) -> str:
       
        if filepath.endswith((".txt", ".md")):
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        elif filepath.endswith(".pdf"):
            reader = PdfReader(filepath)
            return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        else:
            raise ValueError("Unsupported file format.")

    def add_documents(self, filepaths: List[str]):
 
        for filepath in filepaths:
            text = self._load_file(filepath)
            chunks = self._chunk_text(text)
            self.documents.extend(chunks)

        self.embeddings = self.model.encode(self.documents, show_progress_bar=True)
        dimension = len(self.embeddings[0])
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)

    def query(self, question: str, top_k: int = 3) -> List[str]:

        query_embedding = self.model.encode([question])
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.documents[i] for i in indices[0]]

    def save(self, path: str):
        faiss.write_index(self.index, path + ".faiss")
        with open(path + "_docs.pkl", "wb") as f:
            pickle.dump(self.documents, f)

    def load(self, path: str):
        self.index = faiss.read_index(path + ".faiss")
        with open(path + "_docs.pkl", "rb") as f:
            self.documents = pickle.load(f)

