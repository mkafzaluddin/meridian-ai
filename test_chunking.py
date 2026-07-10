# test_chunking.py
import os
from dotenv import load_dotenv

from backend import get_embeddings

load_dotenv()

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

reader = PdfReader("sample.pdf")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

# Step 1 & 2 - Extract and chunk PAGE BY PAGE, so each chunk remembers its page
documents = []
for page_num, page in enumerate(reader.pages, start=1):
    page_text = page.extract_text()
    page_chunks = splitter.split_text(page_text)
    for chunk in page_chunks:
        documents.append(
            Document(
                page_content=chunk, metadata={"page": page_num, "source": "sample.pdf"}
            )
        )

print(f"Built {len(documents)} chunks across {len(reader.pages)} pages")

# Step 3 - Embed and store (now storing Documents, not raw strings)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = InMemoryVectorStore(get_embeddings())
vector_store.add_documents(documents)
print("Stored all chunks as vectors")

# Step 4 - Search, and show exactly where each result came from
query = "What foods should I eat more of?"
results = vector_store.similarity_search(query, k=3)

print(f"\nTop {len(results)} chunks for: '{query}'\n")
for i, r in enumerate(results, 1):
    print(
        f"--- Match {i} (from page {r.metadata['page']} of {r.metadata['source']}) ---"
    )
    print(r.page_content[:300])
    print()
