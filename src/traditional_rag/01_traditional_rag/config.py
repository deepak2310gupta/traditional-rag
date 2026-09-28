
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DIRECTORY_PATH = BASE_DIR / "data"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 3