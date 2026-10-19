"""Configuração validada; segredos nunca são incluídos em mensagens de erro."""
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Settings:
    database_url: str
    collection: str
    pdf_path: Path
    embedding_model: str
    chat_model: str

    @classmethod
    def load(cls):
        load_dotenv(ROOT / '.env')
        required = ['GOOGLE_API_KEY', 'DATABASE_URL', 'GOOGLE_EMBEDDING_MODEL', 'GOOGLE_CHAT_MODEL']
        missing = [name for name in required if not os.getenv(name, '').strip()]
        if missing:
            raise ValueError('Configure no .env: ' + ', '.join(missing))
        url = os.environ['DATABASE_URL']
        if not url.startswith('postgresql+psycopg://'):
            raise ValueError('DATABASE_URL precisa usar o driver postgresql+psycopg://')
        pdf = Path(os.getenv('PDF_PATH', 'document.pdf'))
        if not pdf.is_absolute():
            pdf = ROOT / pdf
        if not pdf.is_file():
            raise ValueError('PDF_PATH não aponta para um arquivo existente.')
        model = os.environ['GOOGLE_EMBEDDING_MODEL']
        suffix = hashlib.sha256(model.encode()).hexdigest()[:12]
        collection = os.getenv('PG_VECTOR_COLLECTION_NAME', 'prisma_pdf') + '_' + suffix
        return cls(url, collection, pdf, model, os.environ['GOOGLE_CHAT_MODEL'])

    def fingerprint(self):
        with self.pdf_path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()


def vector_store(settings):
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_postgres import PGVector
    return PGVector(
        embeddings=GoogleGenerativeAIEmbeddings(model=settings.embedding_model),
        connection=settings.database_url,
        collection_name=settings.collection,
        use_jsonb=True,
    )
