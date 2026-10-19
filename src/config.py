"""Configuração validada; segredos nunca são incluídos em mensagens de erro."""
import hashlib
import math
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
    ingest_batch_size: int = 64
    ingest_batch_delay_seconds: float = 0

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
        try:
            batch_size = int(os.getenv('INGEST_BATCH_SIZE', '16'))
            batch_delay = float(os.getenv('INGEST_BATCH_DELAY_SECONDS', '30'))
        except ValueError as exc:
            raise ValueError('INGEST_BATCH_SIZE e INGEST_BATCH_DELAY_SECONDS precisam ser números válidos.') from exc
        if not 1 <= batch_size <= 64:
            raise ValueError('INGEST_BATCH_SIZE precisa estar entre 1 e 64.')
        if not math.isfinite(batch_delay) or not 0 <= batch_delay <= 300:
            raise ValueError('INGEST_BATCH_DELAY_SECONDS precisa estar entre 0 e 300.')
        suffix = hashlib.sha256(model.encode()).hexdigest()[:12]
        collection = os.getenv('PG_VECTOR_COLLECTION_NAME', 'prisma_pdf') + '_' + suffix
        return cls(url, collection, pdf, model, os.environ['GOOGLE_CHAT_MODEL'], batch_size, batch_delay)

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
