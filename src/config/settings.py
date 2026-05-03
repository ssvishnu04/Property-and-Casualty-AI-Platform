import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    APP_ENV = os.getenv("APP_ENV", "local")
    API_HOST = os.getenv("API_HOST", "127.0.0.1")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    DATA_PATH = os.getenv("DATA_PATH", "data/local_landing")
    MASK_PII = os.getenv("MASK_PII", "true").lower() == "true"

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")
    AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "landing")


settings = Settings()