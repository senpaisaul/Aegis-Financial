import os

class Settings:
    PROJECT_NAME: str = "Integrated AI Platform"
    API_V1_STR: str = "/api/v1"
    
    # Path Configuration
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ASSETS_DIR: str = os.path.join(os.path.dirname(BASE_DIR), "assets")
    MODELS_DIR_CREDIT: str = os.path.join(ASSETS_DIR, "models", "credit_risk")

settings = Settings()
