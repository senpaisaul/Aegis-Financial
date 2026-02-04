from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import credit, stock, option
from app.core.config import settings

app = FastAPI(title=settings.PROJECT_NAME)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(credit.router, prefix=settings.API_V1_STR + "/credit", tags=["Credit Risk"])
app.include_router(stock.router, prefix=settings.API_V1_STR + "/stock", tags=["Stock Prediction"])
app.include_router(option.router, prefix=settings.API_V1_STR + "/option", tags=["Option Pricing"])

@app.get("/health")
def health_check():
    return {"status": "ok"}
