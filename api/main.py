from fastapi import FastAPI

from api.routes.claims import router as claims_router
from api.routes.policies import router as policies_router
from api.routes.treaties import router as treaties_router
from api.routes.exposures import router as exposures_router
from api.routes.cat_events import router as cat_events_router
from api.routes.fnol import router as fnol_router
from api.routes.predict import router as predict_router
from api.routes.genai import router as genai_router


app = FastAPI(
    title="SpecialtyRe AI Source System API",
    description=(
        "Mock enterprise APIs for specialty P&C and reinsurance claims, "
        "policies, treaties, exposures, and catastrophe events."
    ),
    version="1.0.0",
)


app.include_router(claims_router, prefix="/claims", tags=["Claims"])
app.include_router(policies_router, prefix="/policies", tags=["Policies"])
app.include_router(treaties_router, prefix="/treaties", tags=["Treaties"])
app.include_router(exposures_router, prefix="/exposures", tags=["Exposures"])
app.include_router(cat_events_router, prefix="/cat-events", tags=["CAT Events"])
app.include_router(fnol_router, prefix="/fnol", tags=["FNOL Events"])
app.include_router(predict_router, prefix="/predict", tags=["Prediction"])
app.include_router(genai_router, prefix="/genai", tags=["GenAI"])


@app.get("/")
def root():
    return {
        "message": "SpecialtyRe AI Source System API is running",
        "docs": "/docs",
    }

@app.get("/predict/health")
def health_check():
    return {"status": "healthy"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "specialtyre-source-api",
    }