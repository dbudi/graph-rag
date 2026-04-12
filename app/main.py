from fastapi import FastAPI
from database import engine
import models
from api.user_router import router
from api.kg_router import router as kg_router

# Create DB tables
models.Base.metadata.create_all(bind=engine)

# Init app
app = FastAPI()

# Register routes
app.include_router(router)
app.include_router(kg_router)