from fastapi import Depends

from database import SessionLocal
from repository.user_repository import UserRepository
from services.user_services import UserService

# Provide DB session
def get_db():
    db = SessionLocal()      # create session
    try:
        yield db             # inject session
    finally:
        db.close()           # cleanup

# Provide repository
def get_user_repository():
    return UserRepository()  # stateless, safe to reuse

# Provide service (inject repository)
def get_user_service(
    repo: UserRepository = Depends(get_user_repository)
):
    return UserService(repo)