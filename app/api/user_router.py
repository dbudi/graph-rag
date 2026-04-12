from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from schemas import UserCreate
from services.user_services import UserService
from dependencies import get_db, get_user_service

router = APIRouter()

@router.post("/users")
def create_user(
    user: UserCreate,
    db: Session = Depends(get_db),
    service: UserService = Depends(get_user_service),
):
    return service.create_user(db, user.name, user.email)

@router.get("/users")
def get_users(
    db: Session = Depends(get_db),
    service: UserService = Depends(get_user_service),
):
    return service.get_users(db)

@router.get("/users/{user_id}")
def get_user(
    user_id: int,
    db: Session = Depends(get_db),
    service: UserService = Depends(get_user_service),
):
    return service.get_user(db, user_id)

@router.delete("/users/{user_id}")
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    service: UserService = Depends(get_user_service),
):
    return service.delete_user(db, user_id)