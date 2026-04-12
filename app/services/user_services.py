from sqlalchemy.orm import Session
from repository.user_repository import UserRepository

# Business logic layer
class UserService:

    def __init__(self, repo: UserRepository):
        self.repo = repo  # injected dependency

    def create_user(self, db: Session, name: str, email: str):
        if "@" not in email:  # simple validation
            raise ValueError("Invalid email")
        return self.repo.create(db, name, email)

    def get_users(self, db: Session):
        return self.repo.get_all(db)

    def get_user(self, db: Session, user_id: int):
        return self.repo.get_by_id(db, user_id)

    def delete_user(self, db: Session, user_id: int):
        return self.repo.delete(db, user_id)