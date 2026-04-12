from sqlalchemy.orm import Session
from models import User

# Handle direct DB operations
class UserRepository:

    def create(self, db: Session, name: str, email: str):
        user = User(name=name, email=email)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    def get_all(self, db: Session):
        return db.query(User).all()

    def get_by_id(self, db: Session, user_id: int):
        return db.query(User).filter(User.id == user_id).first()

    def delete(self, db: Session, user_id: int):
        user = self.get_by_id(db, user_id)
        if user:
            db.delete(user)
            db.commit()
        return user