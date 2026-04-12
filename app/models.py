from sqlalchemy import Column, Integer, String
from database import Base

# Define User table
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)  # Primary key
    name = Column(String)  # User name
    email = Column(String, unique=True)  # Unique email