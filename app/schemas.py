from pydantic import BaseModel

class UserCreate(BaseModel):
    name: str
    email: str

class UserResponse(UserCreate):
    id: int

    # Pydantic v2 config
    class Config:
        from_attributes = True  # replaces orm_mode