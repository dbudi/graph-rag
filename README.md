# Graph RAG

Technoloy stack:
- python
- fastAPI framework
- langchain
- LiteLLM
- Neo4j graph DB
- PostgreSQL
- SQLAlchemy
- Repository + Service pattern
- .env configuration

---

![Knowledge Graph Builder Workflow](https://www.plantuml.com/plantuml/png/ZLB1Rjim3BthAnvwxsqd7ugcMn440OgWGR0pQp6n6baIA1f9_twGQZTnWmsxKUHxZ-IZ7hEIwD1xh4CyURO7nb8eTNWQdBSi-1tbkFVnP4m-kW29cMkKIAZ7LZyy8wkujgWOTUhvFxIerIZOPHQJIDIrBb5Gt2qsgpjGncApcXT-nNReEIZuh4AjkXWgSCe_VwNdlBORmj65GmN_71zpo7F_njPEcA_N7BHUVk-yNlwjvspfYgsPCJ77ld3yyLunDJAcU8BxX-90WrtoaIpikAHPR5QbThjqWp66ybcnbzXVMBVm8ZAPqM2Rl1k9BcX4zIoR2L30ryjUUd5GfuupA5W8UohbtBcKZOynOHy5J6ttgMW0NEmjgjUsD5XoO25by8J9LU52lfxYbby37Ahv9wVTT32s_ngqw5xh5DrwYWPRZSAZe0kVMLXIUgFCXeCrS-9qjLVMGS98dD_k_g5--kPR4S8vnP3Pc4SETkZz7m00 "Knowledge Graph Builder Workflow")

## Setup

### 1. Install dependencies
pip install -r requirements.txt

### 2. Create .env
DATABASE_URL=postgresql://user:password@localhost:5432/mydb

### 3. Run app
uvicorn app.main:app --reload

---

## API Docs
http://127.0.0.1:8000/docs

---

## Project Structure

- api/ → route layer
- services/ → business logic
- repository/ → database queries
- models.py → ORM models
- schemas.py → request/response models

---

## Example Endpoints

POST /users  
GET /users  
GET /users/{id}  
DELETE /users/{id}

---

## Notes
- Service layer handles validation
- Repository layer handles DB access
- Easy to extend with auth, logging, caching