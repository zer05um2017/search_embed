from fastapi import FastAPI
from api import retriever
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(retriever.router)

@app.get('/')
def root():
    return {"It's a diagnosis system"}


if __name__ == '__main__':
    # embed = VectorDBManager()
    print('hello world')
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000)
