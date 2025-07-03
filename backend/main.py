from fastapi import FastAPI
from utils import load_vectorstore, get_answer
import os
from dotenv import load_dotenv
from fastapi import Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:3000",  # Next.js 개발서버 주소
    "https://your-production-domain.com",  # 배포 후 프론트 도메인 추가
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # 허용할 출처 리스트
    allow_credentials=True,           # 쿠키 같은 인증정보 허용 여부
    allow_methods=["*"],              # 허용할 HTTP 메서드
    allow_headers=["*"],              # 허용할 헤더
)

# .env 파일 로드
load_dotenv()

UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")

vectorstore = load_vectorstore(UPSTAGE_API_KEY)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/chat-request")
async def chat_request(req: str = Query(..., description="질문 내용"), lang: str = Query(..., description="언어 코드")):
    print(f"req: {req}, lang: {lang}")
    ans = get_answer(vectorstore, req, lang)
    return {"question": req,
            "lang": lang,
            "answer": ans}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)