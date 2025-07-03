from fastapi import FastAPI
from utils import load_vectorstore, get_answer
import os
from dotenv import load_dotenv
from fastapi import Query
from fastapi.middleware.cors import CORSMiddleware

from langchain.memory import ConversationBufferMemory

app = FastAPI()

origins = [
    "http://localhost:3000",  # Next.js 개발서버 주소
    "https://navisation.vercel.app",  # 배포 후 프론트 도메인 추가
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

session_memories = {}


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/chat-request")
async def chat_request(req: str = Query(..., description="질문 내용"),
                       lang: str = Query(..., description="언어 코드"),
                       session_id: str = Query(..., description="세션 ID")
                       ):
    print(f"[session: {session_id}] req: {req}, lang: {lang}")
    
    print(session_memories)
    # 세션별 memory 객체 재사용 또는 생성
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory(return_messages=False)

    memory = session_memories[session_id]
    
    ans = get_answer(vectorstore, req, lang, memory)
    return {"question": req,
            "lang": lang,
            "session_id": session_id,
            "answer": ans}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)