# 1. PDF → 텍스트 추출
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import json
import time
from tqdm import tqdm
from langchain_upstage import ChatUpstage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_upstage import UpstageEmbeddings
from ragas.metrics import context_precision, context_recall
from ragas import evaluate
from datasets import Dataset

UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")


def parse_text_from_pdf(pdf_path):
    # pdf_path = "stay.pdf"  # PDF 경로
    pdf_text = extract_text(pdf_path)
    print(pdf_text[:500])  # 일부 미리보기
    return pdf_text


# 2. 텍스트 → 청크 분할
def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.create_documents([text])


# 3. 모델 및 파서 설정
def add_topic_on_each_chunk():
    llm = ChatUpstage(UPSTAGE_API_KEY)
    output_parser = StrOutputParser()

    # 주제 리스트
    # (사증 매뉴얼)
    # visa_topics = [
    #     "기타", "외교(A-1)", "공무(A-2)", "협정(A-3)", "사증면제(B-1)", "관광통과(B-2)",
    #     "일시취재(C-1)", "단기방문(C-3)", "단기취업(C-4)", "문화예술(D-1)", "유학(D-2)",
    #     "기술연수(D-3)", "일반연수(D-4)", "취재(D-5)", "종교(D-6)", "주재(D-7)",
    #     "기업투자(D-8)", "무역경영(D-9)", "구직(D-10)", "교수(E-1)", "회화지도(E-2)",
    #     "연구(E-3)", "기술지도(E-4)", "전문직업(E-5)", "예술흥행(E-6)", "특정활동(E-7)",
    #     "계절근로(E-8)", "비전문취업(E-9)", "선원취업(E-10)", "방문동거(F-1)", "거주(F-2)",
    #     "동반(F-3)", "재외동포(F-4)", "영주(F-5)", "결혼이민(F-6)", "기타(G-1)",
    #     "관광취업(H-1)", "방문취업(H-2)", "탑티어비자(D-10-T)", "탑티어비자(E-7-T)",
    #     "탑티어비자(F-2-T)", "탑티어비자(F-5-T)"
    # ]

    # 주제 리스트 (체류 매뉴얼)
    visa_topics = [
        "기타", "외교(A-1)", "공무(A-2)", "협정(A-3)", "사증면제(B-1)", "관광통과(B-2)",
        "일시취재(C-1)", "단기방문(C-3)", "단기취업(C-4)", "문화예술(D-1)", "유학(D-2)",
        "기술연수(D-3)", "일반연수(D-4)", "취재(D-5)", "종교(D-6)", "주재(D-7)",
        "기업투자(D-8)", "무역경영(D-9)", "구직(D-10)", "교수(E-1)", "회화지도(E-2)",
        "연구(E-3)", "기술지도(E-4)", "전문직업(E-5)", "예술흥행(E-6)", "특정활동(E-7)",
        "계절근로(E-8)", "비전문취업(E-9)", "선원취업(E-10)", "방문동거(F-1)", "거주(F-2)",
        "동반(F-3)", "재외동포(F-4)", "영주(F-5)", "결혼이민(F-6)", "기타(G-1)",
        "관광취업(H-1)", "방문취업(H-2)", "국내 성장 기반 외국인 청소년 취업.정주 체류제도", "탑티어비자(D-10-T)", "탑티어비자(E-7-T)",
        "탑티어비자(F-2-T)", "탑티어비자(F-5-T)"
    ]

    # 프롬프트 템플릿
    prompt_template = PromptTemplate.from_template("""
    다음 텍스트의 주제를 반드시 아래 리스트 중 하나만 골라주세요.
    절대 리스트에 없는 단어들을 사용해서 주제를 만들지 마세요. 나중에 주제명으로 분류할 예정이라 항상 같은 단어로 구성되어 있어야 합니다.
    
    텍스트:
    \"\"\"{text}\"\"\"
    
    주제 리스트:
    {topics}
    
    주제:
    """)
    chain = prompt_template | llm | output_parser

    # 중간 저장용 파일
    SAVE_PATH = "chunk_with_topic_stay.jsonl"

    # 기존 저장된 청크 불러오기 (있으면 이어서)
    existing_chunks = set()
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                existing_chunks.add(data["content"][:100])  # 중복 체크용 (100자 미리보기)

    # 진행
    with open(SAVE_PATH, "a", encoding="utf-8") as out_file:
        for i, doc in enumerate(tqdm(docs, desc="토픽 분류 중")):
            preview = doc.page_content[:100]

            if preview in existing_chunks:
                continue  # 이미 처리된 청크는 스킵

            # API 호출
            try:
                topic = chain.invoke({
                    "text": doc.page_content,
                    "topics": ", ".join(visa_topics)
                }).strip()
            except Exception as e:
                print(f"[에러] 청크 {i + 1} 처리 중 오류 발생: {e}")
                time.sleep(5)
                continue

            doc.metadata["topic"] = topic

            # 출력
            print(f"\n[청크 {i + 1}]")
            print(f"토픽: {topic}")
            print(f"내용 미리보기:\n{preview}...")
            print("-" * 40)

            # 저장
            save_obj = {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            out_file.write(json.dumps(save_obj, ensure_ascii=False) + "\n")
            out_file.flush()

            # 쿼터 보호
            if i % 5 == 0:
                time.sleep(2)


# 처음 데이터 벡터스토어 저장하는 코드
def save_as_vectorstore():
    # 1. jsonl 파일 불러오기
    jsonl_path = "chunk_with_topic_stay_output.jsonl"
    documents = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            content = item["content"]
            metadata = item["metadata"]
            documents.append(Document(page_content=content, metadata=metadata))

    # 2. 임베딩 모델 설정
    embedding_model = UpstageEmbeddings(api_key=UPSTAGE_API_KEY, model="solar-embedding-1-large")

    # 3. 벡터스토어 생성
    vectorstore = FAISS.from_documents(documents, embedding_model)

    # 4. 벡터스토어 저장 (로컬)
    faiss_save_path = "faiss_store"
    vectorstore.save_local(faiss_save_path)

    print(f"✅ {len(documents)}개의 청크가 FAISS에 저장되었습니다.")



# 첫 번째 이후 데이터 벡터스토어 저장하는 코드
def save_next_one_as_vectorstore():
    # =============================================================================
    # 1. 기존 FAISS 벡터스토어 로드
    # =============================================================================

    print("기존 FAISS 벡터스토어 로드 중...")

    try:
        existing_vectorstore = FAISS.load_local(
            "faiss_store",
            UpstageEmbeddings(api_key=UPSTAGE_API_KEY, model="solar-embedding-1-large"),
            allow_dangerous_deserialization=True
        )
        print(f"✅ 기존 벡터스토어 로드 완료: {len(existing_vectorstore.index_to_docstore_id)}개 문서")
    except Exception as e:
        print(f"❌ 기존 벡터스토어 로드 실패: {e}")
        print("새로운 벡터스토어를 생성합니다.")
        existing_vectorstore = None

    # =============================================================================
    # 2. 새로운 JSONL 파일 로드 → Document 객체로 변환
    # =============================================================================

    def load_jsonl_to_documents(jsonl_path):
        """JSONL 파일을 Document 객체 리스트로 변환"""
        documents = []

        # 파일 존재 확인
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"JSONL 파일을 찾을 수 없습니다: {jsonl_path}")

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())

                    # content 필드 확인
                    if "content" not in item or not item["content"].strip():
                        print(f"⚠️  라인 {line_num}: content가 비어있음 - 건너뜀")
                        continue

                    content = item["content"]
                    metadata = item.get("metadata", {})
                    documents.append(Document(page_content=content, metadata=metadata))

                except json.JSONDecodeError as e:
                    print(f"⚠️  라인 {line_num}: JSON 파싱 에러 - {e}")
                    continue

        return documents

    # 새 문서 로드
    print("\n새로운 JSONL 파일 로드 중...")
    try:
        new_docs = load_jsonl_to_documents("chunk_with_topic_stay_output.jsonl")
        print(f"✅ 새 문서 로드 완료: {len(new_docs)}개 문서")
    except Exception as e:
        print(f"❌ JSONL 로드 실패: {e}")
        raise

    # =============================================================================
    # 3. 원래 방식대로 미리 임베딩 생성 후 벡터스토어에 추가 (API 효율성)
    # =============================================================================

    # 임베딩 모델 초기화
    embedding_model = UpstageEmbeddings(api_key=UPSTAGE_API_KEY, model="solar-embedding-1-large")

    # 배치 설정
    BATCH_SIZE = 50  # API 제한에 따라 조정 (30-100 사이 권장)
    DELAY_SECONDS = 1.0  # 배치 간 대기 시간 (초)

    print(f"\n배치 처리 시작: {len(new_docs)}개 문서를 {BATCH_SIZE}개씩 처리")
    print(f"배치 간 대기 시간: {DELAY_SECONDS}초")

    total_batches = (len(new_docs) + BATCH_SIZE - 1) // BATCH_SIZE
    current_vectorstore = existing_vectorstore

    for i in range(0, len(new_docs), BATCH_SIZE):
        batch_docs = new_docs[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        print(f"\n📦 배치 {batch_num}/{total_batches} 처리 중... ({len(batch_docs)}개 문서)")

        try:
            # 텍스트와 메타데이터 추출
            texts = [doc.page_content for doc in batch_docs]
            metadatas = [doc.metadata for doc in batch_docs]

            # 임베딩 생성 (API 호출 - 한 번만)
            print("   🔄 임베딩 생성 중...")
            embeddings = embedding_model.embed_documents(texts)
            print(f"   📊 생성된 임베딩 정보: {len(embeddings)}개, 차원: {len(embeddings[0]) if embeddings else 0}")

            # 벡터스토어에 추가 (디버깅 포함)
            if current_vectorstore is None:
                # 첫 번째 배치면 새 벡터스토어 생성
                print("   🆕 새 벡터스토어 생성")
                current_vectorstore = FAISS.from_texts(
                    texts=texts,
                    embedding=embedding_model,
                    metadatas=metadatas
                )
            else:
                # 기존 벡터스토어에 추가 - 다양한 방법 시도
                print("   ➕ 기존 벡터스토어에 추가")

                try:
                    # 방법 1: 원래 코드 방식
                    print("     🔧 방법 1: add_embeddings 시도...")
                    current_vectorstore.add_embeddings(
                        texts=texts,
                        text_embeddings=embeddings,
                        metadatas=metadatas
                    )
                    print("     ✅ add_embeddings 성공!")

                except Exception as e1:
                    print(f"     ❌ add_embeddings 실패: {e1}")

                    try:
                        # 방법 2: add_texts 사용
                        print("     🔧 방법 2: add_texts 시도...")
                        current_vectorstore.add_texts(
                            texts=texts,
                            metadatas=metadatas
                        )
                        print("     ✅ add_texts 성공!")

                    except Exception as e2:
                        print(f"     ❌ add_texts도 실패: {e2}")
                        raise e2

            print(f"   ✅ 배치 {batch_num} 완료")

            # API 레이트 리미트 방지를 위한 대기 (마지막 배치 제외)
            if i + BATCH_SIZE < len(new_docs):
                print(f"   ⏳ {DELAY_SECONDS}초 대기 중...")
                time.sleep(DELAY_SECONDS)

        except Exception as e:
            print(f"   ❌ 배치 {batch_num} 처리 중 에러: {e}")
            print("   ⏭️  다음 배치로 계속 진행...")
            continue

    # 결과 확인
    if current_vectorstore:
        total_docs = len(current_vectorstore.index_to_docstore_id)
        print(f"\n🎉 모든 배치 처리 완료!")
        print(f"📊 총 문서 수: {total_docs}개")
    else:
        print("\n❌ 벡터스토어 생성 실패")

    # =============================================================================
    # 4. 벡터스토어 저장
    # =============================================================================

    if current_vectorstore:
        print("\n💾 벡터스토어 저장 중...")
        try:
            current_vectorstore.save_local("faiss_store")
            print("✅ 벡터스토어 저장 완료: faiss_store")
        except Exception as e:
            print(f"❌ 벡터스토어 저장 실패: {e}")
    else:
        print("❌ 저장할 벡터스토어가 없습니다.")

    print("\n🚀 작업 완료!")

    # =============================================================================
    # 5. 결과 확인 (선택사항)
    # =============================================================================

    # 저장된 벡터스토어 확인
    print("\n📋 최종 결과 확인...")
    try:
        final_vectorstore = FAISS.load_local(
            "faiss_store",
            UpstageEmbeddings(api_key=UPSTAGE_API_KEY, model="solar-embedding-1-large"),
            allow_dangerous_deserialization=True
        )
        print(f"✅ 최종 벡터스토어 문서 수: {len(final_vectorstore.index_to_docstore_id)}개")
    except Exception as e:
        print(f"❌ 벡터스토어 확인 실패: {e}")


def load_vectorstore(UPSTAGE_API_KEY):
    # 1. 같은 임베딩 모델 인스턴스를 다시 만들어야 합니다 (이게 중요!)
    embedding_model = UpstageEmbeddings(api_key=UPSTAGE_API_KEY, model="solar-embedding-1-large")

    # 2. FAISS 저장된 벡터스토어 경로 지정 (예: 'faiss_store')
    vectorstore = FAISS.load_local("faiss_store", embedding_model, allow_dangerous_deserialization=True)
    return vectorstore


def get_answer(vectorstore, question, lang):
    llm = ChatUpstage()
    embedding_model = UpstageEmbeddings(model="embedding-query")
    language = lang

    visa_topics = [
        "기타", "외교(A-1)", "공무(A-2)", "협정(A-3)", "사증면제(B-1)", "관광통과(B-2)",
        "일시취재(C-1)", "단기방문(C-3)", "단기취업(C-4)", "문화예술(D-1)", "유학(D-2)",
        "기술연수(D-3)", "일반연수(D-4)", "취재(D-5)", "종교(D-6)", "주재(D-7)",
        "기업투자(D-8)", "무역경영(D-9)", "구직(D-10)", "교수(E-1)", "회화지도(E-2)",
        "연구(E-3)", "기술지도(E-4)", "전문직업(E-5)", "예술흥행(E-6)", "특정활동(E-7)",
        "계절근로(E-8)", "비전문취업(E-9)", "선원취업(E-10)", "방문동거(F-1)", "거주(F-2)",
        "동반(F-3)", "재외동포(F-4)", "영주(F-5)", "결혼이민(F-6)", "기타(G-1)",
        "관광취업(H-1)", "방문취업(H-2)", "탑티어비자(D-10-T)", "탑티어비자(E-7-T)",
        "탑티어비자(F-2-T)", "탑티어비자(F-5-T)"
    ]

    # 주제 리스트 (체류 매뉴얼)
    stay_topics = [
        "기타", "외교(A-1)", "공무(A-2)", "협정(A-3)", "사증면제(B-1)", "관광통과(B-2)",
        "일시취재(C-1)", "단기방문(C-3)", "단기취업(C-4)", "문화예술(D-1)", "유학(D-2)",
        "기술연수(D-3)", "일반연수(D-4)", "취재(D-5)", "종교(D-6)", "주재(D-7)",
        "기업투자(D-8)", "무역경영(D-9)", "구직(D-10)", "교수(E-1)", "회화지도(E-2)",
        "연구(E-3)", "기술지도(E-4)", "전문직업(E-5)", "예술흥행(E-6)", "특정활동(E-7)",
        "계절근로(E-8)", "비전문취업(E-9)", "선원취업(E-10)", "방문동거(F-1)", "거주(F-2)",
        "동반(F-3)", "재외동포(F-4)", "영주(F-5)", "결혼이민(F-6)", "기타(G-1)",
        "관광취업(H-1)", "방문취업(H-2)", "국내 성장 기반 외국인 청소년 취업.정주 체류제도", "탑티어비자(D-10-T)", "탑티어비자(E-7-T)",
        "탑티어비자(F-2-T)", "탑티어비자(F-5-T)"
    ]

    # 2. 프롬프트 정의
    prompt_template = PromptTemplate.from_template("""
        당신은 대한민국 출입국 관련 매뉴얼을 바탕으로 질문에 답변하는 전문가이자 통역가입니다. 제공되는 컨텍스트는 다음 두 종류의 매뉴얼 중 하나 또는 모두일 수 있습니다:

        1. 사증 매뉴얼 (Visa Manual) – 한국 입국 전 비자 발급에 대한 정보  
        2. 체류 매뉴얼 (Stay Manual) – 입국 후 체류 연장, 체류 자격 변경 등에 대한 정보

        질문과 컨텍스트를 잘 읽고, 아래 지침에 따라 사용자가 사용한 언어인 {language}으로 가장 적절한 답변을 해 주세요:
        - 반드시 사용자가 사용한 언어인 {language}를 사용하여 답변하세요!!!
        - 각 비자에는 세부 유형(sub-type)이 있을 수 있습니다 (예: D-8-1, D-8-4).  
          → 각 세부 비자별 요건 및 제출 서류가 다르므로, 반드시 정확한 세부 유형을 구분하여 사용자가 사용한 언어인 {language}으로 답변해 주세요.
        - 각 비자에 대한 제출 서류, 대상자, 자격요건 등은 모든 조건을 만족해야 하는지, 아니면 일부만 만족해도 되는지 명확히 구분하여 답변해 주세요.
        - 점수제에도 여러 종류가 있습니다. 비자에 따라 해당하는 점수제가 달라지니 유의해서 답변해 주세요.
        - 컨텍스트에 답변에 필요한 정보가 부족하다면, 절대로 추측하지 말고 주어진 내용에 기반해서만 사용자가 사용한 언어인 {language}으로 답변해 주세요.

        ---

        질문:
        {question}

        ---

        컨텍스트:
        {context}

        ---

        사증 주제 리스트:
        {visa_topics}

        체류 주제 리스트:
        {stay_topics}

        답변:
        """)

    chain = prompt_template | llm | StrOutputParser()

    # 3. Retriever 정의
    faiss_retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={"k": 20}  # 원본보다 조금 더 넓게 검색
    )

    topic_chain = PromptTemplate.from_template("""
                         다음 질문에 가장 적절한 주제를 아래 리스트에서 하나 선택하세요. 설명이나 덧붙이는 말 없이 주제 리스트에 있는 단어로만 답하세요:

                         질문: {question}
                         주제 리스트: {all_topics}

                         선택한 주제:""") | llm | StrOutputParser()

    # all_topics는 visa_topics + stay_topics
    all_topics = visa_topics + stay_topics

    # 1. 토픽 추론
    try:
        inferred_topic = topic_chain.invoke({
            "question": question,
            "all_topics": ", ".join(visa_topics + stay_topics)
        }).strip()
        print(f"{inferred_topic}")
    except Exception as e:
        print(f"❌ 토픽 추론 실패: '{question}' / 에러: {e}")
        inferred_topic = ""

    # 2. 유사 문서 검색
    all_results = faiss_retriever.invoke(question)

    # 3. 토픽 기반 필터링 (정상 추론된 경우만)
    if inferred_topic:
        filtered_docs = [doc for doc in all_results if inferred_topic in doc.metadata.get("topic", "")]
    else:
        filtered_docs = []

    # 4. 문서 선택 전략
    if not inferred_topic:
        print(f"⚠️ [토픽 추론 실패] '{question}' → 전체 검색 결과 사용")
        context_docs = all_results
    elif not filtered_docs:
        print(f"⚠️ [토픽 불일치] '{question}' → 추론된 토픽: {inferred_topic}, 필터링된 문서 없음 → 전체 검색 결과 사용")
        context_docs = all_results
    else:
        print(f"🔍 추론된 토픽: {inferred_topic}")
        context_docs = filtered_docs

    # 5. 컨텍스트 구성
    context = [doc.page_content for doc in context_docs]
    context_str = "\n\n".join(context)

    # 6. LLM 호출 (필수 프롬프트 변수 포함)
    answer = chain.invoke({
        "context": context_str,
        "question": question,
        "language": language,
        "visa_topics": ", ".join(visa_topics),
        "stay_topics": ", ".join(stay_topics)
    })

    return answer


# def get_answer_and_evaluate(vectorstore):
#     # 1. LLM 및 임베딩 모델 정의
#     llm = ChatUpstage()
#     embedding_model = UpstageEmbeddings(model="embedding-query")
#
#     visa_topics = [
#         "기타", "외교(A-1)", "공무(A-2)", "협정(A-3)", "사증면제(B-1)", "관광통과(B-2)",
#         "일시취재(C-1)", "단기방문(C-3)", "단기취업(C-4)", "문화예술(D-1)", "유학(D-2)",
#         "기술연수(D-3)", "일반연수(D-4)", "취재(D-5)", "종교(D-6)", "주재(D-7)",
#         "기업투자(D-8)", "무역경영(D-9)", "구직(D-10)", "교수(E-1)", "회화지도(E-2)",
#         "연구(E-3)", "기술지도(E-4)", "전문직업(E-5)", "예술흥행(E-6)", "특정활동(E-7)",
#         "계절근로(E-8)", "비전문취업(E-9)", "선원취업(E-10)", "방문동거(F-1)", "거주(F-2)",
#         "동반(F-3)", "재외동포(F-4)", "영주(F-5)", "결혼이민(F-6)", "기타(G-1)",
#         "관광취업(H-1)", "방문취업(H-2)", "탑티어비자(D-10-T)", "탑티어비자(E-7-T)",
#         "탑티어비자(F-2-T)", "탑티어비자(F-5-T)"
#     ]
#
#     # 주제 리스트 (체류 매뉴얼)
#     stay_topics = [
#         "기타", "외교(A-1)", "공무(A-2)", "협정(A-3)", "사증면제(B-1)", "관광통과(B-2)",
#         "일시취재(C-1)", "단기방문(C-3)", "단기취업(C-4)", "문화예술(D-1)", "유학(D-2)",
#         "기술연수(D-3)", "일반연수(D-4)", "취재(D-5)", "종교(D-6)", "주재(D-7)",
#         "기업투자(D-8)", "무역경영(D-9)", "구직(D-10)", "교수(E-1)", "회화지도(E-2)",
#         "연구(E-3)", "기술지도(E-4)", "전문직업(E-5)", "예술흥행(E-6)", "특정활동(E-7)",
#         "계절근로(E-8)", "비전문취업(E-9)", "선원취업(E-10)", "방문동거(F-1)", "거주(F-2)",
#         "동반(F-3)", "재외동포(F-4)", "영주(F-5)", "결혼이민(F-6)", "기타(G-1)",
#         "관광취업(H-1)", "방문취업(H-2)", "국내 성장 기반 외국인 청소년 취업.정주 체류제도", "탑티어비자(D-10-T)", "탑티어비자(E-7-T)",
#         "탑티어비자(F-2-T)", "탑티어비자(F-5-T)"
#     ]
#
#     # 2. 프롬프트 정의
#     prompt_template = PromptTemplate.from_template("""
#     당신은 대한민국 출입국 관련 매뉴얼을 바탕으로 질문에 답변하는 도우미입니다. 제공되는 컨텍스트는 다음 두 종류의 매뉴얼 중 하나 또는 모두일 수 있습니다:
#
#     1. 사증 매뉴얼 (Visa Manual) – 한국 입국 전 비자 발급에 대한 정보
#     2. 체류 매뉴얼 (Stay Manual) – 입국 후 체류 연장, 체류 자격 변경 등에 대한 정보
#
#     질문과 컨텍스트를 잘 읽고, 아래 지침에 따라 가장 적절한 답변을 해 주세요:
#
#     - 각 비자에는 세부 유형(sub-type)이 있을 수 있습니다 (예: D-8-1, D-8-4).
#       → 각 세부 비자별 요건 및 제출 서류가 다르므로, 반드시 정확한 세부 유형을 구분하여 사용자가 사용한 언어로 답변해 주세요.
#     - 각 비자에 대한 제출 서류, 대상자, 자격요건 등은 모든 조건을 만족해야 하는지, 아니면 일부만 만족해도 되는지 명확히 구분하여 답변해 주세요.
#     - 컨텍스트에 답변에 필요한 정보가 부족하다면, 절대로 추측하지 말고 주어진 내용에 기반해서만 사용자가 사용한 언어로 답변해 주세요.
#
#     ---
#
#     질문:
#     {question}
#
#     ---
#
#     컨텍스트:
#     {context}
#
#     ---
#
#     사증 주제 리스트:
#     {visa_topics}
#
#     체류 주제 리스트:
#     {stay_topics}
#
#     답변:
#     """)
#
#     chain = prompt_template | llm | StrOutputParser()
#
#     # 3. Retriever 정의
#     faiss_retriever = vectorstore.as_retriever(
#         search_type='mmr',
#         search_kwargs={"k": 20}  # 원본보다 조금 더 넓게 검색
#     )
#
#     # 4. 질문 & 정답 & 주제 매핑
#     questions = [
#         "재외공관에서 방문취업 사증을 신청하는 대상은 누구인가요?",
#         "D-8-4 비자를 취득하고 싶은데 제출 서류 항목 알려줘!",
#         "D-10-2 비자를 소지하고 있는데 영어키즈카페에서 일해도 돼?",
#         # "D-8-4를 위한 점수제는 뭐야?",
#         "내 한국인 남자친구와 오래 사겼지만 아직 결혼은 안했을 때 결혼비자로 한국에서 살 수 있어?",
#         "D-8-4 Visa에 대한 점수제 표는 어디서 확인할 수 있어?",
#     ]
#
#     ground_truths = [
#         "",
#         # "The D-8-4 Visa is a visa for A technology startup founder who has obtained an associate degree or higher domestically, or a bachelor's degree or higher abroad, or who is recommended by the head of a relevant central administrative agency and holds intellectual property rights or equivalent technological capabilities",
#         "① 사증발급신청서 (별지 제17호 서식), 여권, 표준규격사진 1매, 수수료 ② 법인등기사항전부증명서 및 사업자등록증 사본 ③ 학위증명서 사본 또는 관계 중앙행정기관의 장의 추천서 ④ 점수제 해당 항목( 및 점수) 입증 서류",
#         "",
#         # "The corporate investment visas include the D-8-1 for foreign-invested company representatives under the Foreign Investment Promotion Act; the D-8-2 for founders of venture companies established under the Special Act on the Promotion of Venture Businesses (excluding China and certain countries); the D-8-3 qualification visa for foreigners investing in companies managed by nationals; the D-8-4 for technology startup founders with relevant academic degrees or recommendations who hold intellectual property or equivalent technological capabilities; and the D-8-4S for technology startup founders evaluated by the Startup Korea Special Visa Private Evaluation Committee and recommended by the Minister of SMEs and Startups.",
#         # "점수제 표는 체류자격 매뉴얼에서 확인하실 수 있습니다. 지식재산권의 경우 특허증이나 실용신안등록증, 또는 디자인 등록증 사본을 제출하여 입증할 수 있습니다. 특허 등 출원자는 특허청장 발행 출원사실증명서를 제출해야 합니다. 법무부 장관이 지정한 '글로벌 창업 이민 센터'의 장이 발급한 창업이민종합지원시스템(OASIS) 해당 항목 이수 증서, 입상확인서, 선정공문 등으로 입증할 수 있습니다. 기타 점수제 해당 항목 등의 입증서류들이 요구될 수 있습니다."  # 그대로 복사해서 넣으세요,
#         "",
#         "체류자격 매뉴얼에서 D-8-4 Visa에 대한 점수제 표를 확인할 수 있습니다. 이 표는 해당 비자의 점수제 요건을 상세히 설명하고 있습니다. 또한, 지식재산권의 경우 특허증, 실용신안등록증, 디자인 등록증 등의 사본을 제출하여 입증할 수 있으며, 법무부 장관이 지정한 글로벌 창업 이민 센터의 장이 발급한 증서로도 입증할 수 있습니다."
#     ]
#
#     # 주제 (metadata["topic"]) 매핑
#     # question_topic_map = {
#     #     "What is D-8-4 Visa?": "기업투자(D-8)",
#     #     "D-8-4 비자를 취득하고 싶은데 제출 서류 항목 알려줘!": "기업투자(D-8)",
#     #     "What kind of sub-visas under D-8 Visa?": "기업투자(D-8)",
#     #     "D-8-4를 위한 점수제는 뭐야?": "기업투자(D-8)",
#     #     "D-8-4 Visa에 대한 점수제 표는 어디서 확인할 수 있어?": "기업투자(D-8)",
#     # }
#
#     # 5. QA & 평가 데이터 구성
#     faiss_data = {
#         "question": [],
#         "answer": [],
#         "contexts": [],
#         "ground_truth": [],
#     }
#
#     # # 1. 토픽 추론용 체인 (기존처럼 만들어 둔 chain_topic 활용)
#     # for question, gt in zip(questions, ground_truths):
#     #     topic_chain = PromptTemplate.from_template("""
#     #         다음 질문에 가장 적절한 주제를 아래 리스트에서 하나 선택하세요. 설명이나 덧붙이는 말 없이 주제 리스트에 있는 단어로만 답하세요:
#
#     #         질문: {question}
#     #         주제 리스트: {all_topics}
#
#     #         선택한 주제:""") | llm | StrOutputParser()
#
#     #     # all_topics는 visa_topics + stay_topics
#     #     all_topics = visa_topics + stay_topics
#
#     #     # 2. 질문에 대한 토픽 추론
#     #     inferred_topic = topic_chain.invoke({
#     #         "question": [question for question in questions],
#     #         "all_topics": ", ".join(all_topics)
#     #     }).strip()
#     #     print(f"🔍 추론된 토픽: {inferred_topic}")
#     #         # topic = question_topic_map[question]
#
#     #     all_results = faiss_retriever.invoke(question)
#
#     #     # 필터 조건 완화 (완전 일치 대신 포함 여부 사용)
#     #     filtered_docs = [
#     #         doc for doc in all_results
#     #         if inferred_topic in doc.metadata.get("topic", "")
#     #     ]
#
#     #     if not filtered_docs:
#     #         print(f"⚠️ [토픽 불일치] '{question}' 에 대해 topic 필터링된 문서가 없어 전체 결과 사용")
#     #         context_docs = all_results
#     #     else:
#     #         context_docs = filtered_docs
#
#     #     context = [doc.page_content for doc in context_docs]
#     #     context_str = "\n\n".join(context)
#
#     #     answer = chain.invoke({"context": context_str, "question": question, "visa_topics": ", ".join(visa_topics), "stay_topics": ", ".join(stay_topics)})
#
#     #     faiss_data["question"].append(question)
#     #     faiss_data["answer"].append(answer)
#     #     faiss_data["contexts"].append(context)
#     #     faiss_data["ground_truth"].append(gt)
#     for question, gt in zip(questions, ground_truths):
#
#         topic_chain = PromptTemplate.from_template("""
#                      다음 질문에 가장 적절한 주제를 아래 리스트에서 하나 선택하세요. 설명이나 덧붙이는 말 없이 주제 리스트에 있는 단어로만 답하세요:
#
#                      질문: {question}
#                      주제 리스트: {all_topics}
#
#                      선택한 주제:""") | llm | StrOutputParser()
#
#         # all_topics는 visa_topics + stay_topics
#         all_topics = visa_topics + stay_topics
#
#         # 1. 토픽 추론
#         try:
#             inferred_topic = topic_chain.invoke({
#                 "question": question,
#                 "all_topics": ", ".join(visa_topics + stay_topics)
#             }).strip()
#             print(f"{inferred_topic}")
#         except Exception as e:
#             print(f"❌ 토픽 추론 실패: '{question}' / 에러: {e}")
#             inferred_topic = ""
#
#         # 2. 유사 문서 검색
#         all_results = faiss_retriever.invoke(question)
#
#         # 3. 토픽 기반 필터링 (정상 추론된 경우만)
#         if inferred_topic:
#             filtered_docs = [doc for doc in all_results if inferred_topic in doc.metadata.get("topic", "")]
#         else:
#             filtered_docs = []
#
#         # 4. 문서 선택 전략
#         if not inferred_topic:
#             print(f"⚠️ [토픽 추론 실패] '{question}' → 전체 검색 결과 사용")
#             context_docs = all_results
#         elif not filtered_docs:
#             print(f"⚠️ [토픽 불일치] '{question}' → 추론된 토픽: {inferred_topic}, 필터링된 문서 없음 → 전체 검색 결과 사용")
#             context_docs = all_results
#         else:
#             print(f"🔍 추론된 토픽: {inferred_topic}")
#             context_docs = filtered_docs
#
#         # 5. 컨텍스트 구성
#         context = [doc.page_content for doc in context_docs]
#         context_str = "\n\n".join(context)
#
#         # 6. LLM 호출 (필수 프롬프트 변수 포함)
#         answer = chain.invoke({
#             "context": context_str,
#             "question": question,
#             "visa_topics": ", ".join(visa_topics),
#             "stay_topics": ", ".join(stay_topics)
#         })
#
#         # 7. 저장
#         faiss_data["question"].append(question)
#         faiss_data["answer"].append(answer)
#         faiss_data["contexts"].append(context)
#         faiss_data["ground_truth"].append(gt)
#
#     # 필터 없이 모든 검색 결과 사용
#     # for question, gt in zip(questions, ground_truths):
#     #     all_results = faiss_retriever.invoke(question)
#
#     #
#     #     context = [doc.page_content for doc in all_results]
#     #     context_str = "\n\n".join(context)
#
#     #     answer = chain.invoke({"context": context_str, "question": question})
#
#     #     faiss_data["question"].append(question)
#     #     faiss_data["answer"].append(answer)
#     #     faiss_data["contexts"].append(context)
#     #     faiss_data["ground_truth"].append(gt)
#
#     # 6. RAGAS 평가
#     faiss_dataset = Dataset.from_dict(faiss_data)
#     faiss_score = evaluate(
#         faiss_dataset,
#         metrics=[context_precision, context_recall],
#         llm=llm,
#         embeddings=embedding_model,
#     )
#     print("📊 FAISS Evaluation Score:", faiss_score)
#
#     # 7. 출력
#     for idx, (q, a) in enumerate(zip(faiss_data["question"], faiss_data["answer"])):
#         print(f"{idx + 1} Q: {q}\nA: {a}\n{'-' * 40}")
