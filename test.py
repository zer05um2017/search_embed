
import os
import asyncio
import numpy as np
from typing import cast
from fastapi import APIRouter, Depends
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from data.db_manager_gpu import VectorDBManager
# from api.retriever import ContentKeysEnum
from constants import JsonKeysEnum
from constants import FaissIndexMapper


# llm = ChatOllama(model="Llama-3-Open-Ko-8B-Q8_0:latest")
# output_parser = StrOutputParser()
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "너는 도움을 주는 전문 보조역할을하는 챗봇이야. 질문에 반드시 한국어로 답변해."),
#     ("user", "{input}")
# ])

# chain = prompt | llm | output_parser
# for token in chain.stream(
#     {"input": "빛은 파동이야 입자야?"}
# ):
#     print(token, end="")

class KeyParams():
    db_name: str
    keywords: str
    # interestTopics: str
    # address: str
    # addressDetail: str
    max_length: int = 20


async def get_vector_db_manager():
    if not hasattr(get_vector_db_manager, 'instance'):
        get_vector_db_manager.instance = VectorDBManager()
    return get_vector_db_manager.instance


async def retrieve(params:KeyParams):
    try:
        retriever = await get_vector_db_manager()
        product_list = {}
        # proj_contents = {}
        # cases_contents = {}
        # studio_contents = {}
        # other_contents = {}
        # retriever.index_files[name]

        # query_kwrds = {
        #     "suppbiz": f'{params.industry} {params.skillKeywords} {params.interestTopics} {params.address}',
        #     "succ_case": f'{params.industry} {params.interestTopics}',
        #     "studio": f'{params.address} {params.addressDetail}'
        # }

        query_kwrds = {
            "products": f'{params.keywords}'
        }

        # tasks = [
        #     retriever.retrieve(db_name=params.dbname, query=query_kwrds.get("products"), k=params.max_length)
        #     for name, index_file in retriever.faiss_products.items()
        # ]
        #
        # results = await asyncio.gather(*tasks)
        #
        # for name, result in zip(retriever.faiss_products.keys(), results):
        #     product_list[name] = result

        # 특정 db_name을 넘겨받아 처리하는 비동기 코드
        tasks = [
            retriever.retrieve(db_name=params.db_name, query=query_kwrds.get("products"), k=params.max_length)
        ]

        # 결과를 비동기적으로 받아옴
        results = await asyncio.gather(*tasks)

        # 조회된 데이터 저장
        product_list[params.db_name] = results[0]

        print("creating json")

        print({'results': product_list.get(params.db_name)})
        return {'results': product_list.get(params.db_name)}
    except Exception as e:
        print(e)
        return {'results': e}


async def embedding(csv_file, db_name, columns_to_clean, chunk_size=500):
    try:
        retriever = await get_vector_db_manager()
        tasks = [
            retriever.save_embeddings(csv_file=csv_file, db_name=db_name, chunk_size=chunk_size,
                                      columns_to_clean=columns_to_clean)
        ]

        await asyncio.gather(*tasks)

        retriever.reload_index(db_name=db_name)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    try:
        # params = KeyParams()
        # params.industry = "도소매"
        # params.skillKeywords = "경영 마케팅 차별화 재무 사업준비"
        # params.interestTopics = "온라인진출 판로개척 홍보마케팅 해외판매 라이브커머스"
        # params.address = "서울시"
        # params.addressDetail = "양천구"
        # asyncio.run(retrieve(params=params))

        params = KeyParams()
        params.db_name = "wmall"
        params.keywords = "여름"
        params.max_length = 20
        asyncio.run(retrieve(params=params))

        # asyncio.run(embedding(csv_file="product_data", db_name="products", columns_to_clean=["name", "contents"], chunk_size=1200))


    except Exception as e:
        print(f"An error occurred: {e}")
