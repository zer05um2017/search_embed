import asyncio
# import csv
import json
import logging
import os
import re
import pandas as pd
import io
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Type, Optional, Any, AsyncIterable

import aiofiles
import torch
from aiocsv import AsyncDictReader
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from tqdm import tqdm

from constants import FaissIndexMapper
import gc


load_dotenv()
tqdm.monitor_interval = 0

# 조건부 import
if torch.cuda.is_available():
    try:
        import faiss
    except ImportError:
        faiss = None
        logging.warning("CUDA is available but faiss-gpu is not installed. GPU acceleration will be disabled.")
else:
    faiss = None


class VectorDBManager:
    def __init__(self, folder_path='./data/db', device='cpu', encoding="utf-8"):
        print('VectorDBManager is being initialized')
        self.documents = None
        self.vector_db = None
        self.encoding = encoding
        self.folder_path = folder_path
        self.csv_folder_path = "./data/csv_files/"
        self.indexes: Dict[str, FAISS] = {}

        self.use_gpu = torch.cuda.is_available() and os.getenv('USE_GPU', 'false').lower() == 'true'
        self.device = 'cuda' if self.use_gpu else 'cpu'

        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': device},  # 모델이 CPU에서 실행 되도록 설정. GPU를 사용할 수 있는 환경 이라면 'cuda'로 설정할 수도 있음
            encode_kwargs={'normalize_embeddings': True},  # 임베딩 정규화. 모든 벡터가 같은 범위의 값을 갖도록 함. 유사도 계산 시 일관성을 높여줌
        )

        self.gpu_resources = None

        if self.use_gpu:
            try:
                self.embeddings.client.to('cuda')
                ngpus = faiss.get_num_gpus()
                self.gpu_resources = [faiss.StandardGpuResources() for i in range(ngpus)]
                logging.info(f"Using {ngpus} GPU(s) for FAISS")
                print("Using GPU for embeddings")
            except ImportError:
                logging.error("faiss-gpu is not installed. Falling back to CPU.")
                self.use_gpu = False
                self.device = 'cpu'
        else:
            print("GPU not available, using CPU for embeddings")

        logging.info(f"Using {self.device.upper()} for computations. GPU acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")

        self._init_faiss_mapper()
        self._load_faiss_files(self.faiss_files)

        self.executor = ThreadPoolExecutor(max_workers=5)

    def _init_faiss_mapper(self):
        self.faiss_products = {
            FaissIndexMapper.INDEX_KEY.KEY_SKMALL.value: FaissIndexMapper.INDEX_VALUE.VALUE_SKMALL.value,
        }
        self.faiss_nsmall = {
            FaissIndexMapper.INDEX_KEY.KEY_NSMALL.value: FaissIndexMapper.INDEX_VALUE.VALUE_NSMALL.value,
        }
        self.faiss_wmall = {
            FaissIndexMapper.INDEX_KEY.KEY_WMALL.value: FaissIndexMapper.INDEX_VALUE.VALUE_WMALL.value
        }
        self.faiss_ssgdfs = {
            FaissIndexMapper.INDEX_KEY.KEY_SSGDFS.value: FaissIndexMapper.INDEX_VALUE.VALUE_SSGDFS.value
        }

        self.faiss_files = {
            **self.faiss_products,
            **self.faiss_nsmall,
            **self.faiss_wmall,
            **self.faiss_ssgdfs,
        }

    def _load_faiss_files(self, indices: Dict):
        print('loading vector databases')
        # for faiss_files in indices:
        for name, faiss_file in indices.items():
            try:
                file_path = f'{self.folder_path}/{faiss_file}.faiss'
                if not os.path.exists(file_path):
                    print(f"File {file_path} does not exist, skipping.")
                    continue
                print(f'{name}/{faiss_file} database is being loaded')

                self.indexes[name] = FAISS.load_local(
                    self.folder_path,
                    self.embeddings,
                    index_name=faiss_file,
                    allow_dangerous_deserialization=True)

            except Exception as e:
                # print(f"Error loading index {name}/{faiss_file}: {str(e)}")
                print(e)
                continue

    def reload_index(self,
                     db_name=Type[str]):

        if not db_name:
            raise ValueError(
                "No folder_path to retrieve and faiss_name. Please provide valid folder and index path during to load.")
        try:
            # Clear or reset existing index if it exists
            if db_name in self.indexes and hasattr(self.indexes[db_name], 'reset'):
                self.indexes[db_name].reset()
            elif db_name in self.indexes and hasattr(self.indexes[db_name], 'index') and hasattr(
                    self.indexes[db_name].index, 'reset'):
                self.indexes[db_name].index.reset()

            # FAISS index 로드
            index_name = self.faiss_files[db_name]
            self.indexes[db_name] = FAISS.load_local(self.folder_path,
                                                     self.embeddings,
                                                     index_name=index_name,
                                                     allow_dangerous_deserialization=True)

            print(f"ReLoaded FAISS index from {index_name} of {db_name}")
        except Exception as e:
            print(e)
            raise

    @staticmethod
    async def _clean_text(text):
        # 공백 여러 개를 하나로 줄임
        text = re.sub(r'\s+', ' ', text)
        # 연속된 \n을 하나로 줄임
        text = re.sub(r'\n+', '\n', text)
        # 한글, 영문자, 숫자, 공백 외 제거
        return re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)

    async def _process_csv(self, csv_file: str, columns_to_clean: Optional[List[str]]) -> List[Dict[str, Any]]:
        path = os.path.join(self.csv_folder_path, f"{csv_file}.csv")
        try:
            # 파일을 비동기로 읽습니다.
            async with aiofiles.open(path, mode='r', encoding='utf-8-sig') as csvfile:
                file_content = await csvfile.read()
            # pandas로 CSV 파싱
            df = pd.read_csv(io.StringIO(file_content))
            processed_data = df.to_dict('records')
            # 필요한 컬럼 전처리
            for row in processed_data:
                if columns_to_clean:
                    for column in columns_to_clean:
                        if column in row and pd.notnull(row[column]):
                            row[column] = await self._clean_text(str(row[column]))
            return processed_data
        except Exception as e:
            logging.error(f"Error processing CSV file {csv_file}: {str(e)}")
            raise

    async def _prepare_documents(self, chunk_size: int, processed_data: List[Dict[str, Any]]) -> List[Document]:
        temp_json_file = 'temp_processed_data.json'
        async with aiofiles.open(temp_json_file, 'w', encoding='utf-8') as jsonfile:
            for item in processed_data:
                await jsonfile.write(json.dumps(item, ensure_ascii=False) + '\n\n\n')
        print("json file was saved.")

        loop = asyncio.get_event_loop()
        loader = TextLoader(file_path=temp_json_file, encoding=self.encoding)
        raw_documents = await loop.run_in_executor(self.executor, loader.load)

        text_splitter = CharacterTextSplitter(
            separator="\n\n\n",
            chunk_size=chunk_size,  # 원하는 최대 chunk 크기
            chunk_overlap=0  # chunk 간 중복을 추가하여 데이터가 손실되지 않도록
        )

        self.documents = await loop.run_in_executor(self.executor, text_splitter.split_documents, raw_documents)

        valid_documents = []
        for i, doc in enumerate(self.documents):
            if hasattr(doc, 'page_content') and doc.page_content.strip() != "":
                valid_documents.append(doc.page_content)
            else:
                print(f"Skipping invalid or empty document at index {i}: {repr(doc)}")

        return [Document(page_content=text) for text in valid_documents]

    async def save_embeddings(self,
                              csv_file=Type[str],
                              db_name=Type[str],
                              chunk_size=0,
                              columns_to_clean=Type[List[str]]):
        try:
            if not csv_file and not db_name:
                raise ValueError(
                    "No documents to embed. Please provide valid file path and name during initialization.")

            # 코루틴을 await하여 리스트를 받아옵니다.
            processed_data = await self._process_csv(csv_file, columns_to_clean)

            # 문서 준비
            document_objects = await self._prepare_documents(chunk_size, processed_data)

            loop = asyncio.get_event_loop()

            # 배치 크기 설정
            batch_size = 20  # 메모리 상황에 따라 조절 가능
            num_documents = len(document_objects)

            # 빈 벡터 스토어 초기화
            vector_store = None

            for i in range(0, num_documents, batch_size):
                batch_documents = document_objects[i:i+batch_size]
                # 각 배치에 대해 벡터 스토어 생성 또는 기존 벡터 스토어에 추가
                if vector_store is None:
                    vector_store = await loop.run_in_executor(self.executor, FAISS.from_documents, batch_documents, self.embeddings)
                else:
                    # 기존 벡터 스토어에 배치 추가
                    await loop.run_in_executor(self.executor, vector_store.add_documents, batch_documents)

                del batch_documents
                #del vector_store
                gc.collect()
                #torch.cuda.empty_cache()

            if self.use_gpu:
                # GPU 인덱스를 CPU 인덱스로 변환
                cpu_index = faiss.index_gpu_to_cpu(vector_store.index)
                vector_store.index = cpu_index

            index_name = self.faiss_files.get(db_name)
            if not index_name:
                raise ValueError(f"No index name found for database {db_name}")

            await loop.run_in_executor(self.executor, vector_store.save_local, self.folder_path, index_name)

            logging.info(f'Embedding is saved using {self.device.upper()} for {db_name}')
        except ValueError as ve:
            logging.error(f"Value error occurred: {str(ve)}")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred while saving embeddings for {db_name}: {str(e)}")
            logging.exception("Exception details:")
            raise



    async def retrieve(self,
                       db_name: str,
                       query: str,
                       k=3) -> list:

        if not db_name and not query:
            raise ValueError("No database name and query to retrieve. Please provide a valid database name and query.")

        try:
            index_name = self.faiss_files.get(db_name)
            if not index_name:
                raise ValueError(f"No index name found for database {db_name}")

            faiss_index = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: FAISS.load_local(
                    self.folder_path,
                    self.embeddings,
                    index_name,
                    allow_dangerous_deserialization=True
                )
            )

            if self.use_gpu:
                cpu_index = faiss_index.index
                gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
                faiss_index.index = gpu_index

            docs = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: faiss_index.similarity_search(query, k)
            )

            if self.use_gpu:
                # GPU 메모리 해제
                del faiss_index.index
                torch.cuda.empty_cache()

            return [json.loads(item.page_content) for item in docs]
        except ValueError as ve:
            logging.error(f"Value error during retrieval from {db_name}: {str(ve)}")
            return []
        except Exception as e:
            logging.error(f"An unexpected error occurred during retrieval from {db_name}: {str(e)}")
            return []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """
        Close and clean up resources.
        """
        # GPU 리소스 해제
        if self.gpu_resources:
            del self.gpu_resources

        # FAISS 인덱스 정리
        for index in self.indexes.values():
            if hasattr(index, 'index') and hasattr(index.index, 'reset'):
                index.index.reset()
        self.indexes.clear()

        # 임베딩 모델 정리
        if hasattr(self.embeddings, 'client'):
            del self.embeddings.client

        # CUDA 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ThreadPoolExecutor 종료
        self.executor.shutdown(wait=True)
