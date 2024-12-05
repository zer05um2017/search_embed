import os
import re
import csv
import json
from typing import cast, Dict, List, Type
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from constants import FaissIndexMapper


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def clean_text(text):
    return re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)


class VectorDBManager:
    def __init__(self, folder_path='./data/db', device='cpu', encoding="utf-8"):
        print('VectorDBManager is being initialized')
        self.documents = None
        self.vector_db = None
        self.encoding = encoding
        self.folder_path = folder_path
        self.csv_folder_path = "./data/csv_files/"
        self.indexes: Dict[str, FAISS] = {}

        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': device},  # 모델이 CPU에서 실행 되도록 설정. GPU를 사용할 수 있는 환경 이라면 'cuda'로 설정할 수도 있음
            encode_kwargs={'normalize_embeddings': True},  # 임베딩 정규화. 모든 벡터가 같은 범위의 값을 갖도록 함. 유사도 계산 시 일관성을 높여줌
        )

        self._init_faiss_mapper()
        self._load_faiss_files(self.faiss_files)

    def _init_faiss_mapper(self):
        self.faiss_prj_files = {
            FaissIndexMapper.INDEX_KEY.KEY_SUPP_PROJ.value: FaissIndexMapper.INDEX_VALUE.VALUE_SUPP_PROJ.value
        }

        self.faiss_succcase_files = {
            FaissIndexMapper.INDEX_KEY.KEY_SUCC_CASE.value: FaissIndexMapper.INDEX_VALUE.VALUE_SUCC_CASE.value
        }

        self.faiss_other_files = {
            FaissIndexMapper.INDEX_KEY.KEY_FANROTV_VOD.value: FaissIndexMapper.INDEX_VALUE.VALUE_FANROTV_VOD.value,
            FaissIndexMapper.INDEX_KEY.KEY_NEWS_INFO.value: FaissIndexMapper.INDEX_VALUE.VALUE_NEWS_INFO.value,
            FaissIndexMapper.INDEX_KEY.KEY_E_LEARN_A.value: FaissIndexMapper.INDEX_VALUE.VALUE_E_LEARN_A.value
            # FaissIndexMapper.INDEX_KEY.KEY_E_LEARN1.value: FaissIndexMapper.INDEX_VALUE.VALUE_E_LEARN1.value,
            # FaissIndexMapper.INDEX_KEY.KEY_E_LEARN2.value: FaissIndexMapper.INDEX_VALUE.VALUE_E_LEARN2.value,
            # FaissIndexMapper.INDEX_KEY.KEY_E_LEARN3.value: FaissIndexMapper.INDEX_VALUE.VALUE_E_LEARN3.value
        }

        self.faiss_studio_files = {
            FaissIndexMapper.INDEX_KEY.KEY_STUDIO_INFO.value: FaissIndexMapper.INDEX_VALUE.VALUE_STUDIO_INFO.value
        }

        self.faiss_files = {
            **self.faiss_prj_files,
            **self.faiss_other_files,
            **self.faiss_studio_files,
            **self.faiss_succcase_files
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
            # FAISS index 로드
            index_name = self.faiss_files[db_name]
            self.indexes[db_name] = FAISS.load_local(self.folder_path,
                                                     self.embeddings,
                                                     index_name=index_name,
                                                     allow_dangerous_deserialization=True)

            print(f"ReLoaded FAISS index from {index_name} of {db_name}")
        except Exception as e:
            print(e)

    async def save_embeddings(self,
                              csv_file=str,
                              db_name=Type[str],
                              chunk_size=500,
                              columns_to_clean=Type[List[str]]):
        try:
            if not csv_file and not db_name:
                raise ValueError("No documents to embed. Please provide valid file path and name during initialization.")

            # CSV 파일 처리
            processed_data = []
            path = f'{self.csv_folder_path}{csv_file}.csv'

            with open(path, 'r', encoding='utf-8-sig') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # BOM 제거
                    cleaned_row = {key.lstrip('\ufeff'): value for key, value in row.items()}
                    if columns_to_clean:
                        for column in columns_to_clean:
                            if column in cleaned_row:
                                cleaned_row[column] = clean_text(cleaned_row[column])
                    processed_data.append(cleaned_row)

            print("Raw file was cleaned.")

            # 처리된 데이터를 임시 JSON 파일로 저장
            temp_json_file = 'temp_processed_data.json'
            with open(temp_json_file, 'w', encoding='utf-8') as jsonfile:
                for item in processed_data:
                    json.dump(item, jsonfile, ensure_ascii=False)
                    jsonfile.write('\n\n')
            print("json file was saved.")
            # 임시 JSON 파일을 Text Loader 로드
            loader = TextLoader(
                file_path=temp_json_file,
                encoding=self.encoding)

            raw_documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=0,
                separators=["\n\n"])

            self.documents = text_splitter.split_documents(raw_documents)

            valid_documents = []

            for i, doc in enumerate(self.documents):
                if hasattr(doc, 'page_content') and doc.page_content.strip() != "":
                    valid_documents.append(doc.page_content)
                else:
                    print(f"Skipping invalid or empty document at index {i}: {repr(doc)}")

            document_objects = [Document(page_content=text) for text in valid_documents]

            db = FAISS.from_documents(
                document_objects,
                self.embeddings)
            index_name = self.faiss_files[db_name]
            db.save_local(self.folder_path, index_name)

            print('Embedding is saved')
        except Exception as e:
            print(e)

    async def retrieve(self,
                       db_name: str,
                       query: str,
                       k=3) -> list:

        if not db_name and not query:
            raise ValueError("No database name and query to retrieve. Please provide a valid database name and query.")

        try:
            faiss_index = cast(FAISS, self.indexes[db_name])
            docs = faiss_index.similarity_search(query=query, k=k)
            contents = [json.loads(item.page_content) for item in docs]
        except Exception as e:
            print(e)
            contents = []
        # self.indexes[db_name]
        # distances, indices = self.index.search(query_embedding_np, k)
        return contents

    # def save_embeddings(self, file_path=None, index_name=None, chunk_size=500):
    #     if not file_path and not index_name:
    #         raise ValueError("No documents to embed. Please provide valid file path and name during initialization.")
    #
    #     loader = TextLoader(file_path=file_path, encoding=self.encoding)
    #     raw_documents = loader.load()
    #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0, separators=["\n\n"])
    #     self.documents = text_splitter.split_documents(raw_documents)
    #
    #     valid_documents = []
    #
    #     for i, doc in enumerate(self.documents):
    #         # doc이 문자열이 아닌 Document 객체일 경우
    #         if hasattr(doc, 'page_content') and doc.page_content.strip() != "":
    #             valid_documents.append(doc.page_content)
    #         else:
    #             print(f"Skipping invalid or empty document at index {i}: {repr(doc)}")  # 문제가 있는 문서 정보 출력
    #
    #     document_objects = [Document(page_content=text) for text in valid_documents]
    #
    #     db = FAISS.from_documents(document_objects, self.embeddings)
    #     db.save_local(self.folder_path, index_name)
    #     # self.reload_index(index_name)
    #
    #     # # 문서 임베딩 생성
    #     # document_texts = [doc['text'] for doc in self.documents]
    #     # embeddings = self.embeddings.embed_documents(document_texts)
    #
    #     # # FAISS index에 추가
    #     # embeddings_np = np.array(embeddings).astype('float32')
    #     # self.index.add(embeddings_np)
    #
    #     # # FAISS index 저장
    #     # faiss.write_index(self.index, "faiss_index")
    #     print('Embedding is saved')
    #     # print(f"Saved {len(self.embeddings)} embeddings to faiss_index")
