import asyncio
import json
import logging
import os
import re
import pandas as pd
import io
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Type, Optional, Any

# OpenMP 환경 변수 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import aiofiles
import numpy as np
import torch
import openai
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from tqdm import tqdm
from langchain.embeddings.base import Embeddings
from constants import FaissIndexMapper
import gc

load_dotenv()
tqdm.monitor_interval = 0
openai.api_key = os.getenv('OPENAI_API_KEY')


# Custom embeddings class using OpenAI
class OpenAIEmbeddings(Embeddings):
    def __init__(self, model="text-embedding-3-small"):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts using OpenAI's embeddings API."""
        embeddings = []
        # Process in batches of 100 to avoid rate limits
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = openai.embeddings.create(
                model=self.model,
                input=batch,
                encoding_format="float"
            )
            embeddings.extend([item.embedding for item in response.data])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single text using OpenAI's embeddings API."""
        response = openai.embeddings.create(
            model=self.model,
            input=[text],
            encoding_format="float"
        )
        return response.data[0].embedding


class VectorDBManager:
    def __init__(self, folder_path='./data/db', device='cpu', encoding="utf-8"):
        print('VectorDBManager is being initialized')
        self.documents = None
        self.vector_db = None
        self.encoding = encoding
        self.folder_path = folder_path
        self.csv_folder_path = "./data/csv_files/"
        self.indexes: Dict[str, FAISS] = {}

        # ThreadPoolExecutor initialization
        self.executor = ThreadPoolExecutor(max_workers=5)

        # Event loop initialization
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            # Initialize embeddings using OpenAIEmbeddings
            self.embeddings = OpenAIEmbeddings()
            print("Successfully initialized OpenAI embeddings")
        except Exception as e:
            print(f"Error initializing OpenAI embeddings: {str(e)}")
            raise

        self._init_faiss_mapper()
        self._load_faiss_files(self.faiss_files)

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
        self.faiss_shopnt = {
            FaissIndexMapper.INDEX_KEY.KEY_SHOPNT.value: FaissIndexMapper.INDEX_VALUE.VALUE_SHOPNT.value
        }
        self.faiss_gymall = {
            FaissIndexMapper.INDEX_KEY.KEY_GYMALL.value: FaissIndexMapper.INDEX_VALUE.VALUE_GYMALL.value
        }

        self.faiss_files = {
            **self.faiss_products,
            **self.faiss_nsmall,
            **self.faiss_wmall,
            **self.faiss_ssgdfs,
            **self.faiss_shopnt,
            **self.faiss_gymall,
        }

    def _load_faiss_files(self, indices: Dict):
        print('loading vector databases')
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
                print(e)
                continue

    def reload_index(self, db_name: str):
        if not db_name:
            raise ValueError(
                "No folder_path to retrieve and faiss_name. Please provide valid folder and index path during to load.")
        try:
            # Clear or reset existing index if it exists
            if db_name in self.indexes:
                if hasattr(self.indexes[db_name], 'reset'):
                    self.indexes[db_name].reset()
                elif hasattr(self.indexes[db_name], 'index') and hasattr(self.indexes[db_name].index, 'reset'):
                    self.indexes[db_name].index.reset()

            # Load FAISS index
            index_name = self.faiss_files[db_name]
            self.indexes[db_name] = FAISS.load_local(
                self.folder_path,
                self.embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True)

            print(f"ReLoaded FAISS index from {index_name} of {db_name}")
        except Exception as e:
            print(e)
            raise

    @staticmethod
    async def _clean_text(text):
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)
        # Keep only Korean, English, numbers, and spaces
        return re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)

    async def _process_csv(self, csv_file: str, columns_to_clean: Optional[List[str]]) -> List[Dict[str, Any]]:
        path = os.path.join(self.csv_folder_path, f"{csv_file}.csv")
        try:
            async with aiofiles.open(path, mode='r', encoding='utf-8-sig') as csvfile:
                file_content = await csvfile.read()
            df = pd.read_csv(io.StringIO(file_content))
            processed_data = df.to_dict('records')

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
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file_path=temp_json_file, encoding=self.encoding)
        raw_documents = await loop.run_in_executor(self.executor, loader.load)

        text_splitter = CharacterTextSplitter(
            separator="\n\n\n",
            chunk_size=chunk_size,
            chunk_overlap=0
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
                              csv_file: str,
                              db_name: str,
                              chunk_size: int = 1000,
                              columns_to_clean: Optional[List[str]] = None):
        try:
            if not csv_file or not db_name:
                raise ValueError("No documents to embed. Please provide valid file path and name.")

            index_name = self.faiss_files.get(db_name)
            if not index_name:
                raise KeyError(f"No index name is defined for database {db_name}")

            processed_data = await self._process_csv(csv_file, columns_to_clean)
            document_objects = await self._prepare_documents(chunk_size, processed_data)

            loop = asyncio.get_event_loop()
            batch_size = 100  # Adjusted for OpenAI API rate limits
            num_documents = len(document_objects)
            vector_store = None

            for i in range(0, num_documents, batch_size):
                batch_documents = document_objects[i:i + batch_size]
                if vector_store is None:
                    vector_store = await loop.run_in_executor(
                        self.executor,
                        FAISS.from_documents,
                        batch_documents,
                        self.embeddings
                    )
                else:
                    await loop.run_in_executor(
                        self.executor,
                        vector_store.add_documents,
                        batch_documents
                    )

                del batch_documents
                gc.collect()

            index_name = self.faiss_files.get(db_name)
            if not index_name:
                raise ValueError(f"No index name found for database {db_name}")

            await loop.run_in_executor(self.executor, vector_store.save_local, self.folder_path, index_name)
            logging.info(f'Embedding is saved for {db_name}')

        except Exception as e:
            logging.error(f"An error occurred while saving embeddings for {db_name}: {str(e)}")
            raise

    async def retrieve(self,
                       db_name: str,
                       query: str,
                       k: int = 3,
                       distance_threshold: float = 0.9) -> list:
        if not db_name or not query:
            raise ValueError("No database name and query to retrieve. Please provide both.")

        try:
            if db_name not in self.indexes:
                index_name = self.faiss_files.get(db_name)
                if not index_name:
                    raise ValueError(f"No index name found for database {db_name}")

                # Load the index if it's not already loaded
                self.indexes[db_name] = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: FAISS.load_local(
                        self.folder_path,
                        self.embeddings,
                        index_name,
                        allow_dangerous_deserialization=True
                    )
                )

            # Use the cached index
            faiss_index = self.indexes[db_name]

            # Perform the search
            docs_and_scores = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: faiss_index.similarity_search_with_score(query, k)
            )

            filtered_results = []
            for doc, distance in docs_and_scores:
                if distance <= distance_threshold:
                    try:
                        filtered_results.append(json.loads(doc.page_content))
                    except json.JSONDecodeError:
                        # If JSON parsing fails, add the raw content
                        filtered_results.append({"content": doc.page_content})

            return filtered_results
        except Exception as e:
            logging.error(f"An error occurred during retrieval from {db_name}: {str(e)}")
            return []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Clean up resources."""
        # Clear FAISS indexes
        for index in self.indexes.values():
            if hasattr(index, 'index') and hasattr(index.index, 'reset'):
                index.index.reset()
        self.indexes.clear()

        # Clean up embeddings
        if hasattr(self, 'embeddings'):
            del self.embeddings

        # Clean CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Shutdown ThreadPoolExecutor
        self.executor.shutdown(wait=True)