import os
from pathlib import Path
import multiprocessing as mp

import chromadb


class Chroma:
    @staticmethod
    def init(name: str):
        chroma_client = chromadb.Client()
        collection = chroma_client.get_or_create_collection(name=name)

        return collection


class ChromaPersist:
    @staticmethod
    def init(path: Path, name: str):
        chroma_client = chromadb.PersistentClient(path=path.as_posix())
        collection = chroma_client.get_or_create_collection(name=name)

        return collection


class ChromaServer:
    def __init__(self, path: Path, port=5764) -> None:
        self.process = mp.Process(target=self.init, args=[path.as_posix, port])
        self.process.start()

    def init(self, path: Path, port=5764):
        os.system(f"chroma run --path {path.as_posix} --port {port} > /dev/null")

    def __del__(self):
        self.process.terminate()
        self.process.join()
        del self.process


class ChromaClient:
    def __init__(self, host="localhost", port=5764):
        self.chroma_client = chromadb.HttpClient(host=host, port=port)

    def client(self):
        return self.chroma_client
