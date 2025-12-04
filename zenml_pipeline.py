import json
import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from zenml import pipeline, step


DATA_DIR = Path(__file__).parent / "data"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@step
def load_data_step() -> List[str]:
    """
    Data klasöründeki .jsonl ve .txt dosyalarını okur, parçalara böler.
    """
    chunks = []
    print(f"📚 Veriler taranıyor: {DATA_DIR}")
    
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Klasör bulunamadı: {DATA_DIR}")


    files = list(DATA_DIR.glob("*.jsonl")) + list(DATA_DIR.glob("*.txt"))
    
    if not files:
        print("UYARI: Klasörde veri dosyası bulunamadı!")
        return []
    
    for p in files:
        try:
            with open(p, "r", encoding="utf-8") as f:
                if p.suffix == ".jsonl":
                    for line in f:
                        if line.strip():
                            try:
                                obj = json.loads(line)
                                
                                text = f"{obj.get('speaker', 'Bilinmeyen')}: {obj.get('text', '')}"
                                chunks.append(text)
                            except: continue
                else:
                    
                    text = f.read()
                    parts = text.split("\n\n")
                    chunks.extend([p.strip() for p in parts if p.strip()])
        except Exception as e:
            print(f"Dosya okunurken hata: {p.name} - {e}")
                
    print(f"✅ Toplam {len(chunks)} parça veri hafızaya alındı.")
    return chunks


@step
def create_index_step(chunks: List[str]) -> Tuple[bytes, List[str]]:
    """
    Metinleri vektöre çevirir (Embedding) ve FAISS indeksini oluşturur.
    Sonucu byte olarak (pickle) döndürür ki ZenML saklayabilsin.
    """
    if not chunks:
        print(" İşlenecek veri yok, boş indeks dönüyor.")
        return b"", []

    print(" Embeddings ve Index oluşturuluyor (Bu biraz sürebilir)...")
    
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    embeddings = model.encode(chunks)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    index_bytes = faiss.serialize_index(index)
    serialized_index = pickle.dumps(index_bytes)
    
    print("Indeks başarıyla oluşturuldu ve paketlendi.")
    return serialized_index, chunks


@pipeline
def rag_ingestion_pipeline():
    chunks = load_data_step()
    create_index_step(chunks)

if __name__ == "__main__":
    rag_ingestion_pipeline()