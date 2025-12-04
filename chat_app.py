import json
import pickle
import faiss
import numpy as np
import requests
from zenml.client import Client
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/chat"
LLM_MODEL = "qwen2:1.5b" 

class Colors:
    HACIVAT = '\033[94m'
    KARAGOZ = '\033[91m'
    RESET = '\033[0m'

def load_resources_from_zenml():
    print("🌐 ZenML sunucusuna bağlanılıyor...", end=" ", flush=True)
    client = Client()
    
    try:
        
        pipeline_run = client.get_pipeline("rag_ingestion_pipeline").last_successful_run
        
        
        step_name = "create_index_step"
        step = pipeline_run.steps[step_name]
        
        
        if isinstance(step.outputs["output_0"], list):
            serialized_index = step.outputs["output_0"][0].load()
        else:
            serialized_index = step.outputs["output_0"].load()
            
        
        if isinstance(step.outputs["output_1"], list):
            chunks = step.outputs["output_1"][0].load()
        else:
            chunks = step.outputs["output_1"].load()
        
        
        index_bytes = pickle.loads(serialized_index)
        index_arr = np.frombuffer(index_bytes, dtype='uint8')
        index = faiss.deserialize_index(index_arr)
        
        print("Başarılı!")
        print(f"📥 {len(chunks)} satırlık bilgi hafızadan çekildi.")
        
        return index, chunks
        
    except Exception as e:
        print(f"\n HATA: Veri çekilemedi.\nDetay: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def get_ollama_response(question, context, history):
    system_prompt = (
        "Sen kesinlikle bir yapay zeka değilsin. Sen geleneksel Türk gölge oyunu karakterleri Hacivat ve Karagöz'sün. "
        "Asla 'ben bir modelim' veya 'yardımcı olabilirim' gibi robotik cümleler kurma. "
        "Eski İstanbul ağzıyla, komik, kinayeli ve atışmalı konuş. "
        "Karagöz her şeyi yanlış anlasın. Hacivat ise kibar ve bilgili olsun. "
        "Verilen bağlamı (Context) kullanarak cevap ver ama bağlamda bilgi yoksa doğaçlama yap. "
        "Cevabı sadece tiyatro metni formatında ver (HACİVAT: ... KARAGÖZ: ...)."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    messages.extend(history[-4:]) 
    
    user_input = f"BAĞLAM:\n{context}\n\nKULLANICI SORUSU: {question}"
    messages.append({"role": "user", "content": user_input})
    
    payload = {"model": LLM_MODEL, "messages": messages, "stream": True}
    
    print(f"\n{Colors.HACIVAT}🎭 Sahne:{Colors.RESET}")
    full_text = ""
    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    body = json.loads(line)
                    content = body.get("message", {}).get("content", "")
                    print(content, end="", flush=True)
                    full_text += content
    except Exception as e:
        print(f"Ollama Hatası: {e}")
        return "Hata oluştu."
        
    print("\n" + "-"*50)
    return full_text

def main():
    
    index, chunks = load_resources_from_zenml()
    if not index: return

    
    print("🧠 Embedding modeli yükleniyor...", end=" ")
    emb_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Tamam.")
    
    chat_history = []
    print("\n" + "="*40)
    print("🎭 HACİVAT VE KARAGÖZ İLE SOHBET 🎭")
    print("   (Çıkmak için 'q' yazın)")
    print("="*40 + "\n")

    while True:
        try:
            soru = input(f"{Colors.KARAGOZ}Siz:{Colors.RESET} ")
            if soru.lower() in ["q", "exit", "çıkış"]: 
                print("Haydi bana müsaade!")
                break
            
            if not soru.strip(): continue
            
    
            q_vec = emb_model.encode([soru]).astype('float32')
            _, I = index.search(q_vec, k=3)
            
    
            context = "\n".join([chunks[i] for i in I[0]])
            
    
            cevap = get_ollama_response(soru, context, chat_history)
            
            chat_history.append({"role": "user", "content": soru})
            chat_history.append({"role": "assistant", "content": cevap})
            
        except KeyboardInterrupt:
            print("\nÇıkış yapılıyor...")
            break

if __name__ == "__main__":
    main()