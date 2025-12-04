Hacivat & Karagöz: Uçtan Uca LLMOps Projesi
Bu proje, Modern Büyük Dil Modeli (LLM) tekniklerini kullanarak Geleneksel Türk Gölge Oyunu karakterleri Hacivat ve Karagöz'ü canlandıran bir yapay zeka asistanıdır.

Proje, Supervised Fine-Tuning (SFT) ile modele karakterlerin üslubunu öğretmeyi ve RAG (Retrieval-Augmented Generation) ile modele spesifik senaryo bilgisini kazandırmayı amaçlar. Tüm süreç ZenML ile profesyonel bir MLOps boru hattı (pipeline) üzerinde yönetilmiştir.

🚀 Proje Özellikleri
Supervised Fine-Tuning (SFT): Qwen2-1.5B modeli, QLoRA tekniği kullanılarak Hacivat ve Karagöz diyalogları üzerinde eğitildi. Model, "Eski İstanbul Türkçesi" ve karakterlerin mizahi üslubunu (yanlış anlamalar, atışmalar) öğrendi.

RAG (Retrieval-Augmented Generation): .jsonl formatındaki senaryo verileri vektörlere (embeddings) dönüştürülerek FAISS üzerinde indekslendi. Model, sorulan soruya en uygun bağlamı bu veritabanından çeker.

MLOps Pipeline (ZenML): Veri işleme, eğitim ve indeksleme süreçleri ZenML pipeline'ları ile modüler, izlenebilir ve tekrar edilebilir hale getirildi.

Önbellekleme (Caching): ZenML sayesinde işlenen veriler ve oluşturulan indeksler saklanır; tekrar tekrar hesaplama yapılmaz.

Lokal Çıkarım (Inference): Tüm sistem Ollama üzerinden yerel kaynaklarla (Local GPU) çalışır.

🛠️ Kullanılan Teknolojiler
Orkestrasyon: ZenML

Model: Qwen2-1.5B-Instruct (Ollama üzerinden)

Fine-Tuning: Hugging Face transformers, peft (LoRA), bitsandbytes (4-bit Quantization)

Vektör Veritabanı: FAISS, sentence-transformers

Donanım: NVIDIA GPU (CUDA)

📂 Proje Yapısı
Bash

llm/
├── data/                   # Eğitim ve RAG verileri (.jsonl)
├── train_sft.py            # Modeli eğiten kod (Fine-Tuning Pipeline)
├── zenml_pipeline.py       # RAG veritabanını oluşturan kod (Ingestion Pipeline)
├── chat_app.py             # Kullanıcı arayüzü (Inference / Chatbot)
└── zenml_sft_output/       # Eğitilen model adaptörlerinin (LoRA) çıktısı
⚙️ Kurulum
Gerekli kütüphaneleri yükleyin:

Bash

pip install zenml transformers datasets peft bitsandbytes accelerate torch faiss-cpu sentence-transformers
Ollama'yı kurun ve temel modeli çekin:

Bash

ollama pull qwen2:1.5b
ZenML'i başlatın:

Bash

zenml init
🏃‍♂️ Çalıştırma Adımları
1. Modeli Eğitme (SFT)
Modele Hacivat-Karagöz üslubunu öğretmek için eğitimi başlatın:

Bash

python train_sft.py
2. Bilgi Bankasını Oluşturma (RAG Ingestion)
Veri setini okuyup vektör veritabanını oluşturmak ve ZenML Artifact Store'a kaydetmek için:

Bash

python zenml_pipeline.py
3. Sohbeti Başlatma (Chat App)
Eğitilmiş indeksleri ZenML'den çekip Hacivat ve Karagöz ile konuşmak için:

Bash

python chat_app.py
📊 Örnek Çıktılar
Kullanıcı: Yar bana bir eğlence medet!

HACİVAT: Aman efendim, hoş geldiniz sefalar getirdiniz! Gönül neşe ister, kahve bahane...

KARAGÖZ: Hoş bulduk kel kafalı kara üzüm! Ne bağırıp duruyorsun sabah sabah?

Kullanıcı: Hacivat, bana biraz malumat verir misin?

HACİVAT: Efendim, ilim ilim bilmektir, ilim kendin bilmektir. Sana ne hakkında malumat lazım?

KARAGÖZ: Ne? Mahallede turşu mu satacaksın?

🔮 Gelecek Planları
Daha büyük bir model (Qwen2.5-7B veya Llama-3-8B) ile dil yeteneğini artırmak.

Ollama yerine eğitilen LoRA adaptörünü doğrudan sisteme entegre etmek.

ZenML Dashboard üzerinden deney takibi ve model versiyonlama.