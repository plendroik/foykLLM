import json
import torch
from pathlib import Path
from datasets import Dataset
from zenml import pipeline, step
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model


DATA_DIR = Path(__file__).parent / "data"
BASE_MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct" 
OUTPUT_DIR = "zenml_sft_output"
MAX_SEQ_LENGTH = 256


@step
def prepare_dataset_step() -> Dataset:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Klasör bulunamadı: {DATA_DIR}")

    files = list(DATA_DIR.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"'{DATA_DIR}' klasöründe .jsonl dosyası yok!")
    
    file_path = files[0]
    print(f"Veri kaynağı: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data_list = []
    for line in lines:
        if line.strip():
            try:
                obj = json.loads(line)
                data_list.append(obj)
            except: continue


    formatted_data = []
    for i in range(len(data_list) - 1):
        turn1 = data_list[i]
        turn2 = data_list[i+1]
        
        spk1 = turn1.get("speaker", "").upper()
        spk2 = turn2.get("speaker", "").upper()

        if "HACİVAT" in spk1 and "KARAGÖZ" in spk2:
            text = (
                f"<|im_start|>system\nSen Hacivat ve Karagöz oyunundaki komik Karagöz karakterisin.<|im_end|>\n"
                f"<|im_start|>user\nHacivat: {turn1.get('text','')}<|im_end|>\n"
                f"<|im_start|>assistant\nKaragöz: {turn2.get('text','')}<|im_end|>"
            )
            formatted_data.append({"text": text})

    print(f"Hazırlanan veri sayısı: {len(formatted_data)}")
    return Dataset.from_list(formatted_data)


@step
def train_model_step(dataset: Dataset):
    print("Model ve Tokenizer yükleniyor...")
    

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=MAX_SEQ_LENGTH
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)


    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)


    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        optim="paged_adamw_32bit",
        save_strategy="no"
    )


    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    print("Eğitim başlıyor...")
    trainer.train()
    
    final_path = f"{OUTPUT_DIR}/karagoz_adapter"
    model.save_pretrained(final_path)
    print(f"Model başarıyla kaydedildi: {final_path}")

@pipeline
def sft_pipeline():
    ds = prepare_dataset_step()
    train_model_step(ds)

if __name__ == "__main__":
    sft_pipeline()