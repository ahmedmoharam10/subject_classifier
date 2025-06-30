import json
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset, load_dataset, concatenate_datasets
import evaluate
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from transformers import pipeline
import os
import shutil
import transformers
import inspect

print("Transformers version:", transformers.__version__)
print("TrainingArguments loaded from:", inspect.getfile(transformers.TrainingArguments))

# For dynamic keyword generation
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import WordNetLemmatizer

# For dynamic filename generation
from datetime import datetime

# Download NLTK stopwords if not already downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    print("NLTK 'stopwords' downloaded.")

# Download NLTK resources for lemmatization if not already downloaded
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    print("NLTK 'wordnet' downloaded.")
try:
    nltk.data.find('omw-1.4')
except LookupError:
    nltk.download('omw-1.4')
    print("NLTK 'omw-1.4' downloaded.")

# Set a random seed for reproducibility
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# Define common paths for consistency
MODEL_SAVE_DIR = r"E:\Ahmed Sameh Work\Projects\GAIAthon 25\Classification Model\models"
DATASET_PATH = r"E:\Ahmed Sameh Work\Projects\GAIAthon 25\Classification Model\datasets\dataset.json"
CHEMISTRY_DATA_PATH = r"E:\Ahmed Sameh Work\Projects\GAIAthon 25\Classification Model\datasets\chemistry.csv"
COMPUTER_SCIENCE_DATA_PATH = r"E:\Ahmed Sameh Work\Projects\GAIAthon 25\Classification Model\datasets\Computer Science.csv"
NEW_SAMPLES_DIR = r"E:\Ahmed Sameh Work\Projects\GAIAthon 25\Classification Model\new_samples" # From v4

# ========== STEP 2: LOAD AND PREPARE DATA ==========

# V3's load_data_from_file for CSV support
def load_data_from_file(file_path: str):
    """Loads data from a JSON or CSV file into a Pandas DataFrame."""
    if not os.path.exists(file_path):
        print(f"Warning: File not found at: {file_path}. Returning empty DataFrame.")
        return pd.DataFrame(columns=['text', 'label'])

    _, ext = os.path.splitext(file_path)
    if ext.lower() == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    elif ext.lower() == '.csv':
        df = pd.read_csv(file_path)
    else:
        print(f"Unsupported file format: {ext}. Only .json and .csv are supported.")
        return pd.DataFrame(columns=['text', 'label'])
    
    if 'text' not in df.columns or 'label' not in df.columns:
        print(f"Error: Required columns 'text' and 'label' not found in {file_path}.")
        return pd.DataFrame(columns=['text', 'label'])
        
    return df

# V3's load_local_dataset
def load_local_dataset(dataset_path: str):
    """Loads the dataset from a JSON file (or an empty DataFrame if not found)."""
    df = load_data_from_file(dataset_path)
    print(f"Loaded local dataset with {len(df)} samples from {dataset_path}.")
    if not df.empty:
        print("Local dataset label distribution:\n", df['label'].value_counts())
    return df

def load_additional_huggingface_datasets():
    """Loads and preprocesses Hugging Face datasets with proper error handling and V3's broader engineering coverage."""
    print("\n--- Loading additional Hugging Face datasets ---")
    all_hf_data = []

    # 1. Mathematics: math_qa
    try:
        print("Loading math_qa dataset...")
        math_dataset = load_dataset("math_qa", trust_remote_code=True)
        if 'train' in math_dataset:
            math_df = pd.DataFrame(math_dataset['train'])
            if 'Problem' in math_df.columns:
                math_df = math_df[['Problem']].rename(columns={'Problem': 'text'})
                math_df['label'] = "Mathematics"
                all_hf_data.append(math_df)
                print(f"  Loaded {len(math_df)} samples for Mathematics.")
            else:
                print("  'Problem' column not found. Skipping math_qa.")
        else:
            print("  'train' split not found in math_qa dataset. Skipping.")
    except Exception as e:
        print(f"  Error loading math_qa: {str(e)}")

        # 2. English: boolq, squad
    # boolq
    try:
        print("Loading boolq dataset for English (maximum samples)...")
        boolq_dataset = load_dataset("boolq", split="train")
        boolq_df = boolq_dataset.to_pandas()
        boolq_df['text'] = boolq_df['question'] + " " + boolq_df['passage']
        boolq_df = boolq_df[['text']]
        boolq_df['label'] = "English"
        boolq_df = boolq_df[boolq_df['text'].str.len() > 50]
        all_hf_data.append(boolq_df)
        print(f"  Loaded {len(boolq_df)} samples for English (boolq).")
    except Exception as e:
        print(f"  Error loading boolq: {str(e)}")

    # squad
    try:
        print("Loading squad dataset for English (maximum samples)...")
        squad_dataset = load_dataset("squad", split="train")
        squad_df = squad_dataset.to_pandas()
        squad_df['text'] = squad_df['question'] + " " + squad_df['context']
        squad_df = squad_df[['text']]
        squad_df['label'] = "English"
        squad_df = squad_df[squad_df['text'].str.len() > 50]
        all_hf_data.append(squad_df)
        print(f"  Loaded {len(squad_df)} samples for English (squad).")
    except Exception as e:
        print(f"  Error loading squad: {str(e)}")

    # 3. Biology: pubmed_qa and filtered SciQ (from v3)
    try:
        print("Loading pubmed_qa dataset (maximum samples)...")
        pubmed_dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
        pubmed_df = pubmed_dataset.to_pandas()
        pubmed_df['text'] = pubmed_df['question'] + " " + pubmed_df['context'].apply(lambda x: " ".join(x))
        pubmed_df = pubmed_df[['text']]
        pubmed_df['label'] = "Biology"
        all_hf_data.append(pubmed_df)
        print(f"  Loaded {len(pubmed_df)} samples for Biology (pubmed_qa).")
    except Exception as e:
        print(f"  Error loading pubmed_qa: {str(e)}")

    # SciQ for Biology (filtered)
    try:
        print("Loading SciQ dataset for Biology...")
        sciq_bio_dataset = load_dataset("sciq", split="train")
        sciq_bio_df = sciq_bio_dataset.to_pandas()
        sciq_bio_df['text'] = sciq_bio_df['question'] + " " + sciq_bio_df['support']
        
        bio_keywords = ["cell", "organism", "dna", "protein", "gene", "photosynthesis", "enzyme", "chromo", "biology", "biological", "ecosystem", "evolution"]
        pattern = r'\b(?:' + '|'.join(map(re.escape, bio_keywords)) + r')\b'
        sciq_bio_df = sciq_bio_df[sciq_bio_df['text'].str.contains(pattern, case=False, na=False)].copy()
        
        sciq_bio_df = sciq_bio_df[['text']]
        sciq_bio_df['label'] = "Biology"
        all_hf_data.append(sciq_bio_df)
        print(f"  Loaded {len(sciq_bio_df)} samples for Biology (SciQ filtered).")
    except Exception as e:
        print(f"  Error loading SciQ for Biology: {str(e)}")

    # 4. Physics: ai2_arc, mmlu (high_school_physics, college_physics), openbookqa, filtered SciQ (from v3)
    try:
        print("Loading ai2_arc dataset (maximum samples)...")
        arc_dataset = load_dataset("ai2_arc", "ARC-Challenge", split="train")
        arc_df = arc_dataset.to_pandas()
        arc_df['text'] = arc_df.apply(
            lambda row: row['question'] + " " +
            " ".join([c['text'] for c in row['choices']] if isinstance(row['choices'], list) else []),
            axis=1
        )
        arc_df = arc_df[['text']]
        arc_df['label'] = "Physics"
        arc_df = arc_df[arc_df['text'].str.len() > 30]
        all_hf_data.append(arc_df)
        print(f"  Loaded {len(arc_df)} samples for Physics (ai2_arc).")
    except Exception as e:
        print(f"  Error loading ai2_arc: {str(e)}")

    try:
        print("Loading mmlu high_school_physics dataset (combining all splits: test, validation, dev)...")
        hs_physics_datasets = []
        for split_name in ["test", "validation", "dev"]:
            try:
                dataset = load_dataset("cais/mmlu", "high_school_physics", split=split_name)
                hs_physics_datasets.append(dataset.to_pandas())
            except Exception as e:
                print(f"  Error loading mmlu high_school_physics split '{split_name}': {str(e)}")
        
        if hs_physics_datasets:
            mmlu_hs_physics_df = pd.concat(hs_physics_datasets, ignore_index=True)
            mmlu_hs_physics_df['text'] = mmlu_hs_physics_df.apply(
                lambda row: row['question'] + " " + " ".join(row['choices']), axis=1
            )
            mmlu_hs_physics_df = mmlu_hs_physics_df[['text']]
            mmlu_hs_physics_df['label'] = "Physics"
            all_hf_data.append(mmlu_hs_physics_df)
            print(f"  Loaded {len(mmlu_hs_physics_df)} samples for Physics (mmlu high_school_physics from all splits).")
        else:
            print("  No splits loaded for mmlu high_school_physics. Skipping.")
    except Exception as e:
        print(f"  Overall error loading mmlu high_school_physics: {str(e)}")

    try:
        print("Loading mmlu college_physics dataset (combining all splits: test, validation, dev)...")
        college_physics_datasets = []
        for split_name in ["test", "validation", "dev"]:
            try:
                dataset = load_dataset("cais/mmlu", "college_physics", split=split_name)
                college_physics_datasets.append(dataset.to_pandas())
            except Exception as e:
                print(f"  Error loading mmlu college_physics split '{split_name}': {str(e)}")
        
        if college_physics_datasets:
            mmlu_college_physics_df = pd.concat(college_physics_datasets, ignore_index=True)
            mmlu_college_physics_df['text'] = mmlu_college_physics_df.apply(
                lambda row: row['question'] + " " + " ".join(row['choices']), axis=1
            )
            mmlu_college_physics_df = mmlu_college_physics_df[['text']]
            mmlu_college_physics_df['label'] = "Physics"
            all_hf_data.append(mmlu_college_physics_df)
            print(f"  Loaded {len(mmlu_college_physics_df)} samples for Physics (mmlu college_physics from all splits).")
        else:
            print("  No splits loaded for mmlu college_physics. Skipping.")
    except Exception as e:
        print(f"  Overall error loading mmlu college_physics: {str(e)}")

    try:
        print("Loading openbookqa dataset for Physics...")
        openbookqa_dataset = load_dataset("openbookqa", "additional", split="train")
        openbookqa_df = openbookqa_dataset.to_pandas()
        openbookqa_df['text'] = openbookqa_df['question_stem'] + " " + openbookqa_df['choices'].apply(lambda x: " ".join(x['text']))
        openbookqa_df = openbookqa_df[['text']]
        openbookqa_df['label'] = "Physics"
        all_hf_data.append(openbookqa_df)
        print(f"  Loaded {len(openbookqa_df)} samples for Physics (openbookqa).")
    except Exception as e:
        print(f"  Error loading openbookqa: {str(e)}")

    try:
        print("Loading SciQ dataset for Physics...")
        sciq_physics_dataset = load_dataset("sciq", split="train")
        sciq_physics_df = sciq_physics_dataset.to_pandas()
        sciq_physics_df['text'] = sciq_physics_df['question'] + " " + sciq_physics_df['support']
        
        physics_keywords = ["force", "gravity", "motion", "friction", "physics", "mechanics", "quantum", "relativity"]
        pattern = r'\b(?:' + '|'.join(map(re.escape, physics_keywords)) + r')\b'
        sciq_physics_df = sciq_physics_df[sciq_physics_df['text'].str.contains(pattern, case=False, na=False)].copy()
        
        sciq_physics_df = sciq_physics_df[['text']]
        sciq_physics_df['label'] = "Physics"
        all_hf_data.append(sciq_physics_df)
        print(f"  Loaded {len(sciq_physics_df)} samples for Physics (SciQ filtered).")
    except Exception as e:
        print(f"  Error loading SciQ for Physics: {str(e)}")

    # 5. Computer Science: mmlu (high_school_computer_science, college_computer_science), SciQ (from v3)
    try:
        print("Loading mmlu high_school_computer_science dataset (combining all splits: test, validation, dev)...")
        hs_cs_datasets = []
        for split_name in ["test", "validation", "dev"]:
            try:
                dataset = load_dataset("cais/mmlu", "high_school_computer_science", split=split_name)
                hs_cs_datasets.append(dataset.to_pandas())
            except Exception as e:
                print(f"  Error loading mmlu high_school_computer_science split '{split_name}': {str(e)}")
        
        if hs_cs_datasets:
            mmlu_hs_cs_df = pd.concat(hs_cs_datasets, ignore_index=True)
            mmlu_hs_cs_df['text'] = mmlu_hs_cs_df.apply(
                lambda row: row['question'] + " " + " ".join(row['choices']), axis=1
            )
            mmlu_hs_cs_df = mmlu_hs_cs_df[['text']]
            mmlu_hs_cs_df['label'] = "Computer Science"
            all_hf_data.append(mmlu_hs_cs_df)
            print(f"  Loaded {len(mmlu_hs_cs_df)} samples for Computer Science (mmlu high_school_computer_science from all splits).")
        else:
            print("  No splits loaded for mmlu high_school_computer_science. Skipping.")
    except Exception as e:
        print(f"  Overall error loading mmlu high_school_computer_science: {str(e)}")

    try:
        print("Loading mmlu college_computer_science dataset (combining all splits: test, validation, dev)...")
        college_cs_datasets = []
        for split_name in ["test", "validation", "dev"]:
            try:
                dataset = load_dataset("cais/mmlu", "college_computer_science", split=split_name)
                college_cs_datasets.append(dataset.to_pandas())
            except Exception as e:
                print(f"  Error loading mmlu college_computer_science split '{split_name}': {str(e)}")
        
        if college_cs_datasets:
            mmlu_college_cs_df = pd.concat(college_cs_datasets, ignore_index=True)
            mmlu_college_cs_df['text'] = mmlu_college_cs_df.apply(
                lambda row: row['question'] + " " + " ".join(row['choices']), axis=1
            )
            mmlu_college_cs_df = mmlu_college_cs_df[['text']]
            mmlu_college_cs_df['label'] = "Computer Science"
            all_hf_data.append(mmlu_college_cs_df)
            print(f"  Loaded {len(mmlu_college_cs_df)} samples for Computer Science (mmlu college_computer_science from all splits).")
        else:
            print("  No splits loaded for mmlu college_computer_science. Skipping.")
    except Exception as e:
        print(f"  Overall error loading mmlu college_computer_science: {str(e)}")

    try:
        print("Loading SciQ dataset for Computer Science...")
        sciq_cs_dataset = load_dataset("sciq", split="train")
        sciq_cs_df = sciq_cs_dataset.to_pandas()
        
        sciq_cs_df['text'] = sciq_cs_df['question'] + " " + sciq_cs_df['support']
        
        cs_keywords = [
            "algorithm", "recursion", "multithreading", "parallelism", "concurrency",
            "bitwise", "bytecode", "compilation", "debugging", "runtime", "syntax", "semantics",
            "pseudocode", "pipeline", "register", "cache", "address bus", "data bus",
            "object-oriented", "polymorphism", "encapsulation", "functional programming", 
            "procedural programming", "imperative programming", "stack", "queue", "linked list", 
            "hashmap", "hashtable", "heap", "binary heap", "trie", "graph traversal",
            "divide and conquer", "dynamic programming", "greedy algorithm", "backtracking",
            "depth-first search", "breadth-first search", "dijkstra", "a-star",
            "time complexity", "space complexity", "big-o", "asymptotic", "computational complexity",
            "kernel", "process", "thread", "scheduling", "context switching", "virtual memory", 
            "segmentation fault", "stack overflow", "deadlock", "tcp", "udp", "ip address", 
            "socket", "packet", "dns", "http", "ftp", "relational database", "sql", "query", 
            "primary key", "foreign key", "nosql", "mongodb", "api", "rest", "middleware", 
            "git", "docker", "continuous integration", "microservices", "scrum", "agile", 
            "design pattern", "uml", "encryption", "hashing", "authentication", "authorization", 
            "rsa", "aes", "neural network", "backpropagation", "decision tree", 
            "support vector machine", "clustering", "supervised learning", "recursion", "pointers", "loops"
        ]
        pattern = r'\b(?:' + '|'.join(map(re.escape, cs_keywords)) + r')\b'
        sciq_cs_df = sciq_cs_df[sciq_cs_df['text'].str.contains(pattern, case=False, na=False)].copy()
        sciq_cs_df = sciq_cs_df[['text']]
        sciq_cs_df['label'] = "Computer Science"
        all_hf_data.append(sciq_cs_df)
        print(f"  Loaded {len(sciq_cs_df)} samples for Computer Science (SciQ filtered).")
    except Exception as e:
        print(f"  Error loading SciQ for Computer Science: {str(e)}")

    # Chemistry: SciQ (filtered)
    try:
        print("Loading SciQ dataset for Chemistry...")
        sciq_chem_dataset = load_dataset("sciq", split="train")
        sciq_chem_df = sciq_chem_dataset.to_pandas()
        sciq_chem_df['text'] = sciq_chem_df['question'] + " " + sciq_chem_df['support']
        
        chemistry_keywords = ["chemistry", "chemical", "molecule", "atom", "reaction", "compound", "element", "bond", "acid", "base", "ph", "organic", "inorganic", "compound", "periodic", "solution", "experiment", "material"]
        pattern = r'\b(?:' + '|'.join(map(re.escape, chemistry_keywords)) + r')\b'
        sciq_chem_df = sciq_chem_df[sciq_chem_df['text'].str.contains(pattern, case=False, na=False)].copy()
        
        sciq_chem_df = sciq_chem_df[['text']]
        sciq_chem_df['label'] = "Chemistry"
        all_hf_data.append(sciq_chem_df)
        print(f"  Loaded {len(sciq_chem_df)} samples for Chemistry (SciQ filtered).")
    except Exception as e:
        print(f"  Error loading SciQ for Chemistry: {str(e)}")

    # Engineering Subjects from MMLU (only available configs)
    engineering_mmlu_subjects = {
        "electrical_engineering": "Engineering", # Only keep this as it's available in cais/mmlu
    }

    for subject_key, subject_label in engineering_mmlu_subjects.items():
        try:
            print(f"Loading mmlu {subject_key} dataset (combining all splits: test, validation, dev)...")
            engineering_datasets = []
            for split_name in ["test", "validation", "dev"]:
                try:
                    dataset = load_dataset("cais/mmlu", subject_key, split=split_name)
                    engineering_datasets.append(dataset.to_pandas())
                except Exception as e:
                    print(f"  Error loading mmlu {subject_key} split '{split_name}': {str(e)}")
            
            if engineering_datasets:
                combined_engineering_df = pd.concat(engineering_datasets, ignore_index=True)
                combined_engineering_df['text'] = combined_engineering_df.apply(
                    lambda row: row['question'] + " " + " ".join(row['choices']), axis=1
                )
                combined_engineering_df = combined_engineering_df[['text']]
                combined_engineering_df['label'] = subject_label
                all_hf_data.append(combined_engineering_df)
                print(f"  Loaded {len(combined_engineering_df)} samples for {subject_label} (mmlu {subject_key} from all splits).")
            else:
                print(f"  No splits loaded for mmlu {subject_key}. Skipping.")
        except Exception as e:
            print(f"  Overall error loading mmlu {subject_key}: {str(e)}")

    # --- NEW ADDITIONS for Mechanical, Civil, Industrial Engineering ---
    # Mechanical Engineering
    try:
        print("Loading lamm-mit/MechanicsMaterials dataset for Mechanical Engineering...")
        mech_mat_dataset = load_dataset("lamm-mit/MechanicsMaterials", split="train")
        mech_mat_df = mech_mat_dataset.to_pandas()
        # FIX: Check for 'question' and 'answer' columns and concatenate them
        if 'question' in mech_mat_df.columns and 'answer' in mech_mat_df.columns:
            mech_mat_df['text'] = mech_mat_df['question'] + ' ' + mech_mat_df['answer']
            mech_mat_df['label'] = "Engineering"
            all_hf_data.append(mech_mat_df[['text', 'label']])
            print(f"  Loaded {len(mech_mat_df)} samples for Mechanical Engineering.")
        else:
            print("  'question' or 'answer' column not found. Skipping lamm-mit/MechanicsMaterials.")
    except Exception as e:
        print(f"  Error loading lamm-mit/MechanicsMaterials: {str(e)}")

    # Industrial Engineering
    try:
        print("Loading GainEnergy/oilandgas-engineering-dataset for Industrial Engineering...")
        industrial_dataset = load_dataset("GainEnergy/oilandgas-engineering-dataset", split="train")
        industrial_df = industrial_dataset.to_pandas()
        # The dataset viewer for GainEnergy/oilandgas-engineering-dataset suggests 'abstract' as primary text
        if 'text' in industrial_df.columns:
            industrial_df['text'] = industrial_df['text']
            industrial_df['label'] = "Engineering"
            all_hf_data.append(industrial_df[['text', 'label']])
            print(f"  Loaded {len(industrial_df)} samples for Industrial Engineering.")
        else:
            print("  'abstract' column not found. Skipping GainEnergy/oilandgas-engineering-dataset.")
    except Exception as e:
        print(f"  Error loading GainEnergy/oilandgas-engineering-dataset: {str(e)}")
    # --- END NEW ADDITIONS ---


    if all_hf_data:
        combined_df = pd.concat(all_hf_data, ignore_index=True)
        print(f"\nTotal samples loaded: {len(combined_df)}")
        print("Label distribution:\n", combined_df['label'].value_counts())
        return combined_df
    
    print("No additional datasets loaded - using local data only")
    return pd.DataFrame(columns=['text', 'label'])

# Custom Dataset class
class SimpleTextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Tokenize function (global so it can be used by Trainer)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

# ========== STEP 3: TRAIN THE MODEL ==========
def train_model(train_dataset, eval_dataset, label_encoder, num_epochs: int = 3, learning_rate: float = 5e-5, output_dir: str = "./results"):
    """Trains a BERT classification model."""
    print("\nTraining the model...")

    # Load pre-trained model
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(label_encoder.classes_),
        id2label={i: label for i, label in enumerate(label_encoder.classes_)},
        label2id={label: i for i, label in enumerate(label_encoder.classes_)}
    )

    # Define metrics
    metric = evaluate.load("f1") 

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=1)
        # Using 'weighted' average for F1, precision, recall in multi-class to account for label imbalance
        f1 = metric.compute(predictions=predictions, references=labels, average='weighted')['f1']
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(labels, predictions, average='weighted', zero_division=0)
        return {"f1": f1, "accuracy": accuracy, "precision": precision, "recall": recall}

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",      
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,    
        learning_rate=learning_rate,
        report_to="none" # Disable integrations
    )    

    # Data collator for batching
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    # Train and save the model
    trainer.train()
    trainer.save_model(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)
    
    # Save label encoder mapping
    label_map = {i: label for i, label in enumerate(label_encoder.classes_)}
    with open(os.path.join(MODEL_SAVE_DIR, "label_map.json"), "w") as f:
        json.dump(label_map, f)
    
    print(f"Model, tokenizer, and label map saved to {MODEL_SAVE_DIR}")
    return model, trainer, label_encoder

# ========== STEP 4: LOAD THE TRAINED MODEL ==========
def load_model(model_path: str = MODEL_SAVE_DIR):
    """Loads a trained BERT model, tokenizer, and label encoder."""
    print(f"\nLoading model from {model_path}...")
    try:
        loaded_tokenizer = BertTokenizer.from_pretrained(model_path)
        with open(os.path.join(model_path, "label_map.json"), "r") as f:
            label_map = json.load(f)
        le = LabelEncoder()
        le.classes_ = np.array([label_map[str(i)] for i in sorted(label_map, key=int)])
        loaded_model = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=len(le.classes_),
            id2label=label_map,
            label2id={label: i for i, label in enumerate(le.classes_)}
        )
        print("Model, tokenizer, and label encoder loaded successfully.")
        return loaded_model, loaded_tokenizer, le
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

# ========== STEP 5: RETRAIN WITH NEW DATA ==========
def retrain_with_new_data(current_dataset_path: str, new_data_base_dir: str, retrain_trigger: bool = False):
    print("\nChecking for new samples to retrain the model...")
    new_samples_found = False
    new_samples_data = []

    # Iterate through files to find new_samples_*.json
    for filename in os.listdir(new_data_base_dir):
        if filename.startswith("new_samples_") and filename.endswith(".json"):
            new_samples_file_path = os.path.join(new_data_base_dir, filename)
            try:
                with open(new_samples_file_path, 'r', encoding='utf-8') as f:
                    new_data = json.load(f)
                if isinstance(new_data, list):
                    for item in new_data:
                        item['timestamp'] = datetime.now().isoformat() # Keep v4 timestamping
                    new_samples_data.extend(new_data)
                else:
                    new_data['timestamp'] = datetime.now().isoformat()
                    new_samples_data.append(new_data)
                print(f"Found and loaded new samples from: {filename}")
                new_samples_found = True
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {filename}: {e}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    if new_samples_found and new_samples_data or retrain_trigger: # Add v3's manual retrain trigger
        print("New samples found or retraining manually triggered. Merging and retraining...")
        
        # Load existing dataset
        existing_df = load_local_dataset(current_dataset_path)
        
        # Convert new samples to DataFrame and deduplicate (v4's deduplication)
        new_df = pd.DataFrame(new_samples_data)
        if not new_df.empty:
            # V4's deduplication approach
            combined_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=['text']).reset_index(drop=True)
            print(f"Combined dataset size after deduplication: {len(combined_df)} (from {len(existing_df)} existing and {len(new_df)} new samples).")
        else:
            combined_df = existing_df
            print("No new valid samples to add to the dataset.")

        # Save the combined dataset (overwriting the old one)
        combined_df.to_json(current_dataset_path, orient='records', indent=4)
        print(f"Updated dataset saved to {current_dataset_path}")

        # Clear new samples files after merging
        for filename in os.listdir(new_data_base_dir):
            if filename.startswith("new_samples_") and filename.endswith(".json"):
                file_path = os.path.join(new_data_base_dir, filename)
                try:
                    os.remove(file_path)
                    print(f"Removed processed new samples file: {filename}")
                except Exception as e:
                    print(f"Error removing file {filename}: {e}")
        
        # Prepare for retraining
        return prepare_dataset(combined_df)
    else:
        print("No new samples found and no manual retrain trigger. Skipping retraining.")
        return None, None, None, None

# Prepare dataset function (adapted for combined loading)
def prepare_dataset(df: pd.DataFrame):
    """Prepares the dataset for training."""
    if df.empty:
        print("Error: No data to prepare for training.")
        return None, None, None, None

    # V3's NaN handling and label normalization
    original_len = len(df)
    df['label'] = df['label'].replace(['nan', 'Nan', 'NAN', 'None'], pd.NA)
    df.dropna(subset=['text', 'label'], inplace=True)
    df['label'] = df['label'].astype(str).str.strip().str.title().replace('Math', 'Mathematics')\
                .replace('Chemistry ', 'Chemistry') 
    if len(df) < original_len:
        print(f"Removed {original_len - len(df)} rows due to missing text or label, or NaN labels.")

    print("Unique labels in dataset after processing:", df['label'].unique())
    if len(df['label'].unique()) < 2:
        print("Error: Not enough unique labels (at least 2 required) to perform classification. Exiting.")
        return None, None, None, None

    label_encoder = LabelEncoder()
    df['encoded_label'] = label_encoder.fit_transform(df['label'])

    X = df['text'].tolist()
    y = df['encoded_label'].tolist()

    # V3's explicit stratification checks
    min_class_count = df['encoded_label'].value_counts().min()
    if min_class_count < 2: # For stratified split, each class needs at least 2 samples
        print(f"Warning: Smallest class has only {min_class_count} sample(s). Cannot perform stratified split. Falling back to non-stratified split.")
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

    train_dataset = SimpleTextDataset(train_encodings, train_labels)
    eval_dataset = SimpleTextDataset(val_encodings, val_labels)

    return train_dataset, eval_dataset, label_encoder, df

# ========== STEP 6: DYNAMIC KEYWORD GENERATION (COMBINED V3 & V4) ==========

# V3's Hardcoded CS keywords + V4's lemmatization and dynamic approach + V3's generic term blacklist
def generate_dynamic_keywords_from_dataset(dataset_df, label_encoder):
    print("\nGenerating dynamic keywords from the dataset...")
    lemmatizer = WordNetLemmatizer() # V4's lemmatizer

    stop_words = set(stopwords.words('english'))
    # V3's explicit generic term blacklist
    generic_blacklist = {"describe", "explain", "discuss", "what", "how", "define", "identify", "list", "compare", "contrast", "provide", "example"} 
    stop_words.update(generic_blacklist)

    dynamic_keywords = {}
    for label_id, label_name in enumerate(label_encoder.classes_):
        subject_texts = dataset_df[dataset_df['encoded_label'] == label_id]['text'].tolist()
        combined_text = " ".join(subject_texts).lower()
        
        # Remove non-alphabetic characters and tokenize
        words = re.findall(r'\b[a-z]{2,}\b', combined_text)
        
        # Filter out stopwords and lemmatize (V4's lemmatization)
        filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        
        word_counts = Counter(filtered_words)
        
        # V4's dynamic thresholding
        if word_counts:
            most_common_word_count = word_counts.most_common(1)[0][1]
            common_word_threshold = most_common_word_count * 0.05 # Adjust threshold as needed
            
            # Filter out overly common words (dynamic thresholding)
            subject_keywords = [word for word, count in word_counts.items() if count > common_word_threshold]
        else:
            subject_keywords = []

        # Add V3's hardcoded CS/Engineering keywords
        if label_name == "Computer Science":
            cs_hardcoded_keywords = [
                "algorithm", "recursion", "multithreading", "parallelism", "concurrency",
                "bitwise", "bytecode", "compilation", "debugging", "runtime", "syntax", "semantics",
                "pseudocode", "pipeline", "register", "cache", "address bus", "data bus",
                "object-oriented", "polymorphism", "encapsulation", "functional programming", 
                "procedural programming", "imperative programming", "stack", "queue", "linked list", 
                "hashmap", "hashtable", "heap", "binary heap", "trie", "graph traversal",
                "divide and conquer", "dynamic programming", "greedy algorithm", "backtracking",
                "depth-first search", "breadth-first search", "dijkstra", "a-star",
                "time complexity", "space complexity", "big-o", "asymptotic", "computational complexity",
                "kernel", "process", "thread", "scheduling", "context switching", "virtual memory", 
                "segmentation fault", "stack overflow", "deadlock", "tcp", "udp", "ip address", 
                "socket", "packet", "dns", "http", "ftp", "relational database", "sql", "query", 
                "primary key", "foreign key", "nosql", "mongodb", "api", "rest", "middleware", 
                "git", "docker", "continuous integration", "microservices", "scrum", "agile", 
                "design pattern", "uml", "encryption", "hashing", "authentication", "authorization", 
                "rsa", "aes", "neural network", "backpropagation", "decision tree", 
                "support vector machine", "clustering", "supervised learning", "recursion", "pointers", "loops"
            ]
            subject_keywords.extend([kw for kw in cs_hardcoded_keywords if kw not in subject_keywords])
        elif label_name == "Engineering":
            engineering_hardcoded_keywords = [
                "mechanical", "electrical", "industrial", "civil", "aerospace", "chemical", "materials",
                "thermodynamics", "fluid dynamics", "mechanics", "robotics", "automation", "circuit",
                "power systems", "control systems", "manufacturing", "supply chain", "logistics",
                "structural", "design", "stress", "strain", "finite element", "cad", "cam",
                "metallurgy", "composites", "nanotechnology"
            ]
            subject_keywords.extend([kw for kw in engineering_hardcoded_keywords if kw not in subject_keywords])
        elif label_name == "Chemistry": # Add keywords for chemistry (from v3 filter)
            chemistry_hardcoded_keywords = ["chemistry", "chemical", "molecule", "atom", "reaction", "compound", "element", "bond", "acid", "base", "ph", "organic", "inorganic", "compound", "periodic", "solution", "experiment", "material"]
            subject_keywords.extend([kw for kw in chemistry_hardcoded_keywords if kw not in subject_keywords])
        elif label_name == "Physics": # Add keywords for physics (from v3 filter)
            physics_hardcoded_keywords = ["force", "gravity", "motion", "friction", "physics", "mechanics", "quantum", "relativity"]
            subject_keywords.extend([kw for kw in physics_hardcoded_keywords if kw not in subject_keywords])
        elif label_name == "Biology": # Add keywords for biology (from v3 filter)
            bio_hardcoded_keywords = ["cell", "organism", "dna", "protein", "gene", "photosynthesis", "enzyme", "chromo", "biology", "biological", "ecosystem", "evolution"]
            subject_keywords.extend([kw for kw in bio_hardcoded_keywords if kw not in subject_keywords])


        dynamic_keywords[label_name] = list(set(subject_keywords)) # Ensure unique keywords
    return dynamic_keywords

# ========== STEP 7: CLASSIFICATION LOGIC (COMBINED V3 & V4) ==========

# V4's hybrid approach, with V3's NaN checks and label normalization applied in prepare_dataset
def classify_with_unsupervised(text_input: str, dynamic_keywords: dict, classifier_pipeline):
    """
    Classifies text using a hybrid approach:
    1. Keyword matching for strong signals.
    2. Fallback to fine-tuned BERT model if keyword match is inconclusive.
    """
    lemmatizer = WordNetLemmatizer()
    text_input_processed = " ".join([lemmatizer.lemmatize(word) for word in re.findall(r'\b[a-z]{2,}\b', text_input.lower())])

    # 1. Keyword Matching (Hybrid: V4's dynamic + V3's specific)
    # This part should ideally leverage the dynamic_keywords generated using the combined logic.
    keyword_scores = {label: 0 for label in dynamic_keywords}
    for label, keywords in dynamic_keywords.items():
        for keyword in keywords:
            if keyword in text_input_processed:
                keyword_scores[label] += 1
    
    # Check for strong keyword match
    max_keyword_score = 0
    best_keyword_label = None
    for label, score in keyword_scores.items():
        if score > max_keyword_score:
            max_keyword_score = score
            best_keyword_label = label

    # A simple threshold for keyword confidence. Adjust as needed.
    # If a category has at least 2 keywords, consider it a strong signal.
    if max_keyword_score >= 2:
        print(f"  (Keyword match: '{best_keyword_label}' with score {max_keyword_score})")
        return best_keyword_label
    
    # 2. Fallback to Fine-tuned Model (from V4)
    print("  (No strong keyword match, falling back to BERT classifier)")
    try:
        # V3's detailed pipeline failure messages
        if classifier_pipeline is None:
            raise ValueError("Classifier pipeline is not loaded. Cannot perform BERT classification.")

        # Ensure the input is a list of strings
        if not isinstance(text_input, list):
            text_input = [text_input]

        # Use the pipeline for inference
        predictions = classifier_pipeline(text_input)

        # The prediction output structure can vary based on the pipeline.
        # Assuming it returns a list of dicts, e.g., [{'label': 'LABEL_0', 'score': 0.99}]
        if predictions and len(predictions) > 0 and isinstance(predictions[0], dict) and 'label' in predictions[0]:
            # Convert pipeline label (e.g., 'LABEL_0') back to original label name
            predicted_id = int(predictions[0]['label'].split('_')[-1])
            predicted_label = classifier_pipeline.model.config.id2label[predicted_id]
            return predicted_label
        else:
            raise ValueError("Unexpected output format from classifier pipeline.")

    except Exception as e:
        # V3's detailed pipeline failure messages
        print(f"  Error during BERT classification: {e}")
        return "Unknown" # Or handle as appropriate
def generate_keywords_from_dataset(dataset: Dataset, label_encoder, top_n: int = 15, common_word_threshold: float = 0.005):
    """
    Generates subject-specific keywords from the dataset.
    Removes stopwords and common words across all categories.
    Applies lemmatization to keywords.
    """
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    all_words = []
    category_words = {label: [] for label in label_encoder.classes_}

    for example in dataset:
        text = example['text']
        label_id = example['label']
        label = label_encoder.inverse_transform([label_id])[0]

        words = re.findall(r'\b\w+\b', text.lower())
        
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
        
        all_words.extend(lemmatized_words)
        category_words[label].extend(lemmatized_words)

    overall_word_counts = Counter(all_words)
    total_words = sum(overall_word_counts.values())

    explicit_generic_terms = {
        "describe", "explain", "outline", "define", "state", "identify", "compare",
        "number", "function", "cell", "energy", "solution", "power", "base",
        "structure", "system", "process", "information", "table", "write", "equation", "form"
    }

    common_words_to_exclude = {
        word for word, count in overall_word_counts.items()
        if count / total_words > common_word_threshold
    }
    common_words_to_exclude.update(explicit_generic_terms)

    dynamic_subject_keywords = {}
    for label, words in category_words.items():
        category_word_counts = Counter(words)
        
        filtered_words = []
        for word, count in category_word_counts.most_common():
            if word not in common_words_to_exclude and word not in stop_words:
                filtered_words.append(word)
        
        dynamic_subject_keywords[label] = filtered_words[:top_n]

    return dynamic_subject_keywords


# ========== STEP 7: MAIN EXECUTION BLOCK ==========
if __name__ == "__main__":
    current_dataset_path = DATASET_PATH
    new_samples_base_dir = os.path.dirname(DATASET_PATH)

    # --- Load existing local dataset (dataset.json) ---
    initial_df = load_local_dataset(current_dataset_path)

    # --- Load chemistry.csv ---
    chemistry_df = load_data_from_file(CHEMISTRY_DATA_PATH)
    if not chemistry_df.empty:
        print(f"Loaded {len(chemistry_df)} samples from chemistry.csv.")
        
        # Robust NaN handling for chemistry_df before normalization
        chemistry_df['label'] = chemistry_df['label'].replace(['nan', 'Nan', 'NAN', 'None'], np.nan)
        chemistry_df.dropna(subset=['label'], inplace=True)

        # Normalize labels from chemistry_df
        chemistry_df['label'] = chemistry_df['label'].astype(str).str.strip().str.title().replace({'Math': 'Mathematics'})
        print("chemistry.csv label distribution (normalized):\n", chemistry_df['label'].value_counts())
        initial_df = pd.concat([initial_df, chemistry_df], ignore_index=True)
        print(f"Combined local dataset and chemistry.csv. New local dataset size: {len(initial_df)}")
        initial_df.drop_duplicates(subset=['text'], inplace=True)
        print(f"After deduplication, local dataset size: {len(initial_df)}")

    # --- Load Computer Science.csv ---
    computer_science_df = load_data_from_file(COMPUTER_SCIENCE_DATA_PATH)
    if not computer_science_df.empty:
        print(f"Loaded {len(computer_science_df)} samples from Computer Science.csv.")
        
        # Robust NaN handling for computer_science_df before normalization
        computer_science_df['label'] = computer_science_df['label'].replace(['nan', 'Nan', 'NAN', 'None'], np.nan)
        computer_science_df.dropna(subset=['label'], inplace=True)

        # Normalize labels and set to "Computer Science"
        computer_science_df['label'] = "Computer Science" # Force label to "Computer Science"
        print("Computer Science.csv label distribution (normalized):\n", computer_science_df['label'].value_counts())
        initial_df = pd.concat([initial_df, computer_science_df], ignore_index=True)
        print(f"Combined with Computer Science.csv. New local dataset size: {len(initial_df)}")
        initial_df.drop_duplicates(subset=['text'], inplace=True)
        print(f"After deduplication, local dataset size: {len(initial_df)}")


    # Robust NaN handling for initial_df after chemistry merge
    initial_df['label'] = initial_df['label'].replace(['nan', 'Nan', 'NAN', 'None'], np.nan)
    initial_df.dropna(subset=['label'], inplace=True)
    initial_df['label'] = initial_df['label'].astype(str).str.strip().str.title().replace('Math', 'Mathematics')


    # --- Load and combine additional Hugging Face datasets ---
    hf_df = load_additional_huggingface_datasets()

    # Robust NaN handling for hf_df
    if not hf_df.empty:
        hf_df['label'] = hf_df['label'].replace(['nan', 'Nan', 'NAN', 'None'], np.nan)
        hf_df.dropna(subset=['label'], inplace=True)
        hf_df['label'] = hf_df['label'].astype(str).str.strip().str.title().replace('Math', 'Mathematics')


    # Combine all datasets
    if not hf_df.empty:
        combined_df = pd.concat([initial_df, hf_df], ignore_index=True)
        initial_combined_count = len(combined_df)
        combined_df.drop_duplicates(subset=['text'], inplace=True)
        duplicates_removed_overall = initial_combined_count - len(combined_df)
        if duplicates_removed_overall > 0:
            print(f"Removed {duplicates_removed_overall} duplicate entries after combining all datasets.")
        
        final_dataset_df = combined_df
    else:
        print("\nNo Hugging Face datasets added. Proceeding with only the local dataset (and chemistry.csv if loaded).")
        final_dataset_df = initial_df

    # Final robust NaN handling for final_dataset_df after all concatenations
    nan_labels_before_final = final_dataset_df['label'].isnull().sum()
    final_dataset_df['label'] = final_dataset_df['label'].replace(['nan', 'Nan', 'NAN', 'None'], np.nan)
    final_dataset_df.dropna(subset=['label'], inplace=True)
    nan_labels_after_final = final_dataset_df['label'].isnull().sum()
    if nan_labels_before_final > nan_labels_after_final:
        print(f"Removed {nan_labels_before_final - nan_labels_after_final} rows with missing labels from the final dataset.")
    
    # Re-normalize labels for final dataset after final NaN removal and before label encoding
    final_dataset_df['label'] = final_dataset_df['label'].astype(str).str.strip().str.title().replace('Math', 'Mathematics')

    # --- ADDED: Print label counts here ---
    print("\n--- Label counts for all combined and processed datasets ---")
    print(final_dataset_df['label'].value_counts().to_string())
    print("----------------------------------------------------------")
    # --- END ADDED CODE ---

    labels = final_dataset_df['label'].tolist()
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    final_dataset_dict = {'text': final_dataset_df['text'].tolist(), 'label': encoded_labels.tolist()}
    raw_dataset = Dataset.from_dict(final_dataset_dict)

    # FIX: Calculate counts for stratification check directly from encoded_labels using numpy
    unique_labels, counts = np.unique(encoded_labels, return_counts=True)
    can_stratify = all(c >= 2 for c in counts)

    if can_stratify:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            raw_dataset['text'], raw_dataset['label'], test_size=0.1, random_state=42, stratify=raw_dataset['label']
        )
    else:
        print("Warning: Cannot use stratified split due to very small classes (some classes have fewer than 2 samples). Performing non-stratified split.")
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            raw_dataset['text'], raw_dataset['label'], test_size=0.1, random_state=42
        )

    train_hf_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    eval_hf_dataset = Dataset.from_dict({'text': val_texts, 'label': val_labels})

    tokenized_train_dataset = train_hf_dataset.map(tokenize_function, batched=True)
    tokenized_eval_dataset = eval_hf_dataset.map(tokenize_function, batched=True)

    if os.path.exists(MODEL_SAVE_DIR) and os.path.exists(os.path.join(MODEL_SAVE_DIR, "label_map.json")):
        model, tokenizer_loaded, label_encoder_loaded = load_model(MODEL_SAVE_DIR)
        
        if model is None:
            print("Existing model could not be loaded. Retraining from scratch.")
            model, _, label_encoder_loaded = train_model(tokenized_train_dataset, tokenized_eval_dataset, label_encoder)
        else:
            print("Existing model loaded successfully.")
            tokenizer = tokenizer_loaded
            label_encoder = label_encoder_loaded

    else:
        print("No existing model found. Training a new model.")
        model, _, label_encoder = train_model(tokenized_train_dataset, tokenized_eval_dataset, label_encoder)

    print("\nGenerating dynamic keywords from the current dataset...")
    # Robust NaN handling for keyword generation dataframe
    df_for_keywords = final_dataset_df.copy()
    df_for_keywords['label'] = df_for_keywords['label'].replace(['nan', 'Nan', 'NAN', 'None'], np.nan)
    nan_labels_before_keywords = df_for_keywords['label'].isnull().sum()
    df_for_keywords.dropna(subset=['label'], inplace=True)
    nan_labels_after_keywords = df_for_keywords['label'].isnull().sum()
    if nan_labels_before_keywords > nan_labels_after_keywords:
        print(f"Removed {nan_labels_before_keywords - nan_labels_after_keywords} rows with missing labels for keyword generation.")

    updated_labels_encoded = label_encoder.transform(df_for_keywords['label'].tolist())
    updated_raw_dataset = Dataset.from_dict({'text': df_for_keywords['text'].tolist(), 'label': updated_labels_encoded.tolist()})

    dynamic_keywords = generate_keywords_from_dataset(updated_raw_dataset, label_encoder)
    print("\nDynamic keywords generated:")
    for subject, keywords in dynamic_keywords.items():
        print(f"  {subject}: {keywords}")

    classifier_pipeline = None
    try:
        classifier_pipeline = pipeline(
            "text-classification",
            model=MODEL_SAVE_DIR,
            tokenizer=MODEL_SAVE_DIR,
            device=0 if torch.cuda.is_available() else -1
        )
        print("\nHugging Face classification pipeline loaded.")

    except ImportError:
        print("\nHugging Face `pipeline` not available. Install with `pip install transformers[sentencepiece]` for easier inference.")
    except Exception as e:
        print(f"\nCould not load model with pipeline: {e}")
        print("This might be due to a very small/imbalanced dataset leading to issues with pipeline's internal label mapping, or simply the model not being diverse enough for the pipeline to load.")

    if classifier_pipeline:
        print("\n=== Testing the trained classifier with hybrid approach ===")
        while True:
            user_input = input("Enter text to classify (or a file path to classify its content) (or 'quit' to exit, 'retrain' to retrain): ")
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'retrain':
                model, _, label_encoder = retrain_with_new_data(current_dataset_path, new_samples_base_dir)
                
                print("\nRe-generating dynamic keywords after retraining...")
                updated_df_for_keywords = load_local_dataset(current_dataset_path)
                
                # Robust NaN handling for keyword generation dataframe after retraining
                updated_df_for_keywords['label'] = updated_df_for_keywords['label'].replace(['nan', 'Nan', 'NAN', 'None'], np.nan)
                nan_labels_before_keywords_retrain = updated_df_for_keywords['label'].isnull().sum()
                updated_df_for_keywords.dropna(subset=['label'], inplace=True)
                nan_labels_after_keywords_retrain = updated_df_for_keywords['label'].isnull().sum()
                if nan_labels_before_keywords_retrain > nan_labels_after_keywords_retrain:
                    print(f"Removed {nan_labels_before_keywords_retrain - nan_labels_after_keywords_retrain} rows with missing labels for keyword generation after retraining.")
                
                updated_df_for_keywords['label'] = updated_df_for_keywords['label'].astype(str).str.strip().str.title().replace('Math', 'Mathematics')

                updated_labels_encoded = label_encoder.transform(updated_df_for_keywords['label'].tolist())
                updated_raw_dataset = Dataset.from_dict({'text': updated_df_for_keywords['text'].tolist(), 'label': updated_labels_encoded.tolist()})
                
                dynamic_keywords = generate_keywords_from_dataset(updated_raw_dataset, label_encoder)
                print("\nDynamic keywords regenerated:")
                for subject, keywords in dynamic_keywords.items():
                    print(f"  {subject}: {keywords}")
                continue

            if os.path.exists(user_input):
                print(f"\n--- Classifying content from file: '{user_input}' ---")
                try:
                    with open(user_input, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                    
                    if file_content.strip():
                        classified_label = classify_with_unsupervised(file_content, dynamic_keywords, classifier_pipeline)
                        print(f"-> Classified File Content as: {classified_label}")
                    else:
                        print("File is empty or contains no readable text.")
                except UnicodeDecodeError:
                    print(f"Error reading or processing file '{user_input}': 'utf-8' codec can't decode. This might not be a text file (e.g., PDF, image). Only plain text files are supported for direct reading.")
                except Exception as e:
                    print(f"Error reading or processing file '{user_input}': {e}")
            else:
                classified_label = classify_with_unsupervised(user_input, dynamic_keywords, classifier_pipeline)
                print(f"-> Classified Text Input as: {classified_label}")