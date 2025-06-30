import json
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import RobertaTokenizer, RobertaForSequenceClassification
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
from functools import lru_cache

import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import WordNetLemmatizer

from datetime import datetime

from transformers import EarlyStoppingCallback
from transformers.trainer_utils import IntervalStrategy


try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    print("NLTK 'stopwords' downloaded.")

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

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

MODEL_SAVE_DIR = r"E:\Ahmed Sameh Work\Projects\GAIAthon 25\classification_model\models"
DATASET_PATH = r"E:\Ahmed Sameh Work\Projects\GAIAthon 25\classification_model\datasets\dataset.json"
CHEMISTRY_DATA_PATH = r"E:\Ahmed Sameh Work\Projects\GAIAthon 25\classification_model\datasets\chemistry.csv"
COMPUTER_SCIENCE_DATA_PATH = r"E:\Ahmed Sameh Work\Projects\GAIAthon 25\classification_model\datasets\Computer Science.csv"
NEW_SAMPLES_DIR = r"E:\Ahmed Sameh Work\Projects\GAIAthon 25\classification_model\new_samples"

@lru_cache(maxsize=None)
def load_data_from_file(file_path: str):
    if not os.path.exists(file_path):
        print(f"Warning: File not found at: {file_path}. Returning empty DataFrame.")
        return pd.DataFrame(columns=['text', 'label'])

    try:
        _, ext = os.path.splitext(file_path)
        if ext.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                if 'text' in data and 'label' in data:
                    df = pd.DataFrame([data])
                else:
                    df = pd.DataFrame(data.items(), columns=['text', 'label'])
            else:
                df = pd.DataFrame(data)
        elif ext.lower() == '.csv':
            df = pd.read_csv(file_path)
            if 'Text' in df.columns and 'Label' in df.columns:
                df = df.rename(columns={'Text': 'text', 'Label': 'label'})
        else:
            print(f"Unsupported file format: {ext}. Only .json and .csv are supported.")
            return pd.DataFrame(columns=['text', 'label'])
        
        if 'text' not in df.columns or 'label' not in df.columns:
            print(f"Error: Required columns 'text' and 'label' not found in {file_path}.")
            print(f"Actual columns: {df.columns.tolist()}")
            return pd.DataFrame(columns=['text', 'label'])
            
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return pd.DataFrame(columns=['text', 'label'])

def load_local_dataset(dataset_path: str):
    df = load_data_from_file(dataset_path)
    print(f"Loaded local dataset with {len(df)} samples from {dataset_path}.")
    if not df.empty:
        print("Local dataset label distribution:\n", df['label'].value_counts())
    return df

@lru_cache(maxsize=1)
def load_additional_huggingface_datasets():
    print("\n--- Loading additional Hugging Face datasets ---")
    all_hf_data = []

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

    engineering_mmlu_subjects = {
        "electrical_engineering": "Engineering",
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

    try:
        print("Loading lamm-mit/MechanicsMaterials dataset for Mechanical Engineering...")
        mech_mat_dataset = load_dataset("lamm-mit/MechanicsMaterials", split="train")
        mech_mat_df = mech_mat_dataset.to_pandas()
        if 'question' in mech_mat_df.columns and 'answer' in mech_mat_df.columns:
            mech_mat_df['text'] = mech_mat_df['question'] + ' ' + mech_mat_df['answer']
            mech_mat_df['label'] = "Engineering"
            all_hf_data.append(mech_mat_df[['text', 'label']])
            print(f"  Loaded {len(mech_mat_df)} samples for Mechanical Engineering.")
        else:
            print("  'question' or 'answer' column not found. Skipping lamm-mit/MechanicsMaterials.")
    except Exception as e:
        print(f"  Error loading lamm-mit/MechanicsMaterials: {str(e)}")

    try:
        print("Loading GainEnergy/oilandgas-engineering-dataset for Industrial Engineering...")
        industrial_dataset = load_dataset("GainEnergy/oilandgas-engineering-dataset", split="train")
        industrial_df = industrial_dataset.to_pandas()
        if 'text' in industrial_df.columns: 
            industrial_df['text'] = industrial_df['text']
            industrial_df['label'] = "Engineering"
            all_hf_data.append(industrial_df[['text', 'label']])
            print(f"  Loaded {len(industrial_df)} samples for Industrial Engineering.")
        else:
            print("  'text' column not found. Skipping GainEnergy/oilandgas-engineering-dataset.")
    except Exception as e:
        print(f"  Error loading GainEnergy/oilandgas-engineering-dataset: {str(e)}")
    
    clean_hf_data = []
    for df_item in all_hf_data:
        if isinstance(df_item, pd.DataFrame) and 'text' in df_item.columns and 'label' in df_item.columns:
            clean_hf_data.append(df_item)
        else:
            print(f"Warning: A DataFrame loaded from Hugging Face is missing 'text' or 'label' columns and will be skipped. Columns found: {df_item.columns.tolist() if isinstance(df_item, pd.DataFrame) else 'Not a DataFrame'}")
    all_hf_data = clean_hf_data


    if all_hf_data:
        combined_df = pd.concat(all_hf_data, ignore_index=True)
        print(f"\nTotal samples loaded: {len(combined_df)}")
        print("Label distribution:\n", combined_df['label'].value_counts())
        return combined_df
    
    print("No additional datasets loaded - using local data only")
    return pd.DataFrame(columns=['text', 'label'])

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

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    tokenized_results = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
    return tokenized_results

def prepare_dataset(df: pd.DataFrame, max_samples_per_class=None):
    original_rows = len(df)
    df.dropna(subset=['text', 'label'], inplace=True)
    if len(df) < original_rows:
        print(f"Dropped {original_rows - len(df)} rows with NaN values in 'text' or 'label'.")

    initial_row_count = len(df)
    df = df[df['text'].apply(lambda x: isinstance(x, str) and len(x.strip()) >= 10)]
    if len(df) < initial_row_count:
        print(f"Removed {initial_row_count - len(df)} rows with empty or too short 'text' content.")

    label_encoder = LabelEncoder()
    label_encoder.fit(df['label']) 
    df['encoded_label'] = label_encoder.transform(df['label'])
    
    if max_samples_per_class:
        sampled_dfs = []
        for label_name, group_df in df.groupby('label'):
            sampled_dfs.append(group_df.sample(min(len(group_df), max_samples_per_class), random_state=42))
        df = pd.concat(sampled_dfs).reset_index(drop=True)
        print(f"Applied soft-capping: Max {max_samples_per_class} samples per class.")
    
    class_counts = df['label'].value_counts()
    weights = []
    for cls in label_encoder.classes_:
        count = class_counts.get(cls, 0)
        weights.append(len(df) / (len(label_encoder.classes_) * count) if count > 0 else 0.0)
    
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)
    if class_weights_tensor.sum() > 0:
        class_weights_tensor = class_weights_tensor / class_weights_tensor.sum() * len(label_encoder.classes_)
    
    print("Calculated class weights:", class_weights_tensor)

    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['encoded_label'])

    train_dataset = Dataset.from_pandas(train_df).map(
        tokenize_function, batched=True, remove_columns=["__index_level_0__", "text", "label"]
    ).rename_column("encoded_label", "labels")

    eval_dataset = Dataset.from_pandas(eval_df).map(
        tokenize_function, batched=True, remove_columns=["__index_level_0__", "text", "label"]
    ).rename_column("encoded_label", "labels")

    train_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
    eval_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])

    return train_dataset, eval_dataset, label_encoder, df, class_weights_tensor

def train_model(train_dataset, eval_dataset, label_encoder, num_epochs: int = 3, learning_rate: float = 5e-5, output_dir: str = "./results", class_weights_tensor=None):
    print("\nTraining the model...")

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(label_encoder.classes_),
        id2label={i: label for i, label in enumerate(label_encoder.classes_)},
        label2id={label: i for i, label in enumerate(label_encoder.classes_)}
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    metric = evaluate.load("f1") 

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=1)
        f1 = metric.compute(predictions=predictions, references=labels, average='weighted')['f1']
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(labels, predictions, average='weighted', zero_division=0)
        return {"f1": f1, "accuracy": accuracy, "precision": precision, "recall": recall}

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=3e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        lr_scheduler_type="cosine",
        logging_dir='./logs',
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        fp16=True if torch.cuda.is_available() else False,
    )
        
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None): 
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            
            if class_weights_tensor is not None:
                loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor.to(logits.device))
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
            
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
        
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    return model, trainer, tokenizer, label_encoder

def generate_dynamic_keywords_from_dataset(dataset_df, label_encoder):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    generic_blacklist = {"explain", "define", "what", "how", "question", "answer", "problem", "solution"}

    dynamic_keywords = {}
    for label_id, label_name in enumerate(label_encoder.classes_):
        subject_texts = dataset_df[dataset_df['encoded_label'] == label_id]['text'].tolist()
        words = re.findall(r'\b[a-z]{2,}\b', " ".join(subject_texts).lower())
        
        filtered_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and w not in generic_blacklist]
        word_counts = Counter(filtered_words)
        
        if word_counts:
            threshold = max(1, int(word_counts.most_common(1)[0][1] * 0.01)) 
            subject_keywords = [w for w, cnt in word_counts.items() if cnt >= threshold]
        else:
            subject_keywords = []

        dynamic_keywords[label_name] = list(set(subject_keywords))

    return dynamic_keywords

def load_multiple_models(label_encoder, device):
    print("\n--- Loading additional models for ensembling ---")
    models = {}
    tokenizers = {}
    
    num_labels = len(label_encoder.classes_)
    id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
    label2id = {label: i for i, label in enumerate(label_encoder.classes_)}

    try:
        print("Loading SciBERT...")
        tokenizers["scibert"] = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        models["scibert"] = BertForSequenceClassification.from_pretrained(
            'allenai/scibert_scivocab_uncased',
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        ).to(device).eval()
        print("SciBERT loaded.")
    except Exception as e:
        print(f"Error loading SciBERT: {e}")
        models["scibert"] = None
        tokenizers["scibert"] = None

    try:
        print("Loading RoBERTa...")
        tokenizers["roberta"] = RobertaTokenizer.from_pretrained('roberta-base')
        models["roberta"] = RobertaForSequenceClassification.from_pretrained(
            'roberta-base',
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        ).to(device).eval()
        print("RoBERTa loaded.")
    except Exception as e:
        print(f"Error loading RoBERTa: {e}")
        models["roberta"] = None
        tokenizers["roberta"] = None
        
    return models, tokenizers

def evaluate_single_model(model, tokenizer, eval_dataset, device):
    model.eval()
    all_preds = []
    all_labels = []

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=16, shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer)
    )

    with torch.no_grad():
        for batch in eval_dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return f1


def predict(text: str, fine_tuned_model, fine_tuned_tokenizer, label_encoder, dynamic_keywords={}, ensemble_models=None, ensemble_tokenizers=None, ensemble_weights=None):
    try:
        text = str(text).strip()
        if not text:
            return "Input cannot be empty"
        if len(text) < 10:
            return "Input too short"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        combined_probs = None
        
        all_models = {"fine_tuned_bert": fine_tuned_model}
        all_tokenizers = {"fine_tuned_bert": fine_tuned_tokenizer}
        if ensemble_models and ensemble_tokenizers:
            for name, model in ensemble_models.items():
                if model is not None:
                    all_models[name] = model
                    all_tokenizers[name] = ensemble_tokenizers[name]

        if ensemble_weights is None:
            num_active_models = len(all_models)
            if num_active_models == 0:
                print("No active models for ensemble. Falling back to keyword matching.")
            else:
                default_weight = 1.0 / num_active_models
                ensemble_weights = {name: default_weight for name in all_models.keys()}
        
        if all_models and ensemble_weights:
            print(f"Using ensemble with weights: {ensemble_weights}")
            for name, model in all_models.items():
                if model is None:
                    continue
                
                tokenizer_current = all_tokenizers[name]
                inputs = tokenizer_current(text, return_tensors="pt", truncation=True, max_length=512)
                
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    model.eval()
                    outputs = model(**inputs)
                
                if outputs.logits.shape[1] != len(label_encoder.classes_):
                    print(f"Warning: Model {name} outputs {outputs.logits.shape[1]} labels, but expected {len(label_encoder.classes_)}. Skipping this model in ensemble.")
                    continue

                probs = torch.softmax(outputs.logits, dim=-1) * ensemble_weights.get(name, 0.0)
                combined_probs = probs if combined_probs is None else combined_probs + probs
            
            if combined_probs is not None:
                pred_id = torch.argmax(combined_probs, dim=-1).item()
                print(f"(Ensemble prediction: {label_encoder.classes_[pred_id]})")
                return label_encoder.classes_[pred_id]
        
        print("Ensemble prediction failed or not available. Falling back to keyword matching.")
        lemmatizer = WordNetLemmatizer()
        processed_text = " ".join(
            [lemmatizer.lemmatize(word) for word in re.findall(r'\b[a-z]{2,}\b', text.lower())]
        )

        keyword_scores = {label: 0 for label in dynamic_keywords}
        for label, keywords in dynamic_keywords.items():
            for keyword in keywords:
                if keyword in processed_text:
                    keyword_scores[label] += 1

        if keyword_scores and any(score > 0 for score in keyword_scores.values()):
            best_label = max(keyword_scores.items(), key=lambda x: x[1])[0]
            if keyword_scores[best_label] >= 2:
                print(f"(Keyword fallback match: {best_label})")
                return best_label
        
        return "Classification failed: No confident prediction from ensemble or keywords."

    except Exception as e:
        print(f"Prediction error: {e}")
        return "Classification failed"

def save_model(model, tokenizer, label_encoder, save_path: str):
    print(f"\nSaving model to {save_path}...")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    label_encoder_path = os.path.join(save_path, "label_encoder.json")
    with open(label_encoder_path, 'w', encoding='utf-8') as f:
        json.dump(label_encoder.classes_.tolist(), f)
    print("Model, tokenizer, and label encoder saved.")

def load_model(load_path: str):
    print(f"\nLoading model from {load_path}...")
    try:
        tokenizer_loaded = BertTokenizer.from_pretrained(load_path)
        
        label_encoder_path = os.path.join(load_path, "label_encoder.json")
        with open(label_encoder_path, 'r', encoding='utf-8') as f:
            classes = json.load(f)
        label_encoder_loaded = LabelEncoder()
        label_encoder_loaded.classes_ = np.array(classes)
        
        model_loaded = BertForSequenceClassification.from_pretrained(
            load_path,
            num_labels=len(label_encoder_loaded.classes_),
            id2label={i: label for i, label in enumerate(label_encoder_loaded.classes_)},
            label2id={label: i for i, label in enumerate(label_encoder_loaded.classes_)}
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_loaded.to(device)
        model_loaded.eval()
        print("Model, tokenizer, and label encoder loaded successfully.")
        return model_loaded, tokenizer_loaded, label_encoder_loaded
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

def combine_and_limit_datasets(local_df, hf_df, english_limit=2000, other_limit=3000, local_priority_for_other=1500):
    final_df_parts = []
    all_labels = pd.concat([local_df['label'], hf_df['label']]).unique()

    for label in all_labels:
        local_samples = local_df[local_df['label'] == label]
        hf_samples = hf_df[hf_df['label'] == label]

        if label == "English":
            combined_english = pd.concat([local_samples, hf_samples], ignore_index=True).drop_duplicates(subset=['text'])
            num_to_sample = min(len(combined_english), english_limit)
            if num_to_sample > 0:
                final_df_parts.append(combined_english.sample(n=num_to_sample, random_state=42))
            print(f"  Processed '{label}': Sampled {num_to_sample} samples (max {english_limit}).")
        else:
            current_samples_for_label = []

            num_from_local = min(len(local_samples), local_priority_for_other, other_limit)
            if num_from_local > 0:
                current_samples_for_label.append(local_samples.sample(n=num_from_local, random_state=42))
                print(f"  Processed '{label}': Took {num_from_local} from local.")

            remaining_needed = other_limit - num_from_local
            
            if remaining_needed > 0:
                hf_eligible_for_supplement = hf_samples[~hf_samples['text'].isin(local_samples['text'])]
                num_from_hf = min(len(hf_eligible_for_supplement), remaining_needed)
                if num_from_hf > 0:
                    current_samples_for_label.append(hf_eligible_for_supplement.sample(n=num_from_hf, random_state=42))
                    print(f"  Processed '{label}': Took {num_from_hf} from HF to supplement.")
            
            if current_samples_for_label:
                combined_for_label = pd.concat(current_samples_for_label, ignore_index=True).drop_duplicates(subset=['text'])
                final_df_parts.append(combined_for_label.sample(min(len(combined_for_label), other_limit), random_state=42))
                print(f"  Final for '{label}': {len(final_df_parts[-1])} samples (max {other_limit}).")

    if final_df_parts:
        return pd.concat(final_df_parts, ignore_index=True)
    else:
        return pd.DataFrame(columns=['text', 'label'])


def main():
    print("Starting classification model pipeline...")
    
    print("\n--- Loading local dataset ---")
    local_df = load_local_dataset(DATASET_PATH)
    print(f"Local dataset columns: {local_df.columns.tolist()}")
    if not local_df.empty:
        print(f"Local dataset sample:\n{local_df.head()}")
    else:
        print("Local dataset is empty.")
    
    print("\n--- Loading HuggingFace datasets ---")
    hf_df = load_additional_huggingface_datasets()
    print(f"HF dataset columns: {hf_df.columns.tolist() if not hf_df.empty else 'Empty'}")
    if not hf_df.empty:
        print(f"HF dataset sample:\n{hf_df.head()}")
    else:
        print("HuggingFace dataset is empty.")
        
    print("\n--- Combining and limiting datasets based on requirements ---")
    combined_df = combine_and_limit_datasets(
        local_df, 
        hf_df, 
        english_limit=2000, 
        other_limit=3000, 
        local_priority_for_other=1500
    )
    
    if combined_df.empty:
        print("No combined dataset loaded after sampling. Exiting.")
        return

    print(f"\nFinal combined dataset size: {len(combined_df)}")
    print("Final combined dataset columns:", combined_df.columns.tolist())
    print("Final combined dataset label distribution:\n", combined_df['label'].value_counts())

    try:
        train_dataset, eval_dataset, label_encoder, processed_df, class_weights_tensor = prepare_dataset(
            combined_df, max_samples_per_class=None 
        )
    except Exception as e:
        print(f"Error in prepare_dataset: {e}")
        print("Debug info - combined_df columns:", combined_df.columns.tolist())
        print("combined_df sample:\n", combined_df.head())
        return

    dynamic_keywords = generate_dynamic_keywords_from_dataset(processed_df, label_encoder)
    print("\nGenerated dynamic keywords (sample for 'Computer Science'):")
    print(dynamic_keywords.get('Computer Science', [])[:10])

    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_output_dir = os.path.join(MODEL_SAVE_DIR, f"bert_classifier_{current_time_str}")
    
    model, trainer, tokenizer_trained, label_encoder_trained = train_model(
        train_dataset, eval_dataset, label_encoder, num_epochs=3, learning_rate=5e-5,
        output_dir=model_output_dir, class_weights_tensor=class_weights_tensor
    )

    if model and tokenizer_trained and label_encoder_trained:
        save_model(model, tokenizer_trained, label_encoder_trained, model_output_dir)
    
    model_loaded, tokenizer_loaded, label_encoder_loaded = load_model(model_output_dir)

    if model_loaded is None:
        print("Failed to load fine-tuned model for inference. Exiting.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble_models, ensemble_tokenizers = load_multiple_models(label_encoder_loaded, device)
    
    print("\n--- Calculating dynamic ensemble weights ---")
    model_performance = {}
    
    if model_loaded:
        print(f"Evaluating fine-tuned BERT...")
        f1_bert = evaluate_single_model(model_loaded, tokenizer_loaded, eval_dataset, device)
        model_performance["fine_tuned_bert"] = f1_bert
        print(f"  Fine-tuned BERT F1-score: {f1_bert:.4f}")

    for name, ensemble_model in ensemble_models.items():
        if ensemble_model:
            print(f"Evaluating {name}...")
            f1_score_ensemble = evaluate_single_model(ensemble_model, ensemble_tokenizers[name], eval_dataset, device)
            model_performance[name] = f1_score_ensemble
            print(f"  {name} F1-score: {f1_score_ensemble:.4f}")

    total_f1 = sum(model_performance.values())
    ensemble_weights = {}
    if total_f1 > 0:
        for model_name, f1_score_val in model_performance.items():
            ensemble_weights[model_name] = f1_score_val / total_f1
    else:
        print("Warning: Total F1-score is zero, falling back to equal weights for ensemble.")
        num_active_models = len(model_performance)
        if num_active_models > 0:
            equal_weight = 1.0 / num_active_models
            ensemble_weights = {name: equal_weight for name in model_performance.keys()}
        else:
            print("No active models to assign weights to.")


    print(f"\nDynamically calculated ensemble weights: {ensemble_weights}")

    print("\n--- Enter text or file path for classification (type 'exit' to quit) ---")
    while True:
        user_input = input("Input: ").strip()
        if user_input.lower() == 'exit':
            break

        if os.path.exists(user_input):
            print(f"\n--- Classifying content from file: '{user_input}' ---")
            try:
                with open(user_input, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                if file_content.strip():
                    classified_label = predict(
                        file_content, 
                        model_loaded, 
                        tokenizer_loaded, 
                        label_encoder_loaded, 
                        dynamic_keywords,
                        ensemble_models,
                        ensemble_tokenizers,
                        ensemble_weights
                    )
                    print(f"-> Classified File Content as: {classified_label}")
                else:
                    print("File is empty or contains no readable text.")
            except UnicodeDecodeError:
                print(f"Error reading or processing file '{user_input}': 'utf-8' codec can't decode. This might not be a text file (e.g., PDF, image). Only plain text files are supported for direct reading.")
            except Exception as e:
                print(f"Error reading or processing file '{user_input}': {e}")
        else:
            classified_label = predict(
                user_input, 
                model_loaded, 
                tokenizer_loaded, 
                label_encoder_loaded, 
                dynamic_keywords,
                ensemble_models,
                ensemble_tokenizers,
                ensemble_weights
            )
            print(f"-> Classified Text Input as: {classified_label}")

if __name__ == "__main__":
    main()
