# ========== STEP 1: IMPORT LIBRARIES ==========
import json
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset, load_dataset, concatenate_datasets # UPDATED: Added load_dataset, concatenate_datasets
import evaluate # Import the evaluate library
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from transformers import pipeline # Import pipeline for easier inference
import os
import shutil # For removing the new_samples.json after merging
import transformers
import inspect
print("Transformers version:", transformers.__version__)
print("TrainingArguments loaded from:", inspect.getfile(transformers.TrainingArguments))


# For dynamic keyword generation
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import WordNetLemmatizer # UPDATED: Import for lemmatization

# For dynamic filename generation
from datetime import datetime

# Download NLTK stopwords if not already downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    print("NLTK 'stopwords' downloaded.")

# UPDATED: Download NLTK resources for lemmatization if not already downloaded
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    print("NLTK 'wordnet' downloaded.")
try:
    nltk.data.find('corpora/omw-1.4')
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
# IMPORTANT: Adjust these paths to your actual directories
MODEL_SAVE_DIR = r"E:\Ahmed Sameh Work\Projects\GAIAthon 25\Classification Model\models"
DATASET_PATH = r"E:\Ahmed Sameh Work\Projects\GAIAthon 25\Classification Model\datasets\dataset.json"

# ========== STEP 2: LOAD AND PREPARE DATA ==========
def load_local_dataset(dataset_path: str): # RENAMED: from load_dataset to load_local_dataset
    """Loads the dataset from a JSON file."""
    if not os.path.exists(dataset_path):
        # UPDATED: If local dataset not found, create an empty one for combining
        print(f"Warning: Dataset file not found at: {dataset_path}. Starting with empty local dataset.")
        return pd.DataFrame(columns=['text', 'label'])
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    print(f"Loaded local dataset with {len(df)} samples.")
    print("Local dataset label distribution:\n", df['label'].value_counts())
    return df

# UPDATED: New function to load additional Hugging Face datasets with split fix
def load_additional_huggingface_datasets():
    """Loads and preprocesses Hugging Face datasets with proper error handling."""
    print("\n--- Loading additional Hugging Face datasets ---")
    all_hf_data = []

    # 1. Mathematics: math_qa
    try:
        print("Loading math_qa dataset...")
        math_dataset = load_dataset("math_qa", trust_remote_code=True)
        # Fix: Access correct split and check for 'Problem' column directly in the DataFrame
        if 'train' in math_dataset:
            math_df = pd.DataFrame(math_dataset['train'])  # Access correct split
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

    # 2. English: wikihow or fallback to story_cloze
    try:
        print("Loading English dataset from GEM/wiki_how...")
        wikihow_dataset = load_dataset("GEM/wiki_how", split="train")
        wikihow_df = wikihow_dataset.to_pandas()
        wikihow_df = wikihow_df[['source']].rename(columns={'source': 'text'})
        wikihow_df['label'] = "English"
        wikihow_df = wikihow_df[wikihow_df['text'].str.len() > 50].sample(n=1000, random_state=42)
        all_hf_data.append(wikihow_df)
        print(f"  Loaded {len(wikihow_df)} samples for English.")
    except Exception as e:
        print(f"  Error loading GEM/wiki_how: {str(e)}")

        # Fallback: story_cloze
        try:
            print("Falling back to story_cloze dataset...")
            story_dataset = load_dataset("story_cloze", "2016", split="validation")
            story_df = story_dataset.to_pandas()
            story_df['text'] = story_df[['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']].agg(' '.join, axis=1)
            story_df = story_df[['text']]
            story_df['label'] = "English"
            story_df = story_df[story_df['text'].str.len() > 50].sample(n=1000, random_state=42)
            all_hf_data.append(story_df)
            print(f"  Loaded {len(story_df)} fallback samples for English.")
        except Exception as e2:
            print(f"  English fallback also failed: {str(e2)}")

    # 3. Biology: pubmed_qa
    try:
        print("Loading pubmed_qa dataset...")
        pubmed_dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")  # Specify config and split
        pubmed_df = pubmed_dataset.to_pandas()
        pubmed_df['text'] = pubmed_df['question'] + " " + pubmed_df['context'].apply(lambda x: " ".join(x))
        pubmed_df = pubmed_df[['text']]
        pubmed_df['label'] = "Biology"
        all_hf_data.append(pubmed_df)
        print(f"  Loaded {len(pubmed_df)} samples for Biology.")
    except Exception as e:
        print(f"  Error loading pubmed_qa: {str(e)}")

    # 4. Physics: ai2_arc
    try:
        print("Loading ai2_arc dataset...")
        arc_dataset = load_dataset("ai2_arc", "ARC-Challenge", split="train")
        arc_df = arc_dataset.to_pandas()
        # Fix: Safest way to handle the choices field
        arc_df['text'] = arc_df.apply(
            lambda row: row['question'] + " " +
            " ".join([c['text'] for c in row['choices']] if isinstance(row['choices'], list) else []),
            axis=1
        )
        arc_df = arc_df[['text']]
        arc_df['label'] = "Physics"
        arc_df = arc_df[arc_df['text'].str.len() > 30] # Filter out short texts
        all_hf_data.append(arc_df)
        print(f"  Loaded {len(arc_df)} samples for Physics.")
    except Exception as e:
        print(f"  Error loading ai2_arc: {str(e)}")

    # 5. Chemistry: use alternative
    try:
        print("Loading chemistry dataset (alternative bigscience/P3)...")
        chem_dataset = load_dataset("bigscience/P3", "chemistry_classification", split="train")
        chem_df = chem_dataset.to_pandas()[['inputs']]
        chem_df = chem_df.rename(columns={'inputs': 'text'})
        chem_df['label'] = "Chemistry"
        chem_df = chem_df[chem_df['text'].str.len() > 20]
        chem_df = chem_df.sample(n=500, random_state=42)
        all_hf_data.append(chem_df)
        print(f"  Loaded {len(chem_df)} samples for Chemistry.")
    except Exception as e:
        print(f"  Error loading bigscience/P3 chemistry dataset: {str(e)}")

    # Final return
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
        
        # Load label map
        with open(os.path.join(model_path, "label_map.json"), "r") as f:
            label_map = json.load(f)
        
        # Recreate label_encoder
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
# ========== STEP 5: RETRAIN WITH NEW DATA ==========
def retrain_with_new_data(current_dataset_path: str, new_data_base_dir: str):
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
                        # Add timestamp to each new sample
                        for item in new_data:
                            item['timestamp'] = datetime.now().isoformat()
                        new_samples_data.extend(new_data)
                    else: # Handle single object if not a list
                        new_data['timestamp'] = datetime.now().isoformat()
                        new_samples_data.append(new_data)
                print(f"Found and loaded new samples from: {filename}")
                new_samples_found = True
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {filename}: {e}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    if new_samples_found and new_samples_data:
        print(f"Total new samples collected: {len(new_samples_data)}")

        # Load existing dataset
        existing_df = load_local_dataset(current_dataset_path) # UPDATED: Call load_local_dataset
        
        # Convert new samples to DataFrame
        new_df = pd.DataFrame(new_samples_data)
        
        # Check for and handle potential duplicate labels (e.g., "Math" and "Mathematics")
        # Ensure all 'Math' labels are converted to 'Mathematics' for consistency
        new_df['label'] = new_df['label'].replace('Math', 'Mathematics')
        existing_df['label'] = existing_df['label'].replace('Math', 'Mathematics')

        # Merge new samples with existing dataset
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Remove duplicates based on 'text'
        initial_count = len(updated_df)
        updated_df.drop_duplicates(subset=['text'], inplace=True) # Changed from ['text', 'label'] to ['text']
        duplicates_removed = initial_count - len(updated_df)
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate entries after merging.")

        # Save updated dataset
        with open(current_dataset_path, 'w', encoding='utf-8') as f:
            json.dump(updated_df.to_dict('records'), f, indent=2)
        print(f"Dataset updated with {len(new_df)} new samples (after deduplication).")

        # Prepare for retraining
        labels = updated_df['label'].tolist()
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)

        # Prepare Hugging Face Dataset
        dataset_dict = {'text': updated_df['text'].tolist(), 'label': encoded_labels.tolist()}
        hf_dataset = Dataset.from_dict(dataset_dict)

        # Split into training and evaluation sets
        if len(label_encoder.classes_) > 1 and all(count > 1 for count in updated_df['label'].value_counts()):
             train_texts, val_texts, train_labels, val_labels = train_test_split(
                hf_dataset['text'], hf_dataset['label'], test_size=0.1, random_state=42, stratify=hf_dataset['label']
            )
        else: # Fallback for very small datasets or single-class datasets
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                hf_dataset['text'], hf_dataset['label'], test_size=0.1, random_state=42
            )

        # Create Hugging Face Dataset objects for training
        train_hf_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
        eval_hf_dataset = Dataset.from_dict({'text': val_texts, 'label': val_labels})

        # Tokenize datasets
        tokenized_train_dataset = train_hf_dataset.map(tokenize_function, batched=True)
        tokenized_eval_dataset = eval_hf_dataset.map(tokenize_function, batched=True)

        # Retrain the model
        model, trainer, label_encoder = train_model(tokenized_train_dataset, tokenized_eval_dataset, label_encoder)
        
        # Clean up new_samples_*.json files
        for filename in os.listdir(new_data_base_dir):
            if filename.startswith("new_samples_") and filename.endswith(".json"):
                os.remove(os.path.join(new_data_base_dir, filename))
                print(f"Removed processed new samples file: {filename}")
        
        print("\nModel retrained and new samples processed.")
        return model, trainer, label_encoder
    else:
        print("No new samples files found to retrain.")
        # If no new samples, try to load existing model
        return load_model(MODEL_SAVE_DIR)


# UPDATED: Improved generate_keywords_from_dataset function
def generate_keywords_from_dataset(dataset: Dataset, label_encoder, top_n: int = 15, common_word_threshold: float = 0.005):
    """
    Generates subject-specific keywords from the dataset.
    Removes stopwords and common words across all categories.
    Applies lemmatization to keywords.

    Args:
        dataset (datasets.Dataset): The Hugging Face Dataset object.
        label_encoder (sklearn.preprocessing.LabelEncoder): The label encoder used for categories.
        top_n (int): The number of top keywords to select for each subject.
        common_word_threshold (float): Words appearing in more than this proportion of the total words
                                       across all categories will be excluded as generic.
                                       Increasing this value will filter out more common words.
    Returns:
        dict: A dictionary where keys are subject labels and values are lists of their top keywords.
    """
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    all_words = []
    category_words = {label: [] for label in label_encoder.classes_}

    # Collect all words and category-specific words
    for example in dataset: # Iterate directly over the dataset
        text = example['text']
        # The 'label' in the dataset is already integer-encoded, so we need to inverse transform
        label_id = example['label'] # Get the integer label
        label = label_encoder.inverse_transform([label_id])[0] # Decode numeric label back to string

        # Simple tokenization and normalization
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Lemmatize and filter stopwords
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
        
        all_words.extend(lemmatized_words)
        category_words[label].extend(lemmatized_words)

    # Count overall word frequencies to identify very common words across all subjects
    overall_word_counts = Counter(all_words)
    total_words = sum(overall_word_counts.values())

    # Identify words that are too common across the entire dataset
    # These words are considered generic academic verbs/terms.
    # We explicitly add the problematic generic verbs and nouns from your analysis here
    # to ensure they are always excluded if not caught by the threshold.
    explicit_generic_terms = {
        "describe", "explain", "outline", "define", "state", "identify", "compare",
        "number", "function", "cell", "energy", "solution", "power", "base",
        "structure", "system", "process", "information", "table", "write", "equation", "form"
    }

    common_words_to_exclude = {
        word for word, count in overall_word_counts.items()
        if count / total_words > common_word_threshold
    }
    # Combine automatically identified common words with explicitly defined generic terms
    common_words_to_exclude.update(explicit_generic_terms)


    dynamic_subject_keywords = {}
    for label, words in category_words.items():
        # Count word frequencies for the current category
        category_word_counts = Counter(words)
        
        # Filter out common words, stopwords, and explicitly generic terms
        filtered_words = []
        for word, count in category_word_counts.most_common():
            if word not in common_words_to_exclude and word not in stop_words:
                filtered_words.append(word)
        
        # Select the top_n most frequent, subject-specific keywords
        dynamic_subject_keywords[label] = filtered_words[:top_n]

    return dynamic_subject_keywords


# ========== STEP 6: CLASSIFY NEW TEXT (HYBRID APPROACH) ==========
def classify_with_unsupervised(text: str, dynamic_subject_keywords: dict, classifier_pipeline, threshold: float = 0.5):
    """
    Classifies text using a hybrid approach:
    1. Pattern-based classification using dynamic keywords.
    2. Fallback to BERT-based classification if no strong pattern match.
    """
    print(f"\nAttempting to classify: '{text}'")

    # 1. Pattern-based classification
    text_lower = text.lower()
    lemmatizer = WordNetLemmatizer()
    text_words = [lemmatizer.lemmatize(word) for word in re.findall(r'\b\w+\b', text_lower) if word.isalpha()]

    pattern_scores = {label: 0 for label in dynamic_subject_keywords}
    
    # Calculate keyword match scores
    for label, keywords in dynamic_subject_keywords.items():
        for keyword in keywords:
            if keyword in text_words: # Check for exact lemmatized keyword match
                pattern_scores[label] += 1
    
    # Normalize scores (optional, but good for comparison)
    total_matches = sum(pattern_scores.values())
    normalized_pattern_scores = {label: score / total_matches if total_matches > 0 else 0 for label, score in pattern_scores.items()}

    best_pattern_label = None
    max_pattern_score = -1

    for label, score in normalized_pattern_scores.items():
        if score > max_pattern_score:
            max_pattern_score = score
            best_pattern_label = label

    if best_pattern_label and max_pattern_score > 0: # Check if any keyword matched
        print(f"  -> Pattern-based classification: {best_pattern_label} (Score: {max_pattern_score:.2f})")
        # You can set a threshold here if you want to skip BERT for very confident pattern matches
        if max_pattern_score > threshold: # Example threshold
            return best_pattern_label
    else:
        print("  -> No strong pattern match found by keyword analysis.")

    # 2. Fallback to BERT-based classification
    print("  -> Falling back to BERT-based classification...")
    try:
        # The pipeline expects a list of strings
        bert_results = classifier_pipeline(text)
        
        # Get the predicted label and score
        predicted_label = bert_results[0]['label']
        bert_score = bert_results[0]['score']
        
        print(f"  -> BERT-based classification: {predicted_label} (Confidence: {bert_score:.4f})")
        return predicted_label
    except Exception as e:
        print(f"  -> Error during BERT classification: {e}")
        return "Classification Failed"


# ========== STEP 7: MAIN EXECUTION BLOCK ==========
# ... (existing imports and functions remain unchanged above this point) ...

# ========== STEP 7: MAIN EXECUTION BLOCK ==========
if __name__ == "__main__":
    current_dataset_path = DATASET_PATH
    new_samples_base_dir = os.path.dirname(DATASET_PATH) # New samples will be in the same directory as dataset.json

    # --- Initial Model Load/Train ---
    # This block handles the very first load or training when the script starts
    initial_df = load_local_dataset(current_dataset_path)
    hf_df = load_additional_huggingface_datasets()

    if not hf_df.empty:
        initial_df['label'] = initial_df['label'].replace('Math', 'Mathematics')
        hf_df['label'] = hf_df['label'].replace('Math', 'Mathematics')
        combined_df = pd.concat([initial_df, hf_df], ignore_index=True)
        initial_combined_count = len(combined_df)
        combined_df.drop_duplicates(subset=['text'], inplace=True)
        duplicates_removed_overall = initial_combined_count - len(combined_df)
        if duplicates_removed_overall > 0:
            print(f"Removed {duplicates_removed_overall} duplicate entries after combining local and HF datasets.")
        print(f"\nTotal dataset size after combining: {len(combined_df)} samples.")
        print("Combined label distribution:\n", combined_df['label'].value_counts())
        final_dataset_df = combined_df
    else:
        print("\nNo Hugging Face datasets added. Proceeding with only the local dataset.")
        final_dataset_df = initial_df
        final_dataset_df['label'] = final_dataset_df['label'].replace('Math', 'Mathematics')

    labels = final_dataset_df['label'].tolist()
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    raw_dataset = Dataset.from_dict({'text': final_dataset_df['text'].tolist(), 'label': encoded_labels.tolist()})

    if all(count >= 2 for count in final_dataset_df['label'].value_counts()):
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            raw_dataset['text'], raw_dataset['label'], test_size=0.1, random_state=42, stratify=raw_dataset['label']
        )
    else:
        print("Warning: Cannot use stratified split due to very small classes. Performing non-stratified split.")
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            raw_dataset['text'], raw_dataset['label'], test_size=0.1, random_state=42
        )

    train_hf_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    eval_hf_dataset = Dataset.from_dict({'text': val_texts, 'label': val_labels})
    tokenized_train_dataset = train_hf_dataset.map(tokenize_function, batched=True)
    tokenized_eval_dataset = eval_hf_dataset.map(tokenize_function, batched=True)

    model = None
    tokenizer_loaded = None
    label_encoder_loaded = None

    if os.path.exists(MODEL_SAVE_DIR) and os.path.exists(os.path.join(MODEL_SAVE_DIR, "label_map.json")):
        model, tokenizer_loaded, label_encoder_loaded = load_model(MODEL_SAVE_DIR)
        if model is None:
            print("Existing model could not be loaded. Retraining from scratch.")
            model, _, label_encoder = train_model(tokenized_train_dataset, tokenized_eval_dataset, label_encoder)
        else:
            print("Existing model loaded successfully.")
            tokenizer = tokenizer_loaded
            label_encoder = label_encoder_loaded
    else:
        print("No existing model found. Training a new model.")
        model, _, label_encoder = train_model(tokenized_train_dataset, tokenized_eval_dataset, label_encoder)

    # --- Initial Keyword Generation ---
    print("\nGenerating dynamic keywords from the current dataset...")
    updated_labels_encoded = label_encoder.transform(final_dataset_df['label'].tolist())
    updated_raw_dataset = Dataset.from_dict({'text': final_dataset_df['text'].tolist(), 'label': updated_labels_encoded.tolist()})
    dynamic_keywords = generate_keywords_from_dataset(updated_raw_dataset, label_encoder)
    print("\nDynamic keywords generated:")
    for subject, keywords in dynamic_keywords.items():
        print(f"  {subject}: {keywords}")

    # Prepare Hugging Face pipeline for inference
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


    # Interactive classification loop with automatic retraining
    if classifier_pipeline:
        print("\n=== Testing the trained classifier with hybrid approach ===")
        while True:
            # --- Check for new samples and retrain automatically ---
            # Call retrain_with_new_data before each classification attempt
            # This function now returns the (potentially new) model, trainer, and label_encoder
            # if retraining occurred, or the existing ones if no new data was found.
            # We explicitly pass the current dataset path and new samples base directory.
            # This needs to be slightly modified to properly handle the return values and
            # update the global model, tokenizer, label_encoder, and pipeline.
            
            # The retrain_with_new_data function as written already handles loading the model.
            # We just need to call it and ensure the global objects are updated if retraining happened.
            
            # Before classifying, check if new data exists and retrain if necessary.
            # The `retrain_with_new_data` function already handles detecting, merging, and retraining.
            # It returns the updated model, trainer, and label_encoder.
            # We need to re-initialize the pipeline if the model changed.

            # We need a way to know if retrain_with_new_data actually retrained.
            # Let's modify retrain_with_new_data to return a flag.
            
            # Instead of modifying retrain_with_new_data's return, let's just re-load everything if new files exist.
            # This ensures we always use the latest model and keywords.

            # Check for new samples first
            new_samples_exist = any(f.startswith("new_samples_") and f.endswith(".json") for f in os.listdir(new_samples_base_dir))

            if new_samples_exist:
                print("\nNew samples detected! Initiating automatic retraining...")
                model, trainer_dummy, label_encoder = retrain_with_new_data(current_dataset_path, new_samples_base_dir)
                
                # After retraining, reload the full (potentially updated) dataset to regenerate keywords
                print("\nRe-generating dynamic keywords after automatic retraining...")
                updated_df_for_keywords = load_local_dataset(current_dataset_path)
                
                # Ensure labels are consistent for keyword generation, re-encode with the (potentially updated) label_encoder
                updated_labels_encoded = label_encoder.transform(updated_df_for_keywords['label'].tolist())
                updated_raw_dataset = Dataset.from_dict({'text': updated_df_for_keywords['text'].tolist(), 'label': updated_labels_encoded.tolist()})
                
                dynamic_keywords = generate_keywords_from_dataset(updated_raw_dataset, label_encoder)
                print("\nDynamic keywords regenerated:")
                for subject, keywords in dynamic_keywords.items():
                    print(f"  {subject}: {keywords}")
                
                # Re-initialize the pipeline with the newly trained model
                try:
                    classifier_pipeline = pipeline(
                        "text-classification",
                        model=MODEL_SAVE_DIR,
                        tokenizer=MODEL_SAVE_DIR,
                        device=0 if torch.cuda.is_available() else -1
                    )
                    print("\nHugging Face classification pipeline reloaded after retraining.")
                except Exception as e:
                    print(f"\nCould not reload pipeline after retraining: {e}")
                    classifier_pipeline = None # Disable pipeline if reload fails

            if not classifier_pipeline:
                print("Classifier pipeline is not available. Please check model loading issues.")
                break # Exit if pipeline isn't working

            user_input = input("Enter text to classify (or a file path to classify its content) (or 'quit' to exit): ")
            
            if user_input.lower() == 'quit':
                break

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