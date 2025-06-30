# ========== STEP 1: IMPORT LIBRARIES ==========
import json
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset
import evaluate # Import the evaluate library
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import os
import shutil # For removing the new_samples.json after merging

# Set a random seed for reproducibility
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# Global variables for model and tokenizer (will be loaded after training or at startup)
tokenizer = None
model = None
le_mapping = None # Stores the label ID to label name mapping
le = None # LabelEncoder instance

# ========== STEP 2: LOAD AND PREPARE DATA ==========
def load_and_prepare_data(data_path=r"E:\Ahmed Sameh Work\Projects\GAIAthon 25\Classification Model\datasets\dataset.json"):
    """
    Loads data from a JSON file, preprocesses it, and splits into train/test sets.
    """
    global le, le_mapping # Declare global to assign to them

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{data_path}' not found. Please create or specify the correct path.")
        print("Example content for your_dataset.json:")
        print('[{"text": "This is about science.", "label": "Science"}, {"text": "Learn about history events.", "label": "History"}]')
        exit()

    df = pd.DataFrame(data)
    print("Sample of loaded data:")
    print(df.head())
    print(f"\nTotal samples: {len(df)}")
    print(f"Unique labels: {df['label'].nunique()} -> {df['label'].unique().tolist()}")

    # Ensure 'text' and 'label' columns exist
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("DataFrame must contain 'text' and 'label' columns.")

    # Encode labels to integers
    le = LabelEncoder()
    df['label_id'] = le.fit_transform(df['label'])
    num_labels = len(le.classes_)
    print(f"\nLabel mapping: {list(le.classes_)}")

    # Stratified train/test split to ensure balanced representation of labels
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['text'].tolist(), df['label_id'].tolist(), test_size=0.35, random_state=42, stratify=df['label_id']
    )
    print(f"\nTrain set size: {len(train_texts)}")
    print(f"Test set size: {len(test_texts)}")

    # Store label mapping
    le_mapping = {i: label for i, label in enumerate(le.classes_)}

    return train_texts, test_texts, train_labels, test_labels, num_labels

# ========== STEP 3: TOKENIZATION AND DATASET CREATION ==========
def create_datasets(train_texts, test_texts, train_labels, test_labels):
    """
    Tokenizes text and creates Hugging Face Dataset objects.
    """
    global tokenizer # Declare global to assign to it

    # Load pre-trained BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create Hugging Face Dataset objects
    train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    test_dataset = Dataset.from_dict({'text': test_texts, 'label': test_labels})

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=False)

    print("\nTokenizing training data...")
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    print("Tokenizing test data...")
    test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    return train_dataset, test_dataset

# ========== STEP 4: LOAD BERT MODEL ==========
def load_bert_model(num_labels):
    """
    Initializes BertForSequenceClassification model.
    """
    global model, le_mapping # Use global le_mapping

    print(f"\nLoading BERT model with {num_labels} classification heads...")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

    # Map integer labels back to their string names for better output during evaluation
    model.config.id2label = le_mapping
    model.config.label2id = {label: i for i, label in le_mapping.items()}
    print("Model loaded successfully.")
    return model

# ========== STEP 5: TRAINING ARGUMENTS ==========
# ========== STEP 5: TRAINING ARGUMENTS ==========
def get_training_arguments():
    """
    Configures and returns TrainingArguments.
    """
    return TrainingArguments(
        output_dir=r"E:\Ahmed Sameh Work\Projects\GAIAthon 25\Classification Model\outputs",
        eval_strategy="epoch",  # Changed from 'evaluation_strategy' to 'eval_strategy'
        eval_steps=500,
        save_strategy="epoch",
        save_steps=500,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_dir=r"E:\Ahmed Sameh Work\Projects\GAIAthon 25\Classification Model\outputs",
        logging_steps=500,
        report_to="tensorboard",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

# ========== STEP 6: METRICS ==========
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Calculate various metrics
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    precision_macro = precision_score(labels, predictions, average='macro', zero_division=0)
    recall_macro = recall_score(labels, predictions, average='macro', zero_division=0)

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro
    }
# ========== STEP 7: TRAINING FUNCTION ==========
def train_model(model, train_dataset, test_dataset, training_args):
    """
    Initiates the model training process.
    """
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    print("\nStarting model training...")
    trainer.train()
    print("\nTraining complete!")
    return trainer

# ========== STEP 8: EVALUATE ON TEST SET ==========
def evaluate_model(trainer, test_dataset):
    """
    Evaluates the trained model on the test set and prints a classification report.
    """
    print("\nEvaluating model on the test set...")
    preds = trainer.predict(test_dataset)
    y_pred = np.argmax(preds.predictions, axis=-1)

    # Convert test_dataset labels back to a list of integers
    actual_labels = [label.item() for label in test_dataset['label']]

    print("\n=== Final Classification Report ===")
    print(classification_report(actual_labels, y_pred, target_names=list(le.classes_), digits=4))

# ========== STEP 9: SAVE MODEL AND ARTIFACTS ==========
def save_model_artifacts(trainer, model_save_path=r"E:\Ahmed Sameh Work\Projects\GAIAthon 25\Classification Model\models"):
    """
    Saves the fine-tuned model, tokenizer, and label encoder mapping.
    """
    os.makedirs(model_save_path, exist_ok=True)
    print(f"\nSaving model, tokenizer, and label map to '{model_save_path}'...")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    # Save label mapping with integer keys
    int_mapping = {int(k): v for k, v in le_mapping.items()}  # Ensure keys are integers
    with open(os.path.join(model_save_path, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(int_mapping, f, indent=4)

    print("✅ Model, tokenizer, and label map saved successfully.")

# ========== PATTERN-BASED CLASSIFIER ==========
def pattern_based_classifier(text):
    """
    Classify text based on subject-specific keywords.
    Returns label if match found, else None.
    """
    text_lower = text.lower()
    subject_keywords = {
        "Biology": ["photosynthesis", "dna", "cell", "enzyme", "species", "ecosystem", "biology", "organism"],
        "Physics": ["force", "energy", "quantum", "velocity", "electron", "gravity", "physics", "relativity", "mechanics"],
        "Math": ["equation", "derivative", "matrix", "algebra", "geometry", "calculate", "calculus", "statistics", "math"],
        "Computer Science": ["algorithm", "neural network", "python", "database", "programming", "software", "machine learning", "computer science"],
        "Chemistry": ["molecule", "reaction", "acid", "chemical", "periodic table", "chemistry", "compound"],
        "Engineering": ["circuit", "mechanical", "stress", "fluid", "design", "robotics", "engineer", "structural"],
        "English": ["metaphor", "poetry", "literature", "Shakespeare", "grammar", "novel", "essay", "writing", "english"]
    }

    for label, keywords in subject_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            return label
    return None  # No pattern match

# ========== ENHANCED PREDICTION FUNCTION ==========
def classify_bert(text, model_path="bert_subject_classifier"):
    """
    Performs BERT-based classification with error handling.
    """
    global tokenizer, model, le_mapping

    # Load model if not already loaded
    if model is None or tokenizer is None or le_mapping is None:
        print(f"Loading model, tokenizer, and label map from '{model_path}' for inference...")
        try:
            tokenizer = BertTokenizer.from_pretrained(model_path)
            model = BertForSequenceClassification.from_pretrained(model_path)
            # Load label mapping with proper type conversion
            with open(os.path.join(model_path, "label_map.json"), "r", encoding="utf-8") as f:
                le_mapping = {int(k): v for k, v in json.load(f).items()}  # Convert keys to integers
        except Exception as e:
            print(f"Error loading model: {e}")
            return "Unknown"

    model.eval()

    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model.to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        pred_id = torch.argmax(outputs.logits, dim=1).item()
        
        # Handle cases where pred_id isn't in the mapping
        if pred_id in le_mapping:
            return le_mapping[pred_id]
        else:
            print(f"Warning: Predicted ID {pred_id} not found in label mapping. Available IDs: {list(le_mapping.keys())}")
            return "Unknown"
            
    except Exception as e:
        print(f"Error during classification: {e}")
        return "Unknown"
    

# ========== ENHANCED PREDICTION FUNCTION ==========
# (Keep classify_bert as it was in the previous successful response)
# ...

def classify_with_unsupervised(text, model_path="bert_subject_classifier", save_new_samples=True):
    """
    Hybrid classification: Patterns -> BERT -> Save new samples.
    """
    print(f"\nAttempting to classify: '{text}'")
    # Try pattern-based classification first
    pattern_label = pattern_based_classifier(text)
    if pattern_label:
        print(f"  -> Pattern-based classification: {pattern_label}")
        return pattern_label

    # Fall back to BERT
    bert_label = classify_bert(text, model_path)
    print(f"  -> BERT-based classification: {bert_label}")

    # Save new sample to dataset (if enabled)
    if save_new_samples:
        new_sample = {"text": text, "label": bert_label}
        new_samples_file = "new_samples.json"
        
        existing_data = []
        # Check if file exists and has content before trying to load
        if os.path.exists(new_samples_file) and os.path.getsize(new_samples_file) > 0:
            try:
                with open(new_samples_file, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: '{new_samples_file}' is corrupted or empty. Starting a new file.")
                existing_data = [] # Reset if corrupted

        existing_data.append(new_sample)

        # Write the entire updated list back to the file
        with open(new_samples_file, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2)

        print(f"  -> Saved new sample to '{new_samples_file}' for future retraining.")

    return bert_label

# ========== RETRAIN WITH NEW DATA ==========
def retrain_with_new_data(original_data_path=r"E:\Ahmed Sameh Work\Projects\GAIAthon 25\Classification Model\datasets\dataset.json", new_data_path=r"E:\Ahmed Sameh Work\Projects\GAIAthon 25\Classification Model\datasets\new_samples.json"):
    """
    Merge new samples with the original dataset and retrain the model.
    """
    print(f"\nAttempting to retrain with new data from '{new_data_path}'...")
    new_data = []
    if os.path.exists(new_data_path) and os.path.getsize(new_data_path) > 0:
        try:
            with open(new_data_path, "r", encoding="utf-8") as f:
                new_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: '{new_data_path}' is not a valid JSON file. Skipping new data loading.")
            return

    if not new_data:
        print("No new samples found to merge. Skipping retraining.")
        return

    # Load original data
    original_data = []
    if os.path.exists(original_data_path) and os.path.getsize(original_data_path) > 0:
        with open(original_data_path, "r", encoding="utf-8") as f:
            original_data = json.load(f)

    # Merge and deduplicate
    combined_data = original_data + new_data
    unique_data = {json.dumps(item, sort_keys=True): item for item in combined_data}
    combined_data = list(unique_data.values())

    # Save merged dataset
    with open(original_data_path, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=2)

    print(f"Added {len(new_data)} new samples to '{original_data_path}'. Total samples now: {len(combined_data)}")

    # Clear new_samples.json after merging
    if os.path.exists(new_data_path):
        os.remove(new_data_path)
        print(f"Removed '{new_data_path}' after successful merge.")

    print("\n--- Starting Model Retraining with Merged Data ---")
    # Re-run the main training pipeline
    main_training_pipeline(original_data_path)
    print("--- Retraining complete with new data! ---")

def main_training_pipeline(data_path="your_dataset.json"):
    """
    Encapsulates the main training workflow.
    """
    train_texts, test_texts, train_labels, test_labels, num_labels = load_and_prepare_data(data_path)
    
    # Debug: Print label distribution
    print("\nLabel distribution in training set:")
    from collections import Counter
    print(Counter(train_labels))
    
    train_dataset, test_dataset = create_datasets(train_texts, test_texts, train_labels, test_labels)
    model = load_bert_model(num_labels)
    
    # Verify model config
    print("\nModel configuration:")
    print(f"Number of labels: {model.config.num_labels}")
    print(f"id2label: {model.config.id2label}")
    
    training_args = get_training_arguments()
    trainer = train_model(model, train_dataset, test_dataset, training_args)
    evaluate_model(trainer, test_dataset)
    save_model_artifacts(trainer)

# ========== MAIN EXECUTION FLOW ==========
if __name__ == "__main__":
    # Create a dummy dataset.json if it doesn't exist for demonstration
    if not os.path.exists("your_dataset.json"):
        dummy_data = [
            {"text": "Photosynthesis is key to plant life.", "label": "Biology"},
            {"text": "Newton's laws describe motion.", "label": "Physics"},
            {"text": "Algebra involves variables and equations.", "label": "Math"},
            {"text": "Developing algorithms for data structures.", "label": "Computer Science"},
            {"text": "Chemical reactions and balancing equations.", "label": "Chemistry"},
            {"text": "Designing bridges requires structural analysis.", "label": "Engineering"},
            {"text": "Literary analysis of Shakespeare's Macbeth.", "label": "English"},
            {"text": "Understanding cell division in organisms.", "label": "Biology"},
            {"text": "Quantum mechanics explores subatomic particles.", "label": "Physics"},
            {"text": "Solving differential equations is complex.", "label": "Math"},
            {"text": "Machine learning models and artificial intelligence.", "label": "Computer Science"},
            {"text": "The properties of acids and bases.", "label": "Chemistry"},
            {"text": "Robotics and automation in manufacturing.", "label": "Engineering"},
            {"text": "Poetic devices and their effect on meaning.", "label": "English"},
            {"text": "The circulatory system transports blood.", "label": "Biology"},
            {"text": "Thermodynamics deals with heat and energy.", "label": "Physics"},
            {"text": "Introduction to calculus and limits.", "label": "Math"},
            {"text": "Object-oriented programming concepts.", "label": "Computer Science"},
            {"text": "Organic chemistry and carbon compounds.", "label": "Chemistry"},
            {"text": "Civil engineering for urban planning.", "label": "Engineering"},
            {"text": "Narrative structures in contemporary fiction.", "label": "English"},
        ]
        with open("your_dataset.json", "w", encoding="utf-8") as f:
            json.dump(dummy_data, f, indent=2)
        print("Created a dummy 'your_dataset.json' for demonstration.")

    # 1. Initial Training of the BERT Model
    print("\n--- Performing Initial Model Training ---")
    main_training_pipeline()

    # 2. Test the enhanced classifier with new, potentially unknown texts
    print("\n=== Testing the trained classifier with hybrid approach ===")
    test_texts_for_inference = [
        "The recent discovery of gravitational waves opens new avenues in astrophysics.", # Should hit pattern for Physics
        "The Roman Empire's decline was a complex process involving many factors.", # Should fall to BERT
        "How does photosynthesis work in plants?", # Should hit pattern for Biology
        "Review of the new abstract art exhibition.", # Should fall to BERT
        "Write a short story about a detective in a foggy city.", # Should fall to BERT
        "Investigating the quantum entanglement phenomenon.", # Should hit pattern for Physics
        "The importance of data structures in software development.", # Should hit pattern for Computer Science
        "Balancing chemical equations is a fundamental skill.", # Should hit pattern for Chemistry
        "Analyzing the latest economic trends and market forecasts.", # Should fall to BERT
        "Exploring the concepts of differential calculus.", # Should hit pattern for Math
        "Building automated systems for industrial applications.", # Should hit pattern for Engineering
    ]

    for text_to_predict in test_texts_for_inference:
        predicted_label = classify_with_unsupervised(text_to_predict, save_new_samples=True)
        print(f"  -> Final Classified Label: {predicted_label}\n")

    # 3. Periodically Retrain the model with accumulated new samples
    # For demonstration, we'll run it immediately after testing.
    # In a real scenario, this would be a scheduled task (e.g., daily/weekly).
    print("\n--- Initiating Periodic Retraining ---")
    retrain_with_new_data()

    # 4. Test the classifier again after retraining to see if it improved (optional)
    print("\n=== Testing classifier after retraining (optional) ===")
    post_retrain_test_texts = [
        "The new theory on dark matter revises our understanding of the universe.", # New text, might have been "unknown" previously
        "Advanced robotics for medical surgery.", # Might now be better classified due to more "Engineering" samples
    ]
    for text_to_predict in post_retrain_test_texts:
        predicted_label = classify_with_unsupervised(text_to_predict, save_new_samples=False) # Don't save again
        print(f"  -> Final Classified Label: {predicted_label}\n")


    print("\n=== Example of loading and using the saved model with Hugging Face pipeline ===")
    try:
        from transformers import pipeline
        saved_model_path = "bert_subject_classifier"

        # Load the label mapping
        with open(os.path.join(saved_model_path, "label_map.json"), "r", encoding="utf-8") as f:
            loaded_label_map = json.load(f)

        # Create an id2label dictionary for the pipeline
        id2label_for_pipeline = {int(k): v for k, v in loaded_label_map.items()}

        # Create a Hugging Face pipeline for easy inference
        classifier_pipeline = pipeline(
            "text-classification",
            model=saved_model_path,
            tokenizer=saved_model_path,
            id2label=id2label_for_pipeline,
            device=0 if torch.cuda.is_available() else -1
        )

        pipeline_test_text = "What is the capital of France?"
        pipeline_result = classifier_pipeline(pipeline_test_text)
        print(f"Pipeline prediction for '{pipeline_test_text}': {pipeline_result}")

        pipeline_test_text_2 = "Analyzing the latest economic trends and market forecasts."
        pipeline_result_2 = classifier_pipeline(pipeline_test_text_2)
        print(f"Pipeline prediction for '{pipeline_test_text_2}': {pipeline_result_2}")

    except ImportError:
        print("\nHugging Face `pipeline` not available. Install with `pip install transformers[sentencepiece]` for easier inference.")
    except Exception as e:
        print(f"\nCould not load model with pipeline (this might be normal if the dataset/labels are not diverse enough for a good pipeline mapping): {e}")