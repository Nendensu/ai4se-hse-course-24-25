from statistics import mean, stdev

import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk

def classifier(dataset, model):
    X = [d['message'] for d in dataset]
    y = [d['is_toxic'] for d in dataset]
    
    vectorizer = TfidfVectorizer(max_features=5000)#, ngram_range=(1,3))
    X = vectorizer.fit_transform(X)
    
    model_obj = LogisticRegression(max_iter=9000, C=2)
    
    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    scores = []
    all_conf_matrices = []
    
    for train_index, test_index in tqdm(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
        
        model_obj.fit(X_train, y_train)
        y_pred = model_obj.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        scores.append(acc)
        all_conf_matrices.append(confusion_matrix(y_test, y_pred))
    
    print('Mean value:', mean(scores))
    print('Stdev:', stdev(scores))
    print('Confusion matrix:')
    print(all_conf_matrices[0])

def run_codebert(dataset_path, model_name="microsoft/codebert-base"):
    dataset = load_from_disk(dataset_path)

    train_testvalid = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_testvalid["train"]
    eval_dataset = train_testvalid["test"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        tokenized = tokenizer(examples["message_clean"], padding="max_length", truncation=True, max_length=128)
        tokenized["labels"] = examples["is_toxic"]
        return tokenized

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        acc = accuracy_score(labels, predictions)
        return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

    training_args = TrainingArguments(
        output_dir="./codebert_results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        logging_dir="./logs",
        logging_steps=50,
        save_total_limit=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Eval metrics:", metrics)