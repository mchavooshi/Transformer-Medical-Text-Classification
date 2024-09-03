
import re
import random
from collections import Counter
import torch
from torch.utils.data import TensorDataset



def clean_text(text):
    # Remove leading and trailing white spaces
    text = text.strip()
    
    # Replace multiple white spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Convert text to lowercase (if using an uncased model)
    text = text.lower()
    
    return text

def balance_dataset(texts, labels):
    """
    Balance the dataset by undersampling the majority classes to match the minority class count.
    
    Parameters:
    texts (list of str): List of text samples.
    labels (list of int): List of corresponding labels.
    
    Returns:
    balanced_texts (list of str): List of balanced text samples.
    balanced_labels (list of int): List of corresponding balanced labels.
    """
    # Count the number of instances for each class
    label_counts = Counter(labels)
    print("Original label distribution:", label_counts)

    # Determine the number of samples in the minority class
    minority_class_count = min(label_counts.values())

    # Create a balanced dataset by undersampling the majority classes
    balanced_texts = []
    balanced_labels = []

    for label in set(labels):
        # Get indices of all samples with the current label
        label_indices = [i for i, lbl in enumerate(labels) if lbl == label]
        # Undersample to the number of samples in the minority class
        sampled_indices = random.sample(label_indices, minority_class_count)
        # Add the undersampled texts and labels to the balanced dataset
        balanced_texts.extend([texts[i] for i in sampled_indices])
        balanced_labels.extend([labels[i] for i in sampled_indices])

    # Verify the new label distribution
    balanced_label_counts = Counter(balanced_labels)
    print("Balanced label distribution:", balanced_label_counts)

    return balanced_texts, balanced_labels

def preprocess(texts, labels, tokenizer, max_length=512):
    """
    Preprocess the texts and labels for model input.
    
    Parameters:
    texts (list of str): List of text samples.
    labels (list of int): List of corresponding labels.
    tokenizer: Tokenizer to use.
    max_length (int): Maximum length for tokenization.
    
    Returns:
    TensorDataset: PyTorch dataset containing input_ids, attention_masks, and labels.
    """
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    dataset = TensorDataset(input_ids, attention_masks, labels)
    
    return dataset
