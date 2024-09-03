

import torch

from torch import nn

from torch.utils.data import DataLoader, TensorDataset, random_split

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, AdamW

import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from datasets import load_dataset

from data_utils import clean_text, balance_dataset, preprocess


# Load dataset from Hugging Face

dataset = load_dataset("123rc/medical_text")



train_data = dataset['train']

test_data = dataset['test']



# Extract and clean texts and labels from train_data

train_texts = [clean_text(item['medical_abstract']) for item in train_data]

train_labels = [item['condition_label'] for item in train_data]



# Extract and clean texts and labels from test_data

test_texts = [clean_text(item['medical_abstract']) for item in test_data]

test_labels = [item['condition_label'] for item in test_data]



# Adjust labels to be in the range [0, num_labels-1]

train_labels = [label - 1 for label in train_labels]

test_labels = [label - 1 for label in test_labels]



# Split training data into training and validation sets

train_size = int(0.8 * len(train_texts))

val_size = len(train_texts) - train_size



train_texts, val_texts = train_texts[:train_size], train_texts[train_size:]

train_labels, val_labels = train_labels[:train_size], train_labels[train_size:]



# Hyperparameters

num_epochs = 20

learning_rate = 1e-6

batch_size = 16

num_labels = 5



# Load the Bio_ClinicalBERT tokenizer and model

bio_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

bio_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=num_labels)



# Function for training and evaluating the model

def train_and_evaluate(learning_rate, batch_size):

    # Initialize the model

    model = BioClinicalClassifier(bio_model, num_labels)



    # Preprocess the data

    train_dataset = preprocess(train_texts, train_labels, bio_tokenizer)

    val_dataset = preprocess(val_texts, val_labels, bio_tokenizer)

    test_dataset = preprocess(test_texts, test_labels, bio_tokenizer)



    # Create DataLoaders

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



    # Set up the optimizer and learning rate scheduler

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    total_steps = len(train_loader) * num_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)



    # Define the loss function

    loss_fn = nn.CrossEntropyLoss()



    # Move model to GPU

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)



    # Track losses

    train_losses = []

    val_losses = []



    # Training loop

    for epoch in range(num_epochs):

        model.train()

        total_loss = 0



        for batch in train_loader:

            input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)



            model.zero_grad()

            output = model(input_ids, attention_mask)



            loss = loss_fn(output, labels)

            total_loss += loss.item()



            loss.backward()

            optimizer.step()

            scheduler.step()



        avg_train_loss = total_loss / len(train_loader)

        train_losses.append(avg_train_loss)

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss}")



        # Validation

        model.eval()

        val_loss = 0



        with torch.no_grad():

            for batch in val_loader:

                input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)

                output = model(input_ids, attention_mask)

                loss = loss_fn(output, labels)

                val_loss += loss.item()



        avg_val_loss = val_loss / len(val_loader)

        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs} - Val Loss: {avg_val_loss}")



    # Plot training and validation loss on the same plot

    plt.figure(figsize=(10, 6))

    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')

    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')

    plt.xlabel('Epochs')

    plt.ylabel('Loss')

    plt.title(f'Loss Plot (LR={learning_rate}, BS={batch_size})')

    plt.legend()

    plt.savefig(os.path.join(save_path, f"loss_plot_lr_{learning_rate}_bs_{batch_size}_ep_{num_epochs}.png"))



    # Evaluation loop

    model.eval()

    eval_accuracy = 0

    all_preds = []

    all_labels = []



    for batch in test_loader:

        input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)



        with torch.no_grad():

            output = model(input_ids, attention_mask)



        preds = torch.argmax(output, dim=1).flatten()

        eval_accuracy += (preds == labels).cpu().numpy().mean()



        all_preds.extend(preds.cpu().numpy())

        all_labels.extend(labels.cpu().numpy())



    eval_accuracy /= len(test_loader)

    print(f"Evaluation Accuracy: {eval_accuracy}")



    # Plot confusion matrix

    conf_matrix = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))

    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(train_labels), yticklabels=np.unique(train_labels))

    plt.xlabel('Predicted')

    plt.ylabel('Actual')

    plt.title(f'Confusion Matrix (LR={learning_rate}, BS={batch_size})')

    plt.savefig(os.path.join(save_path, f"confusion_matrix_lr_{learning_rate}_bs_{batch_size}_ep_{num_epochs}.png"))



    # Plot confusion matrix percentages

    conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))

    sns.heatmap(conf_matrix_percentage, annot=True, fmt='.2%', cmap='Blues', xticklabels=np.unique(train_labels), yticklabels=np.unique(train_labels))

    plt.xlabel('Predicted')

    plt.ylabel('Actual')

    plt.title(f'Confusion Matrix Percentage (LR={learning_rate}, BS={batch_size})')

    plt.savefig(os.path.join(save_path, f"confusion_matrix_percentage_lr_{learning_rate}_bs_{batch_size}_ep_{num_epochs}.png"))



    # Get the classification report

    report = classification_report(all_labels, all_preds, output_dict=True)



    # Evaluate the model and create the table

    data = []

    for label in range(num_labels):

        precision = report[str(label)]['precision']

        recall = report[str(label)]['recall']

        f1 = report[str(label)]['f1-score']

        

        # Calculate TP, FN, FP, TN for each class

        tp = conf_matrix[label, label]

        fn = np.sum(conf_matrix[label, :]) - tp

        fp = np.sum(conf_matrix[:, label]) - tp

        tn = np.sum(conf_matrix) - (tp + fn + fp)

        

        sensitivity = tp / (tp + fn) if tp + fn != 0 else 0

        specificity = tn / (tn + fp) if tn + fp != 0 else 0

        ppv = tp / (tp + fp) if tp + fp != 0 else 0

        npv = tn / (tn + fn) if tn + fn != 0 else 0



        data.append({

            'Class': label,

            'Precision': precision,

            'Recall': recall,

            'F1 Score': f1,

            'Sensitivity': sensitivity,

            'Specificity': specificity,

            'PPV': ppv,

            'NPV': npv

        })



    # Accuracy

    ACC = accuracy_score(all_labels, all_preds)



    # Append the accuracy

    data.append({

        'Class': 'Overall',

        'Precision': None,

        'Recall': None,

        'F1 Score': None,

        'Sensitivity': None,

        'Specificity': None,

        'PPV': None,

        'NPV': None,

        'Accuracy': ACC

    })



    # Create a pandas DataFrame

    df = pd.DataFrame(data)



    # Save to CSV

    df.to_csv(os.path.join(save_path, f'classification_report_lr_{learning_rate}_bs_{batch_size}_ep_{num_epochs}.csv'), index=False)



    # Return the number of epochs trained

    return num_epochs







epochs = train_and_evaluate(learning_rate, batch_size)

print(f"Training completed in {epochs} epochs for learning rate {learning_rate} and batch size {batch_size}and epochs {num_epochs}")







print("DONE!")
