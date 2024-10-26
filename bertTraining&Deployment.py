# Library imports
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import aivm_client as aic # Import the Nillion-AIVM client
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import numpy as np
import time
import tempfile, onnx
from datasets import load_dataset

# Checking hardware availability for AI training
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} can be used for training!")
else:
    print("No GPU available... Going to use the CPU for AI training.")


# Define constructor for custom dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Layer freeze for the bottom-layers of BERT. We will use the parameter 'model' to pass the BerTiny model and we will specify how many layers to freeze.
def freeze_bert_layers(model, num_layers_to_freeze):
    # Freeze embeddings
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False

    # Freeze the specified number of encoder layers
    for layer in model.bert.encoder.layer[:num_layers_to_freeze]:
        for param in layer.parameters():
            param.requires_grad = False

    # Print trainable parameters info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")
    
    
# Function for training of BERTiny 
def Training_Function(model, train_loader, val_loader, device, num_epochs=3):
    # Initialize optimizer only with parameters that require gradients
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-5
    )

    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        total_train_loss = 0
        right_predictions = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            model.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # Calculate accuracy
            preds = torch.argmax(outputs.logits, dim=1)
            right_predictions += torch.sum(preds == labels).item()

            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Validation
        model.eval()
        total_val_loss = 0
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_val_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))
        end_time = time.time()

        print(f'Epoch {epoch + 1}:')
        print(f'Average training loss: {avg_train_loss:.4f}')
        print(f'Average validation loss: {avg_val_loss:.4f}')
        print(f'Right predictions: {right_predictions} out of {len(train_loader) * 32}')
        print(f'Validation Accuracy: {accuracy:.4f}')
        print(f'Time taken for epoch: {end_time - start_time:.2f} seconds')
        print('-' * 60)

# Function for the conversion of Pytorch to Onnx, with tokenization included
def convert_pytorch_to_onnx_with_tokenizer(model, tokenizer, max_length=128, onnx_file_path=None):
    """
    Converts a PyTorch model to ONNX format, using tokenizer output as input.

    Args:
    model (torch.nn.Module): The PyTorch model to be converted.
    tokenizer: The tokenizer used to preprocess the input.
    onnx_file_path (str): The file path where the ONNX model will be saved.
    max_length (int): Maximum sequence length for the tokenizer.

    Returns:
    None
    """
    model.eval()

    # Prepare dummy input using the tokenizer
    dummy_input = "This is a sample input text for ONNX conversion."
    inputs = tokenizer(
        dummy_input,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # # Get the input names
    input_names = list(inputs.keys())
    input_names = ["input_ids", "attention_mask"]
    print(f"Input names: {input_names}")

    # # Create dummy inputs for ONNX export
    # dummy_inputs = tuple(encoded_input[name] for name in input_names)
    if onnx_file_path is None:
      onnx_file_path = tempfile.mktemp(suffix=".onnx")
    dynamic_axes = {name: {0: "batch_size"} for name in input_names}
    dynamic_axes.update({f"logits": {0: "batch_size"}})
    print(f"dynamic_axes: {dynamic_axes}")
    # Export the model
    torch.onnx.export(
        model,  # model being run
        tuple(inputs[k] for k in input_names),  # model inputs
        onnx_file_path,  # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=20,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=input_names,  # the model's input names
        output_names=["logits"],  # the model's output names
        dynamic_axes=dynamic_axes,
    )  # variable length axes

    print(f"Model exported to {onnx_file_path}")

    # Verify the exported model
    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid.")
    return onnx_file_path, input_names


# Initialising tokenization and new model
model_name = "prajjwal1/bert-tiny"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2  # Binary classification for sentiment
)

def main(model, tokenizer):
    # Load IMDB dataset
    dataset = load_dataset("stanfordnlp/imdb") # hugging-face dataset

    # Prepare train and validation datasets
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']

    # Split training data to create a validation set
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )

    # Create datasets
    train_dataset = CustomDataset(train_texts, train_labels, tokenizer)
    val_dataset = CustomDataset(val_texts, val_labels, tokenizer)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, num_workers=2
    )

    # Set device and move data to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train the model
    Training_Function(model, train_loader, val_loader, device, num_epochs=3)

    # Make model tensors contiguous and move to CPU before saving
    model = model.cpu()

    # Save the fine-tuned model as an ONNX file
    onnx_file_path, input_names = convert_pytorch_to_onnx_with_tokenizer(
        model, tokenizer, max_length=128, onnx_file_path="./my_new_bert_model.onnx"
    )
    print(f"ONNX file path: {onnx_file_path}")
    print(f"Input names: {input_names}")

    # Test the model on a few examples
    model.eval()
    test_texts = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Terrible waste of time. The plot made no sense and the acting was awful."
    ]

    with torch.no_grad():
        inputs = tokenizer(
            test_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        model.to(device)
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=1)

        for text, pred in zip(test_texts, predictions):
            sentiment = "Positive" if pred[1] > pred[0] else "Negative"
            confidence = max(pred[0], pred[1]).item()
            print(f"\nText: {text}")
            print(f"Sentiment: {sentiment} (confidence: {confidence:.2%})")

# Call the main function
main(model, tokenizer)

# After calling main function, we want to upload the model
MODEL_NAME = "IMDB-Tutorial-BertTiny" # Name of the model to be used
aic.upload_bert_tiny_model("./imdb-bert-tiny.onnx", MODEL_NAME) # Upload the model to the server

# Preparing inputs to be tokenized and assessed through inference to new model.

tokenized_positive_inputs = aic.tokenize("This movie was absolutely fantastic! I loved every minute of it.",)
encrypted_positive_inputs = aic.BertTinyCryptensor(tokenized_positive_inputs[0], tokenized_positive_inputs[1])

tokenized_negative_inputs = aic.tokenize("Terrible waste of time. The plot made no sense and the acting was awful.")
encrypted_negative_inputs = aic.BertTinyCryptensor(tokenized_negative_inputs[0], tokenized_negative_inputs[1])

result_positive = aic.get_prediction(encrypted_positive_inputs, MODEL_NAME)
result_negative = aic.get_prediction(encrypted_negative_inputs, MODEL_NAME)

# Printing the predictions for the results.
print("Positive review prediction: ", result_positive)
print("Negative review prediction: ", result_negative)

sentiment = lambda x: "Negative" if torch.argmax(x) == 0 else "Positive" if torch.argmax(x) == 2 else "Neutral"
print("Positive review prediction: ", sentiment(result_positive))
print("Negative review prediction: ", sentiment(result_negative))