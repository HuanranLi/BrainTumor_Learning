import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
import torch

from nltk.translate.bleu_score import corpus_bleu



import os
# Get the absolute path of the current directory
current_directory = os.path.abspath(os.path.dirname(__file__))
# Set the TORCH_HOME environment variable to the current directory
os.environ['TORCH_HOME'] = current_directory
os.environ['TRANSFORMERS_CACHE'] = current_directory

import torch
import torch.nn as nn
from torchvision.models import resnet18
from transformers import GPT2Model

from utils_main_v2 import *

class ImageToTextModel(nn.Module):
    def __init__(self):
        super(ImageToTextModel, self).__init__()

        # Load the pretrained ResNet18 model
        self.resnet = resnet18(pretrained=True)
        features = self.resnet.fc.in_features
        # Remove the last fully connected layer (classification layer)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        # Freeze the parameters (weights) of ResNet18
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Load the pretrained GPT-2 model
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
        # Freeze the parameters (weights) of GPT-2
        for param in self.gpt2.parameters():
            param.requires_grad = False

        # Define a 3-layer FFN
        self.ffn = nn.Sequential(
            nn.Linear(features, 512),  # Adjust input size based on ResNet's last layer's output
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.gpt2.config.n_embd)  # Output size to match GPT-2's embedding size
        )

    def get_tokenizer(self):
        return GPT2Tokenizer.from_pretrained('gpt2')

    def forward(self, images):
        # Pass images through ResNet18
        image_features = self.resnet(images)
        image_features = torch.flatten(image_features, 1)  # Flatten the output

        # Pass the image features through the FFN
        transformed_features = self.ffn(image_features)

        # Use the transformed features as inputs to GPT-2
        gpt2_outputs = self.gpt2(inputs_embeds=transformed_features)

        return gpt2_outputs


def pad_or_truncate_logits(logits, target_length, pad_value=0):
    """
    Pad or truncate the logits to match the target sequence length.

    :param logits: Tensor of logits from the model, shape [batch_size, seq_length, num_classes].
    :param target_length: The desired sequence length after padding/truncating.
    :param pad_value: The value used for padding.
    :return: Adjusted logits with shape [batch_size, target_length, num_classes].
    """
    current_length = logits.size(1)

    if current_length > target_length:
        # Truncate the logits
        return logits[:, :target_length]
    elif current_length < target_length:
        # Pad the logits
        padding_size = target_length - current_length
        padding = torch.full((logits.size(0), padding_size), pad_value)
        return torch.cat([logits, padding], dim=1)

    return logits

# Default Hyperparameters
train_hyperparams = {
    'batch_size': 32,
    'num_epochs': 200,
    'resize': (224, 224),
    'normalize_means': (0.5, 0.5, 0.5),
    'normalize_stds': (0.5, 0.5, 0.5),
    'temperature': 1,
    'train_dataset_dir': '../dataset/brain_tumor/Training',
    'test_dataset_dir': '../dataset/brain_tumor/Testing',
}

# Data Init
train_loader, test_loader = load_data(train_hyperparams)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 # Only optimize FFN parameters

# Instantiate the model
model = ImageToTextModel().to(device)
tokenizer = model.get_tokenizer()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.ffn.parameters(), lr=0.001)

epochs = 100
max_seq_length = 128

# Training loop
model.train()
for epoch in range(epochs):
    train_loss = 0
    for image,label in train_loader:

        inputs = image.to(device)
        label = label.to(device)
        outputs_logits = model(inputs).logits
        outputs_logits = pad_or_truncate_logits(outputs_logits, max_seq_length)

        labels_id = torch.tensor([tokenizer.encode(f"This is an MRI image with {single_label}.") for single_label in label])
        labels_id = pad_or_truncate_logits(labels_id, max_seq_length).float()
        labels_id = labels_id.to(device)

        loss = criterion(outputs_logits, labels_id)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    average_train_loss = train_loss / len(train_loader)
    print(f'Epoch: {epoch}, Avg Train Loss: {average_train_loss}')

    generated_texts = []
    references = []
    # Model inference
    for images, true_texts in test_loader:
        image = image.to(device)
        true_texts = true_texts.to(device)
        with torch.no_grad():
            outputs = model(images)
            logits = outputs.logits
            # Post-process logits to generate text
            generated_ids = torch.argmax(logits, dim=-1)
            generated_text = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
            generated_texts.extend(generated_text)

        # Store references for BLEU calculation
        references.extend([tokenizer.decode(tokenizer.encode(f"This is an MRI image with {t}."), skip_special_tokens=True) for t in true_texts])
    bleu_score = corpus_bleu([[ref.split()] for ref in references], [gen.split() for gen in generated_texts])
    print(f"BLEU Score: {bleu_score}")





#
#
# # Example prompt
# prompt = "The future of AI in medicine is"
#
# # Encode the prompt
# encoded_input = tokenizer.encode(prompt, return_tensors='pt')
#
# # Generate text
# output_sequences = model.generate(
#     input_ids=encoded_input,
#     max_length=50,  # Set the maximum length of the generated text
#     num_return_sequences=1,  # Number of sequences to generate
#     no_repeat_ngram_size=2,  # Prevents the model from repeating shorter chunks
#     temperature=0.7  # Controls the randomness of the generated text
# )
#
# # Decode the generated text
# generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
# print(generated_text)
