import os
import random

from similarity_search import image_similarity, get_images, get_embedding
from embeddings_manager import plot_embeddings

import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as functional

# Modell laden
model = timm.create_model('convnext_base', pretrained=True)
model.train()

# Nur die letzten Schichten trainierbar machen
for param in model.parameters():
    param.requires_grad = False  # Erst alles einfrieren

for param in model.stem.parameters():  # Optional: Frühere Schichten auftauen
    param.requires_grad = True

for param in model.head.parameters():  # Head trainieren
    param.requires_grad = True

# GPU nutzen, falls vorhanden
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_image(path):
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)  # Batch-Dimension + GPU

def generate_nograd_embedding(image_tensor):
    with torch.no_grad():
        features = model.forward_features(image_tensor)
        embedding = torch.mean(features, dim=[2, 3])  # Global Average Pooling
    return embedding


def generate_embedding(image_tensor):
    features = model.forward_features(image_tensor)
    embedding = torch.mean(features, dim=[2, 3])  # Global Average Pooling
    return embedding

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        # Berechne euklidische Distanz
        distance = torch.norm(emb1 - emb2, p=2, dim=1)

        # Berechne Loss (Y=0 -> ähnliche Paare, Y=1 -> unähnliche Paare)
        loss = (1 - label) * distance.pow(2) + label * torch.clamp(self.margin - distance, min=0).pow(2)
        return loss.mean()
    

def load_and_label(anchor_name, negatives_names, num_samples=40):
    img_folder = 'training_data'
    anchor_folder = os.path.join(img_folder,anchor_name)
    negative_folders = [os.path.join(img_folder, neg_name) for neg_name in negatives_names]

    anchor_images = get_images(anchor_folder)[0:num_samples*2]
    stack_of_negative_images = [get_images(neg_folder)[0:num_samples*2] for neg_folder in negative_folders]

    for img in anchor_images:
        img.img = load_image(img.path) # für die richtige batch-dimension lokale load-funktion verwenden
        #img.emb = generate_embedding(img.img)
        img.label = 'anchor'
    for i in range(len(stack_of_negative_images)):
        for img in stack_of_negative_images[i]:
            img.img = load_image(img.path) # für die richtige batch-dimension lokale load-funktion verwenden
            #img.emb = generate_embedding(img.img)
            img.label = negatives_names[i]

    batch_length = len(anchor_images)//2

    anchors = [img.img for img in anchor_images[:batch_length]] # erste hälfte von houses sind die anchor bilder
    positives = [img.img for img in anchor_images[batch_length:2*batch_length]] # zweites hälfte (genauso lang wie erste hälfte) von houses sind die positiven bilder
    negatives = [[img.img for img in stack[:batch_length]] for stack in stack_of_negative_images]# airplanes sind die negativen (genauso lang wie die anchor bilder)

    return anchors, positives, negatives

def generate_plot(anchors, positives, negatives, save_to=None, show_plot=True):
    all_embeddings = [generate_embedding(anch) for anch in anchors] + [generate_embedding(pos) for pos in positives]
    for stack in negatives:
        all_embeddings = all_embeddings + [generate_embedding(neg) for neg in stack]

    all_label = ['anchor' for i in range(len(anchors))] + ['positive' for i in range(len(positives))]
    for stack in negatives:
        all_label = all_label + ['negative' for i in range(len(stack))]

    plot_embeddings(all_embeddings, all_label, save_to=save_to, show_plot=show_plot)

if __name__ == '__main__':
    ### SETTINGS ###
    anchor_drawing = 'house'
    negative_drawings = ['airplane', 'face', 'bathtub', 'cloud', 'mailbox']
    # negative_drawings = ['airplane', 'face']
    num_samples_per_category = 6
    epochs = 25
    margin = 50
    learning_rate = 0.0001
    ###

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = ContrastiveLoss(margin=margin)

    info_string = f"""### SETTINGS ###\nanchor_drawing = {anchor_drawing}\nnegative_drawings = {negative_drawings}\nnumber of samples per drawing category = {num_samples_per_category}\nepochs = {epochs}\nmargin = {margin}\nlearning_rate = {learning_rate}###\n\n"""

    # ordner mit random zahl bennen, damit nicht ausversehen was überschrieben wird. übergangslösung
    results_folder = os.path.join('finetuning_results', 'nc_'+str(len(negative_drawings))+'_spc_'+str(num_samples_per_category)+'_epochs_'+str(epochs)+'_'+str(random.randint(1,100000)))
    os.makedirs(results_folder)

    # Negatives ist jetzt ein Stack aus mehreren samples
    anchors, positives, negatives = load_and_label(anchor_name=anchor_drawing, negatives_names=negative_drawings, num_samples=num_samples_per_category)


    # print similarities
    info_string += 'Similaritys vorher:\n'
    for q in range(3):
        info_string += 'sample_'+str(q+1)+':\n'
        # cosine_similarity_12 = functional.cosine_similarity(generate_embedding(anchors[q]), generate_embedding(positives[q])).item()
        # cosine_similarity_13 = functional.cosine_similarity(generate_embedding(anchors[q]), generate_embedding(negatives[0][q])).item()
        distance_pos = torch.norm(generate_embedding(anchors[q]) - generate_embedding(positives[q]), p=2)
        distance_neg = torch.norm(generate_embedding(anchors[q]) - generate_embedding(negatives[0][q]), p=2)
        info_string += f"Ähnlichkeit zwischen anchor & positive: {distance_pos}\n"
        info_string += f"Ähnlichkeit zwischen anchor & negative: {distance_neg}\n"
    info_string += '\n'
    print(info_string)
    

    for epoch in range(epochs):  # Anzahl der Epochen
        if epoch % 5 == 0:
            generate_plot(anchors, positives, negatives, save_to=os.path.join(results_folder, 'epochs_'+str(epoch)+'.png'), show_plot=False)

        optimizer.zero_grad()

        batch_anchors = torch.stack([generate_embedding(img) for img in anchors])  # Anker-Bilder
        batch_pos = torch.stack([generate_embedding(img) for img in positives])  # Positive Paare

        # batch_neg = torch.stack([generate_embedding(img) for img in negatives])  # Negative Paare

        # is a sstack of negative samples
        batch_neg = torch.stack([torch.stack([generate_embedding(img) for img in stack]) for stack in negatives]) # Shape: (num_negatives, batch_size, embedding_dim)

        labels_positive = torch.zeros(batch_anchors.shape[0], device=device).unsqueeze(1).unsqueeze(2).expand(-1, 1, 1024)  # 0 = Ähnliche Bilder
        labels_negative = torch.ones(batch_anchors.shape[0], device=device).unsqueeze(1).unsqueeze(2).expand(-1, len(negatives), 1024)  # 1 = Unterschiedliche Bilder

        loss_pos = criterion(batch_anchors, batch_pos, labels_positive)
        # loss_neg = criterion(batch_anchors, batch_neg, labels_negative)
        # Berechne den Durchschnitt über alle negativen Paare
        loss_neg = torch.mean(torch.stack([criterion(batch_anchors, batch_neg[i], labels_negative[:, i]) for i in range(len(negatives))]))

        loss = (loss_pos + loss_neg) / 2  # Durchschnitt über alle Paare
        loss.backward()
        optimizer.step()
        

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        info_string += f"Epoch {epoch+1}, Loss: {loss.item()}\n"

    info_string += '\nSimilaritys nachher:\n'
    for q in range(3):
        info_string += 'sample_'+str(q+1)+':\n'
        # cosine_similarity_12 = functional.cosine_similarity(generate_embedding(anchors[q]), generate_embedding(positives[q])).item()
        # cosine_similarity_13 = functional.cosine_similarity(generate_embedding(anchors[q]), generate_embedding(negatives[0][q])).item()
        distance_pos = torch.norm(generate_embedding(anchors[q]) - generate_embedding(positives[q]), p=2)
        distance_neg = torch.norm(generate_embedding(anchors[q]) - generate_embedding(negatives[0][q]), p=2)
        info_string += f"Ähnlichkeit zwischen anchor & positive: {distance_pos}\n"
        info_string += f"Ähnlichkeit zwischen anchor & negative: {distance_neg}\n"

    generate_plot(anchors, positives, negatives, save_to=os.path.join(results_folder, 'epochs_'+str(epochs)+'.png'), show_plot=False)
    
    with open(os.path.join(results_folder, 'info.txt'), "w", encoding="utf-8") as file:
        file.write(info_string)