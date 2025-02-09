import os

from similarity_search import image_similarity, get_images, get_embedding

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
    
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = ContrastiveLoss(margin=1.0)

# img_folder = 'training_data'
# # Beispiel-Daten: (Positives Paar, Negatives Paar)
# img1 = load_image(os.path.join(img_folder, "airplane1.png"))
# img2 = load_image(os.path.join(img_folder, "airplane2.png"))  # Positiv
# img3 = load_image(os.path.join(img_folder, "apple1.png"))  # Negativ

# emb1 = generate_embedding(img1)
# emb2 = generate_embedding(img2)
# emb3 = generate_embedding(img3)

# print('\nSimilaritys vorher:')
# cosine_similarity_12 = functional.cosine_similarity(emb1, emb2).item()
# cosine_similarity_13 = functional.cosine_similarity(emb1, emb3).item()
# print(f"Ähnlichkeit zwischen Bild1 & Bild2: {cosine_similarity_12}")
# print(f"Ähnlichkeit zwischen Bild1 & Bild3: {cosine_similarity_13}\n")



def load_and_label():
    img_folder = 'training_data'
    houses_folder = os.path.join(img_folder,'house')
    airplane_folder = os.path.join(img_folder,'airplane')

    houses = get_images(houses_folder)[0:6]
    airplanes = get_images(airplane_folder)[0:6]

    for img in houses:
        img.img = load_image(img.path) # für die richtige batch-dimension lokale load-funktion verwenden
        #img.emb = generate_embedding(img.img)
        img.label = 'house'

    for img in airplanes:
        img.img = load_image(img.path) # für die richtige batch-dimension lokale load-funktion verwenden
        #img.emb = generate_embedding(img.img)
        img.label = 'airplane'

    batch_length = len(houses)//2

    anchors = [img.img for img in houses[:batch_length]] # erste hälfte von houses sind die anchor bilder
    positives = [img.img for img in houses[batch_length:2*batch_length]] # zweites hälfte (genauso lang wie erste hälfte) von houses sind die positiven bilder
    negatives = [img.img for img in airplanes[:batch_length]] # airplanes sind die negativen (genauso lang wie die anchor bilder)


    return anchors, positives, negatives

anchors, positives, negatives = load_and_label()

for q in range(3):
    print('\nSimilaritys vorher:')
    cosine_similarity_12 = functional.cosine_similarity(generate_embedding(anchors[q]), generate_embedding(positives[q])).item()
    cosine_similarity_13 = functional.cosine_similarity(generate_embedding(anchors[q]), generate_embedding(negatives[q])).item()
    print(f"Ähnlichkeit zwischen anchor & positive: {cosine_similarity_12}")
    print(f"Ähnlichkeit zwischen anchor & negative: {cosine_similarity_13}\n")

for epoch in range(5):  # Anzahl der Epochen
    optimizer.zero_grad()

    batch_anchors = torch.stack([generate_embedding(img) for img in anchors])  # Anker-Bilder
    batch_pos = torch.stack([generate_embedding(img) for img in positives])  # Positive Paare
    batch_neg = torch.stack([generate_embedding(img) for img in negatives])  # Negative Paare

    labels_positive = torch.zeros(batch_anchors.shape[0], device=device).unsqueeze(1).unsqueeze(2).expand(-1, 1, 1024)  # 0 = Ähnliche Bilder
    labels_negative = torch.ones(batch_anchors.shape[0], device=device).unsqueeze(1).unsqueeze(2).expand(-1, 1, 1024)  # 1 = Unterschiedliche Bilder


    loss_pos = criterion(batch_anchors, batch_pos, labels_positive)
    loss_neg = criterion(batch_anchors, batch_neg, labels_negative)

    loss = (loss_pos + loss_neg) / 2  # Durchschnitt über alle Paare
    loss.backward()
    optimizer.step()
    

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

for q in range(3):
    print('\nSimilaritys nachher:')
    cosine_similarity_12 = functional.cosine_similarity(generate_embedding(anchors[q]), generate_embedding(positives[q])).item()
    cosine_similarity_13 = functional.cosine_similarity(generate_embedding(anchors[q]), generate_embedding(negatives[q])).item()
    print(f"Ähnlichkeit zwischen anchor & positive: {cosine_similarity_12}")
    print(f"Ähnlichkeit zwischen anchor & negative: {cosine_similarity_13}\n")

