import timm
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as functional

def image_similarity(img1, img2, model_name='efficientnet_b0'):
    timm_models = [
        'efficientnet_b0',
        'efficientnet_b1',
        'efficientnet_b2',
        'swin_base_patch4_window7_224',
        'convnext_base',
        'regnetx_400mf',
        'resnet50',
        'vgg19',
    ]

    tensorflow_models = [
        'sketchrnn_model###'
    ]

    if model_name in timm_models:
        model = timm.create_model(model_name, pretrained=True)
        model.eval() 
    else:
        print('     ### invalid modelname ###')
        return None

    transform = transforms.Compose([
        transforms.Resize(224),  # EffizientNet erwartet 224x224 pixel als eingabe
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def generate_embedding(path):
        image = Image.open(path).convert("RGB")  
        input_tensor = transform(image).unsqueeze(0)  # Bild transformieren und Batch-Dimension hinzufügen
        with torch.no_grad():
            features = model.forward_features(input_tensor)  # Feature-Map extrahieren
            embedding = torch.mean(features, dim=[2, 3])  # Global Average Pooling 
        return embedding

    emb1 = generate_embedding(img1.path)
    emb2 = generate_embedding(img2.path)

    cosine_similarity = functional.cosine_similarity(emb1, emb2).item()

    return cosine_similarity