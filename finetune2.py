import os
import random
import time

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
        embedding = torch.mean(features, dim=[2, 3]).squeeze(0)  # Global Average Pooling
    return embedding


def generate_embedding(image_tensor):
    features = model.forward_features(image_tensor)
    embedding = torch.mean(features, dim=[2, 3]).squeeze(0) # Global Average Pooling
    return embedding

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        # Berechne euklidische Distanz
        distance = torch.norm(emb1 - emb2, p=2)

        # Berechne Loss (Y=0 -> ähnliche Paare, Y=1 -> unähnliche Paare)
        loss = (1 - label) * distance.pow(2) + label * torch.clamp(self.margin - distance, min=0).pow(2)
        return loss.mean()

# class TripletLoss(nn.Module):
#     def __init__(self, margin=1.0):
#         super(TripletLoss, self).__init__()
#         self.margin = margin

#     def forward(self, anchor, positive, negative):
#         # Berechne euklidische Distanzen
#         pos_dist = torch.norm(anchor - positive, p=2, dim=1)  # Distanz zum positiven Beispiel
#         neg_dist = torch.norm(anchor - negative, p=2, dim=1)  # Distanz zum negativen Beispiel

#         # Berechne Triplet Loss
#         loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0)

#         return loss.mean()
    
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, pos_dist, neg_dist):
        # Berechne Triplet Loss
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0)

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

def get_distance(batch1, batch2):
    return torch.norm(batch1 - batch2, p=2, dim=1).mean()

def get_multi_l2distance(batch1, batch2, elementwise=True):  # expects to lists of images and gives back the normed distance between those two batches
    diff = torch.stack([generate_embedding(b1) for b1 in batch1]) - torch.stack([generate_embedding(b2) for b2 in batch2])   # shape t([2, 1024])
    if elementwise:
        dist = torch.norm(diff, p=2, dim=1).mean() # shape t([2]) -> [ ||a1-p1||2, ||a2-p2||2 ] = [dap1, dap2]
    else:
        dist = torch.norm(diff, p=2)
    return dist

def test_image_batches():
    img_folder = 'training_data'

    anchor1 = load_image(os.path.join(img_folder, 'airplane1.png'))
    anchor2 = load_image(os.path.join(img_folder, 'airplane2.png'))
    pos1 = load_image(os.path.join(img_folder, 'airplane3.png'))
    pos2 = load_image(os.path.join(img_folder, 'airplane4.png'))
    neg11 = load_image(os.path.join(img_folder, 'apple1.png'))
    neg12 = load_image(os.path.join(img_folder, 'apple2.png'))
    neg21 = load_image(os.path.join(img_folder, 'cloud1.png'))
    neg22 = load_image(os.path.join(img_folder, 'cloud2.png'))
    anchor = [anchor1, anchor2]
    pos = [pos1, pos2]
    neg1 = [neg11, neg12]
    neg2 = [neg21, neg22]
    neg = [neg1, neg2]
    return anchor, pos, neg

def main_old():
    ### SETTINGS ###
    anchor_drawing = 'house'
    negative_drawings = ['airplane', 'face', 'bathtub', 'cloud', 'mailbox']
    negative_drawings = ['airplane', 'face']
    num_samples_per_category = 6
    epochs = 3
    margin = 120
    learning_rate = 0.001
    metric = 'l2-norm'
    ###

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # criterion = ContrastiveLoss(margin=margin)
    criterion = TripletLoss(margin=margin)

    info_string = f"""### SETTINGS ###\nanchor_drawing = {anchor_drawing}\nnegative_drawings = {negative_drawings}\nnumber of samples per drawing category = {num_samples_per_category}\nepochs = {epochs}\nmargin = {margin}\nlearning_rate = {learning_rate}\nmetric = {metric}\n###\n\n"""

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

    batch_anchors = torch.stack([generate_embedding(img) for img in anchors])  # Anker-Bilder
    batch_pos = torch.stack([generate_embedding(img) for img in positives])  # Positive Paare
    batch_neg = torch.stack([torch.stack([generate_embedding(img) for img in stack]) for stack in negatives])
    info_string += f"Ähnlichkeit batch anchor & positive: {get_distance(batch_anchors, batch_pos)}\n"
    info_string += f"Ähnlichkeit batch anchor & negative: {get_distance(batch_anchors, batch_neg)}\n\n"
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

        # loss_pos = criterion(batch_anchors, batch_pos, labels_positive)
        # # loss_neg = criterion(batch_anchors, batch_neg, labels_negative)
        # # Berechne den Durchschnitt über alle negativen Paare
        # loss_neg = torch.mean(torch.stack([criterion(batch_anchors, batch_neg[i], labels_negative[:, i]) for i in range(len(negatives))]))

        # loss = (loss_pos + loss_neg) / 2  # Durchschnitt über alle Paare
        loss = criterion(batch_anchors, batch_pos, batch_neg)
        loss.backward()
        optimizer.step()
        

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        info_string += f"Epoch {epoch+1}, Loss: {loss.item()}\n"

    info_string += '\nSimilaritys nachher:\n'
    for q in range(3):
        info_string += 'sample_'+str(q+1)+':\n'
        # cosine_similarity_12 = functional.cosine_similarity(generate_embedding(anchors[q]), generate_embedding(positives[q])).item()
        # cosine_similarity_13 = functional.cosine_similarity(generate_embedding(anchors[q]), generate_embedding(negatives[0][q])).item()
        distance_pos = torch.norm(generate_embedding(anchors[q]) - generate_embedding(positives[q]), p=2, dim=1).mean()
        distance_neg = torch.norm(generate_embedding(anchors[q]) - generate_embedding(negatives[0][q]), p=2, dim=1).mean()
        info_string += f"Ähnlichkeit zwischen anchor & positive: {distance_pos}\n"
        info_string += f"Ähnlichkeit zwischen anchor & negative: {distance_neg}\n"

    batch_anchors = torch.stack([generate_embedding(img) for img in anchors])  # Anker-Bilder
    batch_pos = torch.stack([generate_embedding(img) for img in positives])  # Positive Paare
    batch_neg = torch.stack([torch.stack([generate_embedding(img) for img in stack]) for stack in negatives])
    info_string += f"\nÄhnlichkeit batch anchor & positive: {get_distance(batch_anchors, batch_pos)}\n"
    info_string += f"Ähnlichkeit batch anchor & negative: {get_distance(batch_anchors, batch_neg)}\n"

    generate_plot(anchors, positives, negatives, save_to=os.path.join(results_folder, 'epochs_'+str(epochs)+'.png'), show_plot=False)
    
    with open(os.path.join(results_folder, 'info.txt'), "w", encoding="utf-8") as file:
        file.write(info_string)

def main(testing=False):
    ### SETTINGS ###
    anchor_drawing = 'house'
    negative_drawings = ['airplane', 'face', 'bathtub', 'cloud', 'mailbox']
    negative_drawings = ['airplane', 'face', 'bathtub']
    num_samples_per_category = 10
    epochs = 100
    margin = 120
    learning_rate = 0.0005
    metric = 'l2-norm'
    loss_function = 'ContrastiveLoss'
    ###

    if testing:
        info_txt = f"""### SETTINGS ###\nepochs = {epochs}\nmargin = {margin}\nlearning_rate = {learning_rate}\nmetric = {metric}\nloss_function = {loss_function}\n###\n\n"""
        anchor, pos, neg =  test_image_batches()
        results_folder = os.path.join('finetuning_tests', str(random.randint(1,100000)))
        os.makedirs(results_folder)
    else:
        info_txt = f"""### SETTINGS ###\nanchor_drawing = {anchor_drawing}\nnegative_drawings = {negative_drawings}\nnumber of samples per drawing category = {num_samples_per_category}\nepochs = {epochs}\nmargin = {margin}\nlearning_rate = {learning_rate}\nmetric = {metric}\nloss_function = {loss_function}\n###\n\n"""
        anchor, pos, neg = load_and_label(anchor_name=anchor_drawing, negatives_names=negative_drawings, num_samples=num_samples_per_category)
        results_folder = os.path.join('finetuning_results', loss_function+'_epochs='+str(epochs)+'_negcat='+str(len(negative_drawings))+'_samplespercat='+str(num_samples_per_category)+'_'+str(random.randint(1,100000)))
        os.makedirs(results_folder)


    # optimizer und loss function festlegen
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if loss_function == 'ContrastiveLoss':
        criterion = ContrastiveLoss(margin=margin)
    elif loss_function == 'TripletLoss':
        criterion = TripletLoss(margin=margin)


    ### INFO-LOG (vorher)
    dap1 = torch.norm(generate_embedding(anchor[0]) - generate_embedding(pos[0]), p=2)
    dap2 = torch.norm(generate_embedding(anchor[1]) - generate_embedding(pos[1]), p=2)
    dan11 = torch.norm(generate_embedding(anchor[0]) - generate_embedding(neg[0][0]), p=2)
    dan12 = torch.norm(generate_embedding(anchor[1]) - generate_embedding(neg[0][1]), p=2)
    dan21 = torch.norm(generate_embedding(anchor[0]) - generate_embedding(neg[1][0]), p=2)
    dan22 = torch.norm(generate_embedding(anchor[1]) - generate_embedding(neg[0][1]), p=2)
    print(dap1.item(), dap2.item(), dan11.item(), dan12.item(), dan21.item(), dan22.item())

    mdap = get_multi_l2distance(anchor, pos, elementwise=True)
    mdan1 = get_multi_l2distance(anchor, neg[0], elementwise=True)
    mdan2 = get_multi_l2distance(anchor, neg[1], elementwise=True)
    print(mdap.item(), mdan1.item(), mdan2.item())
    
    info_txt += 'before:\npos = '+str(dap1.item())+', '+str(dap2.item())+' \t(avg: '+str(mdap.item())+')\nneg1 = '+str(dan11.item())+', '+str(dan12.item())+' \t(avg: '+str(mdan1.item())+')\nneg2 = '+str(dan21.item())+', '+str(dan22.item())+' \t(avg: '+str(mdan2.item())+')\n\n'

    ### TRAINING
    start_time = time.time()
    for epoch in range(epochs):
        if epoch % 5 == 0:
            generate_plot(anchor, pos, neg, save_to=os.path.join(results_folder, 'epochs_'+str(epoch)+'.png'), show_plot=False)

        optimizer.zero_grad()

        if loss_function == 'ContrastiveLoss':
            anchor_emb = [generate_embedding(img) for img in anchor]
            pos_emb = [generate_embedding(img) for img in pos]
            neg_emb = [[generate_embedding(img) for img in stack] for stack in neg]

            sum_of_losses = 0
            n = 0
            for anch in anchor_emb:
                for p_emb in pos_emb:
                    s_loss = criterion(anch, p_emb, 0)
                    sum_of_losses += s_loss
                    n += 1
                for neg_stack in neg_emb:
                    for n_emb in neg_stack:
                        s_loss = criterion(anch, n_emb, 1)
                        sum_of_losses += s_loss
                        n += 1
            loss = sum_of_losses/n

        # NOCH NICHT FÜR STACK VON NEGATIVES AUSGELEGT -> neg[0] workaround nur für ersten stack
        elif loss_function == 'TripletLoss':
            mdap = get_multi_l2distance(anchor, pos, elementwise=True)
            mdan1 = get_multi_l2distance(anchor, neg[0], elementwise=True)
            loss = criterion(mdap, mdan1)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        info_txt += f"Epoch {epoch+1}, Loss: {loss.item()}\n"
    end_time = time.time()
    info_txt += '\t -> training-time: ' + str(round((end_time-start_time)/60,2)) + ' min.\n'
    ### INFO-LOG (nachher)
    dap1 = torch.norm(generate_embedding(anchor[0]) - generate_embedding(pos[0]), p=2)
    dap2 = torch.norm(generate_embedding(anchor[1]) - generate_embedding(pos[1]), p=2)
    dan11 = torch.norm(generate_embedding(anchor[0]) - generate_embedding(neg[0][0]), p=2)
    dan12 = torch.norm(generate_embedding(anchor[1]) - generate_embedding(neg[0][1]), p=2)
    dan21 = torch.norm(generate_embedding(anchor[0]) - generate_embedding(neg[1][0]), p=2)
    dan22 = torch.norm(generate_embedding(anchor[1]) - generate_embedding(neg[0][1]), p=2)
    print(dap1.item(), dap2.item(), dan11.item(), dan12.item(), dan21.item(), dan22.item())
    
    mdap = get_multi_l2distance(anchor, pos, elementwise=True)
    mdan1 = get_multi_l2distance(anchor, neg[0], elementwise=True)
    mdan2 = get_multi_l2distance(anchor, neg[1], elementwise=True)
    print(mdap.item(), mdan1.item(), mdan2.item())
    
    info_txt += '\nafter:\npos = '+str(dap1.item())+', '+str(dap2.item())+' \t(avg: '+str(mdap.item())+')\nneg1 = '+str(dan11.item())+', '+str(dan12.item())+' \t(avg: '+str(mdan1.item())+')\nneg2 = '+str(dan21.item())+', '+str(dan22.item())+' \t(avg: '+str(mdan2.item())+')\n\n'
 
    # save last plot
    generate_plot(anchor, pos, neg, save_to=os.path.join(results_folder, 'epochs_'+str(epoch)+'.png'), show_plot=False)

    # save infolog
    with open(os.path.join(results_folder, 'info.txt'), "w", encoding="utf-8") as file:
        file.write(info_txt)

if __name__ == '__main__':
    main(testing=False)