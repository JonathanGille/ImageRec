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
    
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, pos_dist, neg_dist):
        # Berechne Triplet Loss
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0)

        return loss.mean()
    

def sketchy_image_batches(anchor_name, negatives_names, num_samples=40, as_objects=False):
    img_folder = 'training_data'
    anchor_folder = os.path.join(img_folder,anchor_name)
    negative_folders = [os.path.join(img_folder, neg_name) for neg_name in negatives_names]


    anchor_images = get_images(anchor_folder)[0:num_samples*2]
    stack_of_negative_images = [get_images(neg_folder)[0:num_samples*2] for neg_folder in negative_folders]

    for img in anchor_images:
        img.img = load_image(img.path) # für die richtige batch-dimension lokale load-funktion verwenden
        #img.emb = generate_embedding(img.img)
        img.label = anchor_name
    for i in range(len(stack_of_negative_images)):
        for img in stack_of_negative_images[i]:
            img.img = load_image(img.path) # für die richtige batch-dimension lokale load-funktion verwenden
            #img.emb = generate_embedding(img.img)
            img.label = negatives_names[i]

    batch_length = len(anchor_images)//2

    if as_objects:
        anchors = [img for img in anchor_images[:1]] # erste hälfte von houses sind die anchor bilder
        positives = [img for img in anchor_images[1:5]] # zweites hälfte (genauso lang wie erste hälfte) von houses sind die positiven bilder
        negatives = [[img for img in stack[:6]] for stack in stack_of_negative_images]# airplanes sind die negativen (genauso lang wie die anchor bilder)
    else:
        anchors = [img.img for img in anchor_images[:1]] # erste hälfte von houses sind die anchor bilder
        positives = [img.img for img in anchor_images[1:5]] # zweites hälfte (genauso lang wie erste hälfte) von houses sind die positiven bilder
        negatives = [[img.img for img in stack[:6]] for stack in stack_of_negative_images]# airplanes sind die negativen (genauso lang wie die anchor bilder)

    return anchors, positives, negatives

def generate_plot(anchors, positives, negatives, save_to=None, show_plot=True, only_apn_label=True):
    if only_apn_label:
        all_embeddings = [generate_embedding(anch) for anch in anchors] + [generate_embedding(pos) for pos in positives]
        for stack in negatives:
            all_embeddings = all_embeddings + [generate_embedding(neg) for neg in stack]
    else:
        all_embeddings = [generate_embedding(anch.img) for anch in anchors] + [generate_embedding(pos.img) for pos in positives]
        for stack in negatives:
            all_embeddings = all_embeddings + [generate_embedding(neg.img) for neg in stack]        

    if only_apn_label:
        all_label = ['anchor' for i in range(len(anchors))] + ['positive' for i in range(len(positives))]
        for stack in negatives:
            all_label = all_label + ['negative' for i in range(len(stack))]
    else:
        all_label = [str(a.label)+' (anchor)' for a in anchors] + [str(p.label)+' (pos.)' for p in positives]
        for stack in negatives:
            all_label = all_label + [n.label+' (neg.)' for n in stack]

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

def bridge_sections_image_batches(anchor_name, ignore=None, as_objects=False):
    img_folder = 'bridge_sections_seperated'
    wireframes_folder = os.path.join(img_folder,'wireframes')
    scans_folder = os.path.join(img_folder,'scans')
    
    anchor_folder = os.path.join(wireframes_folder, anchor_name)
    positives_folder = os.path.join(scans_folder, anchor_name)
    negative_parts = [p for p in os.listdir(scans_folder) if p != anchor_name]
    for ign in ignore:
        negative_parts = [p for p in negative_parts if p != ign]
    negative_folders = [os.path.join(scans_folder, p) for p in negative_parts]

    anchor_images = get_images(anchor_folder)
    positive_images = get_images(positives_folder)
    stack_of_negative_images = [get_images(neg_folder) for neg_folder in negative_folders]

    for img in anchor_images:
        img.img = load_image(img.path) # für die richtige batch-dimension lokale load-funktion verwenden
        #img.emb = generate_embedding(img.img)
        img.label = anchor_name
    for img in positive_images:
        img.img = load_image(img.path) # für die richtige batch-dimension lokale load-funktion verwenden
        #img.emb = generate_embedding(img.img)
        img.label = anchor_name
    for i in range(len(stack_of_negative_images)):
        for img in stack_of_negative_images[i]:
            img.img = load_image(img.path) # für die richtige batch-dimension lokale load-funktion verwenden
            #img.emb = generate_embedding(img.img)
            img.label = negative_parts[i]


    if as_objects:
        anchors = [img for img in anchor_images] # erste hälfte von houses sind die anchor bilder
        positives = [img for img in positive_images] # zweites hälfte (genauso lang wie erste hälfte) von houses sind die positiven bilder
        negatives = [[img for img in stack] for stack in stack_of_negative_images]# airplanes sind die negativen (genauso lang wie die anchor bilder)
    else:
        anchors = [img.img for img in anchor_images] # erste hälfte von houses sind die anchor bilder
        positives = [img.img for img in positive_images] # zweites hälfte (genauso lang wie erste hälfte) von houses sind die positiven bilder
        negatives = [[img.img for img in stack] for stack in stack_of_negative_images] # airplanes sind die negativen (genauso lang wie die anchor bilder)

    return anchors, positives, negatives

def main(use_case='sketchy', anchor=None):
    ### SETTINGS ###
    anchor_drawing = 'house'
    negative_drawings = ['airplane', 'face', 'bathtub', 'cloud', 'mailbox']
    num_samples_per_category = 10
    epochs = 70
    margin = 120
    learning_rate = 0.001
    metric = 'l2-norm'
    loss_function = 'ContrastiveLoss'
    decay_step = 25
    decay_rate = 0.005
    ###

    if use_case == 'testing':
        info_txt = f"""### SETTINGS ###\nepochs = {epochs}\nmargin = {margin}\nlearning_rate = {learning_rate}\nmetric = {metric}\nloss_function = {loss_function}\n###\n\n"""
        anchor, pos, neg =  test_image_batches()
        anchor_objs, pos_objs, neg_objs = anchor, pos, neg
        label_only_apn=True #(nur anchor, positive, negative als label in plots)
        results_folder = os.path.join('finetuning_tests', str(random.randint(1,100000)))
        os.makedirs(results_folder)
    elif use_case == 'sketchy':
        info_txt = f"""### SETTINGS ###\nanchor_drawing = {anchor_drawing}\nnegative_drawings = {negative_drawings}\nnumber of samples per drawing category = {num_samples_per_category}\nepochs = {epochs}\nmargin = {margin}\nlearning_rate = {learning_rate}\nmetric = {metric}\nloss_function = {loss_function}\n###\n\n"""
        anchor, pos, neg = sketchy_image_batches(anchor_name=anchor_drawing, negatives_names=negative_drawings, num_samples=num_samples_per_category)
        anchor_objs, pos_objs, neg_objs = sketchy_image_batches(anchor_name=anchor_drawing, negatives_names=negative_drawings, num_samples=num_samples_per_category, as_objects=True)
        label_only_apn = False #(in den plots werden alle kategorien farblich unterschieden und beschriftet)
        results_folder = os.path.join('finetuning_results', loss_function+'_epochs='+str(epochs)+'_negcat='+str(len(negative_drawings))+'_samplespercat='+str(num_samples_per_category)+'_'+str(random.randint(1,100000)))
        os.makedirs(results_folder)
    elif use_case == 'bridge_sections':
        anchor_name = anchor
        info_txt = f"""### SETTINGS ###\nanchor_drawing = {anchor_drawing}\nnegative_drawings = {negative_drawings}\nnumber of samples per drawing category = {num_samples_per_category}\nepochs = {epochs}\nmargin = {margin}\nlearning_rate = {learning_rate}\nmetric = {metric}\nloss_function = {loss_function}\n###\n\n"""
        anchor, pos, neg = bridge_sections_image_batches(anchor_name=anchor_name, ignore=['none'], as_objects=False)
        anchor_objs, pos_objs, neg_objs = bridge_sections_image_batches(anchor_name=anchor_name, ignore=['none'], as_objects=True)
        label_only_apn = False #(in den plots werden alle kategorien farblich unterschieden und beschriftet)
        results_folder = os.path.join('finetuning_results', 'BS(anchor='+anchor_name+')_'+loss_function+'_epochs='+str(epochs)+'_lr='+str(learning_rate)+'_'+str(random.randint(1,100000)))
        os.makedirs(results_folder)
    else:
        print('##### ERROR: Invalid use case !!! #####')
        return


    # optimizer und loss function festlegen
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if loss_function == 'ContrastiveLoss':
        criterion = ContrastiveLoss(margin=margin)
    elif loss_function == 'TripletLoss':
        criterion = TripletLoss(margin=margin)


    ### INFO-LOG (vorher)
    mdap = get_multi_l2distance(anchor, pos, elementwise=True)
    info_txt += 'before:\npos = (avg: '+str(mdap.item())+')\n'
    for ne in neg:
        mdan = get_multi_l2distance(anchor, ne, elementwise=True)
        info_txt += 'neg = (avg: '+str(mdan.item())+')\n'
    info_txt += '\n'
    
    ### TRAINING
    start_time = time.time()
    for epoch in range(epochs):
        if epoch % 5 == 0:
            generate_plot(anchor_objs, pos_objs, neg_objs, save_to=os.path.join(results_folder, 'epochs_'+str(epoch)+'.png'), show_plot=False, only_apn_label=label_only_apn)

        if epoch == 50:
            learning_rate = 0.001
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        if epoch == 100:
            learning_rate = 0.0005
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        if epoch == 150:
            learning_rate = 0.0001
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        if epoch == 200:
            learning_rate = 0.00005
            optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

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
    mdap = get_multi_l2distance(anchor, pos, elementwise=True)
    info_txt += '\nafter:\npos = (avg: '+str(mdap.item())+')\n'
    for ne in neg:
        mdan = get_multi_l2distance(anchor, ne, elementwise=True)
        info_txt += 'neg = (avg: '+str(mdan.item())+')\n'

    # save last plot
    generate_plot(anchor_objs, pos_objs, neg_objs, save_to=os.path.join(results_folder, 'epochs_'+str(epoch)+'.png'), show_plot=False, only_apn_label=label_only_apn)

    # save infolog
    with open(os.path.join(results_folder, 'info.txt'), "w", encoding="utf-8") as file:
        file.write(info_txt)

if __name__ == '__main__':
    main(use_case='bridge_sections', anchor='seitenansicht')
    main(use_case='bridge_sections', anchor='draufsicht')
    main(use_case='bridge_sections', anchor='deck')