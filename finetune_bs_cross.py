import os
import random
import time
import numpy as np

from similarity_search import image_similarity, get_images, get_embedding
from embeddings_manager import plot_embeddings

import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as functional
import matplotlib.pyplot as plt


# Basis-Modell laden
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

def load_images_of_folder(folder):
    images = get_images(folder)
    for img in images:
        img.img = load_image(img.path)
    return images

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

def embedding(img, model):
    features = model.forward_features(img)
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
    

def sketchy_image_batches(anchor_name, negatives_names, num_samples=40, as_objects=True):
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

def generate_plot(anchors, positives, negatives, save_to=None, show_plot=True, only_apn_label=True, normalize=True):
    if only_apn_label:
        all_embeddings = [anch.norm_emb.clone() for anch in anchors] + [pos.norm_emb.clone() for pos in positives]
        for stack in negatives:
            all_embeddings = all_embeddings + [neg.norm_emb for neg in stack]
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

    if normalize:
        # normalize embeddings by largest l2-norm of all embeddings
        maxl2 = max_l2(all_embeddings)
        all_embeddings = [emb / maxl2 for emb in all_embeddings]

    plot_embeddings(all_embeddings, all_label, save_to=save_to, show_plot=show_plot)

def l2_distance(img1, img2):
    return torch.norm(img1 - img2, p=2)

def get_multi_l2distance(batch1, batch2, elementwise=True):  # expects to lists of images and gives back the normed distance between those two batches
    diff = torch.stack([generate_embedding(b1.img) for b1 in batch1]) - torch.stack([generate_embedding(b2.img) for b2 in batch2])   # shape t([2, 1024])
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

def bridge_sections_image_batches(anchor_name, ignore=None, as_objects=True):
    img_folder = 'bridge_sections_labelled'
    wireframes_folder = os.path.join(img_folder,'wireframes')
    scans_folder = os.path.join(img_folder,'scans')
    
    anchor_folder = os.path.join(wireframes_folder, anchor_name)
    positives_folder = os.path.join(scans_folder, anchor_name)
    # alle folder außer dem anchor folder
    negative_parts = [p for p in os.listdir(scans_folder) if p != anchor_name]
    if ignore != None and type(ignore) == list:
        # folder die in ignore gelistet sind ignorieren
        for ign in ignore:
            negative_parts = [p for p in negative_parts if p != ign]
    negative_folders = [os.path.join(scans_folder, p) for p in negative_parts]

    # bilder laden (als c_image objekt) 
    anchor_images = get_images(anchor_folder, load_image=False)
    positive_images = get_images(positives_folder, load_image=False)
    stack_of_negative_images = [get_images(neg_folder, load_image=False) for neg_folder in negative_folders]


    for img in anchor_images:
        img.img = load_image(img.path) # für die richtige batch-dimension lokale load-funktion verwenden
        #img.emb = generate_embedding(img.img.clone())
        img.label = anchor_name
    for img in positive_images:
        img.img = load_image(img.path) # für die richtige batch-dimension lokale load-funktion verwenden
        #img.emb = generate_embedding(img.img.clone())
        img.label = anchor_name
    for i in range(len(stack_of_negative_images)):
        for img in stack_of_negative_images[i]:
            img.img = load_image(img.path) # für die richtige batch-dimension lokale load-funktion verwenden
            #img.emb = generate_embedding(img.img.clone())
            img.label = negative_parts[i]
    print('-> ##### c_images created (anchor='+anchor_name+') #####')

    # # max l2-norm aller embeddings berechnen
    # all_images = [img for img in anchor_images] + [img for img in positive_images] + [img for stack in stack_of_negative_images for img in stack]
    # all_embeddings = torch.stack([img.emb.clone() for img in all_images])
    # max_l2 = torch.norm(all_embeddings, p=2, dim=1, keepdim=True).max()
    # # embeddings mit maximaler l2-norm normalisieren (l2-norm aller embeddings auf Werte zwischen 0-1)
    # for img in all_images:
    #     img.norm_emb = img.emb / max_l2
    # print('-> ##### embeddings mormalized (c_image.norm_emb) #####')

    anchors = [c_img for c_img in anchor_images] # erste hälfte von houses sind die anchor bilder
    positives = [c_img for c_img in positive_images] # zweites hälfte (genauso lang wie erste hälfte) von houses sind die positiven bilder
    negatives = [[c_img for c_img in stack] for stack in stack_of_negative_images]# airplanes sind die negativen (genauso lang wie die anchor bilder)

    return anchors, positives, negatives

def load_and_test(model_name=None):
    if model_name == None:
        model_name = 'BS(cross)_ContrastiveLoss_epochs=100_lr=0.001_41921'
    loaded_model = timm.create_model('convnext_base', pretrained=True)
    loaded_model.load_state_dict(torch.load(os.path.join('finetuned_models', model_name+'.pth'), weights_only=True))
    # loaded_model = torch.load(os.path.join('finetuned_models', 'BS(cross)_ContrastiveLoss_epochs=2_lr=0.001_30451.pth'), weights_only=False)
    img1 = load_image(os.path.join('bridge_sections_labelled', 'wireframes', 'deck', 'qs_w2e_middle.png'))
    img2 = load_image(os.path.join('bridge_sections_labelled', 'scans', 'deck', 'ANSICHT WIDERLAGER .... M 1 25.png'))
    img3 = load_image(os.path.join('bridge_sections_labelled', 'scans', 'seitenansicht', 'Achse  0.png'))
    features = loaded_model.forward_features(img1)
    emb1 = torch.mean(features, dim=[2, 3]).squeeze(0) # Global Average Pooling
    features = loaded_model.forward_features(img2)
    emb2 = torch.mean(features, dim=[2, 3]).squeeze(0) # Global Average Pooling
    features = loaded_model.forward_features(img3)
    emb3 = torch.mean(features, dim=[2, 3]).squeeze(0) # Global Average Pooling

    d12 = torch.norm(emb1 - emb2, p=2)
    d13 = torch.norm(emb1 - emb3, p=2)
    print(d12.item())
    print(d13.item())

def max_l2(all_embeddings):
    # max l2-norm aller embeddings berechnen
    max_l2 = torch.norm(torch.stack(all_embeddings), p=2, dim=1, keepdim=True).max()
    return max_l2

def load_model(model_name):
    loaded_model = timm.create_model('convnext_base', pretrained=True)
    loaded_model.load_state_dict(torch.load(os.path.join('finetuned_models', model_name+'.pth'), weights_only=True))
    return loaded_model

def bridge_sections_image_dictionarys(ignore=None):
    img_folder = 'bridge_sections_labelled'
    wireframes_folder = os.path.join(img_folder,'wireframes')
    scans_folder = os.path.join(img_folder,'scans')

    scans_dic = {}
    for scan_type in os.listdir(scans_folder):
        if scan_type in ignore:
            continue
        scans_dic[scan_type] = load_images_of_folder(os.path.join(scans_folder, scan_type))
        for img in scans_dic[scan_type]:
            img.label = scan_type

    wireframes_dic = {}
    for wireframe_type in os.listdir(scans_folder):
        if wireframe_type in ignore:
            continue
        wireframes_dic[wireframe_type] = load_images_of_folder(os.path.join(wireframes_folder, wireframe_type))
        for img in wireframes_dic[wireframe_type]:
            img.label = wireframe_type

    return wireframes_dic, scans_dic

def plot_distances(model, show=True, save_imgs_to=None):
    wireframes_dic, scans_dic = bridge_sections_image_dictionarys(ignore=['none'])
    all_wireframes = []
    for key, lst in wireframes_dic.items():
        all_wireframes += lst

    for key, scans in scans_dic.items():
        for scan in scans:
            colors = ['green' if wf.label == scan.label else 'red' for wf in all_wireframes]
            wf_names = [wf.name for wf in all_wireframes]
            distances = [l2_distance(embedding(scan.img, model), embedding(wireframe.img, model)).item() for wireframe in all_wireframes]
            plt.scatter(wf_names, distances, c=colors)
            plt.xlabel('wireframes')
            #  beschriftung x-achse rotieren für lesbarkeit
            plt.xticks(rotation=30, ha='right')
            plt.ylabel('scan: '+scan.name)

            plt.tight_layout()
            if save_imgs_to != None:
                save_folder = os.path.join(save_imgs_to, key)
                os.makedirs(save_folder, exist_ok=True)
                plt.savefig(os.path.join(save_folder, scan.name+'.png'), format='png')
            if show:
                plt.show()
            else:
                plt.clf()
        
def main(anchor_names, ignore=None, save_model=True):
    ### SETTINGS ###
    epochs = 2
    margin = 1
    learning_rate = 0.001
    metric = 'l2-norm'
    loss_function = 'ContrastiveLoss'
    decay_step = 25
    decay_rate = 0.005
    ###

    info_txt = f"""### SETTINGS ###\nanchors = {anchor_names}\nepochs = {epochs}\nmargin = {margin}\nlearning_rate = {learning_rate}\nmetric = {metric}\nloss_function = {loss_function}\n###\n\n"""
    label_only_apn = False #(in den plots werden alle kategorien farblich unterschieden und beschriftet)
    results_folder = os.path.join('finetuning_results', 'BS(cross)_'+loss_function+'_epochs='+str(epochs)+'_lr='+str(learning_rate)+'_'+str(random.randint(1,100000)))
    os.makedirs(results_folder)
    print('creating batches...')
    apn_obj_batches = [bridge_sections_image_batches(anchor_name=anchor_name, ignore=ignore, as_objects=True) for anchor_name in anchor_names]


    # optimizer und loss function festlegen
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if loss_function == 'ContrastiveLoss':
        criterion = ContrastiveLoss(margin=margin)
    elif loss_function == 'TripletLoss':
        criterion = TripletLoss(margin=margin)


    # ### INFO-LOG (vorher)
    # info_txt += '--> before\n'
    # for anchor, pos, neg in apn_obj_batches:
    #     mdap = get_multi_l2distance(anchor, pos, elementwise=True)
    #     info_txt += '('+str(anchor[0].name)+'):\npos = (avg: '+str(mdap.item())+')\n'
    #     for ne in neg:
    #         mdan = get_multi_l2distance(anchor, ne, elementwise=True)
    #         info_txt += 'neg = (avg: '+str(mdan.item())+')\n'
    #     info_txt += '\n'
    
    ### TRAINING
    print('start training...')
    loss_log = []
    mean_log_pos = []
    mean_log_neg = []
    start_time = time.time()
    for epoch in range(epochs):
        loss_mean = []
        mlp = []
        mln = []
        for anchor, pos, neg in apn_obj_batches:
            if epoch % 5 == 0:
                generate_plot(anchor, pos, neg, save_to=os.path.join(results_folder, 'epochs_'+str(epoch)+'_'+anchor[0].name+'.png'), show_plot=False, only_apn_label=label_only_apn)
                

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
                anchor_emb = [generate_embedding(img.img) for img in anchor]
                pos_emb = [generate_embedding(img.img) for img in pos]
                neg_emb = [[generate_embedding(img.img) for img in stack] for stack in neg]

                # normalize embeddings bei the maximum l2-distance of all embeddings
                all_embeddings = anchor_emb + pos_emb + [emb for stack in neg_emb for emb in stack]
                maxl2 = max_l2(all_embeddings)
                anchor_emb_norm = [emb / maxl2 for emb in anchor_emb]
                pos_emb_norm = [emb / maxl2 for emb in pos_emb]
                neg_emb_norm = [[emb / maxl2 for emb in stack] for stack in neg_emb]


                sum_of_losses = 0
                mean_pos_sum = 0
                mean_neg_sum = 0
                n = 0
                n_pos = 0
                n_neg = 0
                for anch in anchor_emb_norm:
                    for p_emb in pos_emb_norm:
                        s_loss = criterion(anch, p_emb, 0)
                        sum_of_losses = sum_of_losses + s_loss
                        n += 1
                        l2 = torch.norm(anch-p_emb, p=2).item()
                        mean_pos_sum += l2
                        n_pos += 1
                    for neg_stack in neg_emb_norm:
                        for n_emb in neg_stack:
                            s_loss = criterion(anch, n_emb, 1)
                            sum_of_losses = sum_of_losses + s_loss
                            n += 1
                            l2 = torch.norm(anch-n_emb, p=2).item()
                            mean_neg_sum += l2
                            n_neg += 1
                loss = sum_of_losses/n
                mlp.append(mean_pos_sum / n_pos)
                mln.append(mean_neg_sum / n_neg)


            # NOCH NICHT FÜR STACK VON NEGATIVES AUSGELEGT -> neg[0] workaround nur für ersten stack
            elif loss_function == 'TripletLoss':
                mdap = get_multi_l2distance(anchor, pos, elementwise=True)
                mdan1 = get_multi_l2distance(anchor, neg[0], elementwise=True)
                loss = criterion(mdap, mdan1)
            # for the loss_log (to get the mean loss for all anchor combination)
            loss_mean.append(loss.item())
            # for training
            loss.backward()
            optimizer.step()
            if epoch == epochs-1:
                generate_plot(anchor, pos, neg, save_to=os.path.join(results_folder, 'epochs_'+str(epoch)+'_'+anchor[0].name+'.png'), show_plot=False, only_apn_label=label_only_apn)


        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        info_txt += f"Epoch {epoch+1}, Loss: {loss.item()}\n"
        loss_log.append(np.mean(loss_mean))
        mean_log_pos.append(np.mean(mlp))
        mean_log_neg.append(np.mean(mln))

    end_time = time.time()
    info_txt += '\t -> training-time: ' + str(round((end_time-start_time)/60,2)) + ' min.\n'

    # ### INFO-LOG (nachher)    
    # info_txt += '\n--> after\n'
    # for anchor, pos, neg in apn_obj_batches:
    #     mdap = get_multi_l2distance(anchor, pos, elementwise=True)
    #     info_txt += '('+str(anchor[0].name)+'):\npos = (avg: '+str(mdap.item())+')\n'
    #     for ne in neg:
    #         mdan = get_multi_l2distance(anchor, ne, elementwise=True)
    #         info_txt += 'neg = (avg: '+str(mdan.item())+')\n'
    #     info_txt += '\n'

    # # save last plot
    generate_plot(anchor, pos, neg, save_to=os.path.join(results_folder, 'epochs_'+str(epoch)+'.png'), show_plot=False, only_apn_label=label_only_apn)

    # save infolog
    with open(os.path.join(results_folder, 'info.txt'), "w", encoding="utf-8") as file:
        file.write(info_txt)

    # save loss_log as pyplot
    plt.plot(list(range(len(loss_log))), loss_log)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(os.path.join(results_folder, 'loss.png'), format='png')
    plt.clf()

    # save means of positive pairs distances and negative pairs distances
    plt.plot(list(range(len(mean_log_pos))), mean_log_pos, label='positive', color='green')
    plt.plot(list(range(len(mean_log_pos))), mean_log_neg, label='negative', color='red')
    plt.xlabel('epochs')
    plt.ylabel('mean of l2-distances')
    plt.savefig(os.path.join(results_folder, 'mean_distances.png'), format='png')
    plt.clf()

    # save model
    if save_model:
        model_save_path = 'finetuned_models'
        custom_name = 'BS(cross)_'+loss_function+'_epochs='+str(epochs)+'_lr='+str(learning_rate)+'_'+str(random.randint(1,100000))
        # torch.save(model, os.path.join(model_save_path, custom_name+'.pth'))
        torch.save(model.state_dict(), os.path.join(model_save_path, custom_name+'.pth'))

if __name__ == '__main__':
    main(anchor_names=['seitenansicht', 'draufsicht', 'deck', 'widerlager'], ignore=['none'], save_model=True)
    # load_and_test()
    # bridge_sections_image_dictionarys(ignore=['none'])
    # model = load_model('BS(cross)_ContrastiveLoss_epochs=100_lr=0.001_41921')
    # plot_distances(model, show=False, save_imgs_to=os.path.join('results', 'finetuned', str(random.randint(1,100000))))
    # a,p,n = bridge_sections_image_batches('deck')
    # print(torch.norm(a[0].norm_emb, p=2))
    # print(torch.norm(p[0].norm_emb, p=2))
    # print(torch.norm(n[0][0].norm_emb, p=2))