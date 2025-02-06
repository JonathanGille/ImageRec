import pandas as pd
import os

scans_folder = os.path.join('scans','parkhaus_melaten')
vMRTs_folder = os.path.join('vMRTs','parkhaus_melaten_v1')

def labels(key):
    labels = {
        'Complicated': -1,
        'None': 0,
        'Widerlager_West': 1, 
        'Widerlager_Ost': 2,
        'Deck': 3,
        'Seitenansicht': 4,
        'Draufsicht': 5,
    }

    label_32 = {
        'ANSICHT SEITE SÜD, M 1  100': 3,
        'ANSICHT WIDERLAGER, M 1 50': 2,
        'DETAIL  A , M 1 25': 0,
        'DETAIL  B , M=1 25': 0,
        'DRAUFSICHT AUF .... N': 5,
        'KAPPE AUF DER STÜTZWAND': 0,
        'LÄNGSSCHNITT A-A, M 1 100': 4,
        'Raumfuge (RF) n. RiZ Fug 1 Bild 2': 0,
        'REGELQUERSCHNITT': 3,
        'SICHTFLÄCHENSCHALUNG   ÜBERBAU': 0,
        'Sickerschacht': 0,
    }

    #df_label_32 = pd.DataFrame(scans_32_label, index=label).T

    label_33 = {
        'Absteckpunkte ': 0,
        'Absteckpunkte _1': 0,
        'In vertragticher Hinsicht einschließlich der Übereinstimmung mit den': 0,
        'ZEICHNUNGEN UND DER BAUAUSFÜHRUNG ÜBEREIN': 0,
    }
    #df_label_33 = pd.DataFrame(label_33, index=label).T


    label_34 = {
        'ABSTECKPUNKTE': 0,
        'Achse   0': 1,
        'ANSICHT WIDERLAGER, M 1 50': 2,
        'ANSICHT WIDERLAGER, M 1 50_1': 2,
        'DATUM  .': 0,
        'DETAIL  A , M 1 10': 0,
        'DETAIL  A , M 1 10_1': 0,
        'DETAIL  C , M=1 10': 0,
        'DETAIL  C , M=1 10_1': 0,
        'DRAUFSICHT, M 1 50': -1,
        'DRAUFSICHT, M 1 50_1': -1,
        'DRAUFSICHT, M 1 50_2': -1,
        'FLÜGELSCHNITT, M 1 50': 0,
        'Lage  der Arbeits-und': 0,
        'LÄNGSSCHNITT A-A, M 1 50': 0,
        'Sauberkeilsschicht € 8 10': 1,
    }

    #df_label_34 = pd.DataFrame(label_34, index=label).T

    label_35 = {
        'ANSICHT WIDERLAGER .... M 1 25': 3,
        'ANSICHT WIDERLAGER .... M 1 25_1': 3,
        'DETAIL  A  M 1 10': 0,
        'DIESE ZEICHNUNG STIMMT MIT DEN FREIGEGEBENEN': 0,
        'DRAUFSICHT, M 1 50': 5,
        'LÄNGSSCHNITT A-A, M 1 50': 4,
        'REGELQUERSCHNITT , M 1 25': 0,
        'REGELQUERSCHNITT , M 1 25_1': 0,
        'REGELQUERSCHNITT , M 1 25_2': 3,
        'SICHTBARE RECHTWINKLIGE UND SPITZE KANTEN': 0,
        'Übergangskonstrukion': 0,
    }
        
    #df_label_35 = pd.DataFrame(label_35, index=label).T

    label_36 = {
        'Absteckpunkte  (6),(9(10)': 0,
        'DETAIL  A , M 1 10': 0,
        'DIESE ZEICHNUNG STIMMT MIT DEN FREIGEGEBENEN': 0,
        'DRAUFSICHT, M 1 50': 0,
        'LÄNGSSCHNITT A-A, M 1 50': 0,
        'QUERSCHNITT B-B. M 1 50': 0,
    }

    #df_label_36 = pd.DataFrame(label_36, index=label).T

    label_37 = {
        'ANSCHLUSSBEWEHRUNG': 0,
        'ANSCHLUSSBEWEHRUNG_1': 0,
        'Biegerollendurchmesser': 0,
        'DRAUFSICHT AUF. ... M 1 50': 0,
        'FLÜGELSCHNITT, M 1 50': 0,
        'OBERE LAGE': 0,
        'OBERE LAGE_1': 0,
        'RUNDSTAHL - STÜCKLISTE': 0,
        'SEITE OST': 0,
        'SEITE OST_1': 0,
        'SEITE WEST': 0,
        'SEITE WEST_1': 0,
        'ZEICHNUNGEN UND DER BAUAUSFÜHRUNG ÜBEREIN': 0,
        'Zut(7)¢ 25 s': 0,
        'Zut(7)¢ 25 s_1': 0,
    }

    # df_label_37 = pd.DataFrame(label_37, index=label).T

    label_38 = {
        'Achise   0': 1,
        'ANSICHT FLÜGEL': -1,
        'Biegerollendurchmess': 0,
        'DETAIL A , M=1 25': 0,
        'DIESE ZEICHNUNG STIMMT MIT DEN FREIGEGEBENC': 0,
        'Draufsicht': 0,
        'ES (48) 31 Ø 14 $=15': -1,
        'FLÜGELSCHNITT, M 1 50': 0,
        'GRUNDRISS, M 1 50': 0,
        'GRUNDRISS, M 1 50_1': 0,
        'RUNDSTAHL - STÜCKLISTE': 0,
        'SCHNITT A-A. M 1 50': 2,
        'Tabelle zu (33)': 0,
        'Tabelle zu': 0,
        'Tabelle zu_1': 0,
        'WIDERLAGER ACHSE 0, M 1 50': 0,
        'WIDERLAGER ACHSE 50, M 1 50': 2,
    }


    dic = {
        '32': label_32,
        '33': label_33,
        '34': label_34,
        '35': label_35,
        '36': label_36,
        '37': label_37,
        '38': label_38,
    }
    return dic[key]


# for name in os.listdir(os.path.join(scans_folder,'38')):
#     if name.endswith('.jpg') or name.endswith('.png'):
#         print(name[:-4])
