import pandas as pd
import os

scans_folder = os.path.join('scans','parkhaus_melaten')
wireframes_folder = os.path.join('wireframes','all')

def get_labels(key):
    label_assigment_dictionary = {
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
        'DRAUFSICHT, M 1 50': 2,
        'DRAUFSICHT, M 1 50_1': -1,
        'DRAUFSICHT, M 1 50_2': 1,
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
        'ES (48) 31 Ø 14 $=15': 2,
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

    label_39 = {
    'Bewehrung Fundamente': 0,
    'Biegerottendurchmesser dbr': 0,
    'Biegerottendurchmesser dbr_1': 0,
    'DATUM  . .': 0,
    'DETAIL A , M 1 25': 0,
    'DRAUFSICHT AUF. . .. M 1 50': 0,
    'LÄNGSSCHNITT A-A, M 1 50': 0,
    'ME  LS (15) 7 ¢ 10 sy=1.00': 0,
    'ME  LS (15) 7 ¢ 10 sy=1.00_1': 0,
    'QUERSCHNITTE ...': 0,
    'RUNDSTAHL - STÜCKLISTE': 0,
    'Tabelle zu': 0,
    'ZuL (25 )27)': 0,
    }

    label_40 = {
    'AUSTAUSCHBARE ELASTOMERE LAGER DER FA. SCHREIBER, MAINHARDT': 0,
    'Bauwerks-Nr.': 0,
    'DRAUFSICHT, M 1 25': 0,
    'DRAUFSICHT. M 1 50': 0,
    'DRAUFSICHT. M 1 50_1': 0,
    'DRAUFSICHT. M 1 50_2': 2,
    'EINBAU DER LAGER (UNTERE ANKERPLATTE)': 0,
    'Hauptmaßen des Entwurfes geprüft': 0,
    'Lager Nr .   50 1': 0,
    'Lager Nr .   50 1_1': 0,
    'Lager Nr .  50 2': 0,
    'LAGERSYSTEMSKIZZE': 0,
    }

    label_41 = {
    'Achse  0': 4,
    'Anspannstelle Seite Ost, M 1 25': 3,
    'Anspannstelle und Festankerstet le vertausch! b.beiunstellund. Feb 2011. M': 0,
    'ANSPANNSTELLE, M 1 25': 0,
    'DIESE ZEICHNUNG STIMMT MIT DEN FREIGEGEBENEN': 0,
    'Festanker  Fe': 0,
    'Festankerstelle Seite West ,': 3,
    'FESTANKERSTELLE, M 1 25': 0,
    'LÄNGSVORSPANNUNG': 0,
    'PASZE  IN CM GELTEN NUR FU': 5,
    'QUERSCHNITT IN PUNKT  0 , M 1 25': 3,
    'RUNDSTAHL - STÜCKLISTE': 0,
    'Stahlbeton': 0,
    'Tabelle zu (11': 0,
    'Tabelle zu': 0,
    'Tabelle zu_1': 0,
    'ZUSATZBEWEHRUNG M=1 10': 0,
    }

    label_42 = {
    'DIESE ZEICHNUNG STIMMT MIT DEN FREIGEGEBENEN': 0,
    'DRAUFSICHT, M 1 50': 5,
    'ITLÄNGE ..': 0,
    'Legende': 0,
    'LÄNGSSCHNITT A-A, M 1 50': 4,
    'QUERSCHNITT B-B , M 1 25': 3,
    'RUNDSTAHL': 0,
    'SCHNITT VOR WIDERLAGER.': 3,
    'SPALTZUGBEWEHRUNG': 0,
    'SPALTZUGBEWEHRUNG_1': 3,
    'Tabelle zu (37': 0,
    'Tabelle zu': 0,
    'Tabelle zu_1': 0,
    'Tabelle zu_2': 0,
    }

    label_43 = {
        'Abdeckblech 5 mm': 0,
        'Draufsicht Achse 50 M. 1 10': 0,
        'EDV-Nr .': 0,
        'EINSTELLMASS BEI': 0,
        'Hutmutter Wsf. 1.4301': 0,
        'Schnitt A-A': 3,
        'Schnitt A-A_1': 0,
        'Schnitt B-B M. 1 2': 0,
        'Schnitt C-C M. 1 2': 0,
        'Seite WL': 0,
        'Seite WL_1': 0,
        'STÜCKLISTE': 0,
    }

    label_44 = {
        'Bauwerks-Nr .': 0,
        'Brückenquerrichtung --- +': 0,
        'Brückenquerrichtung --- +_1': 0,
        'Brückenquerrichtung --- +_2': 0,
        'Gleit blech rundum': 0,
        'HV - Verschraubung': 0,
        'Lagerschema': 0,
        'Montagehalterung (Prinzip) @': 0,
        'Sechskantmutter': 0,
        'Sechskantmutter_1': 0,
        'Typenschild Pos. 100 (Prinzip)': 0,
    }

    label_45 = {
    'Auftraggeber': 0,
    'Brückenquerrichtung': 0,
    'Brückenquerrichtung_1': 0,
    'Fritz Meyer GmbH.': 0,
    'In vertraglicher Hinsicht einschließlich der Übereinstimmung mit den': 0,
    'Lagerschema': 0,
    'Lagerschema_1': 0,
    'Montagehalterung (Prinzip) @': 0,
    'S235JR(+ AR)': 0,
    }

    label_46 = {
    'Ansicht Brückengeländer M 1 10': 0,
    'Bewegungsfuge M 1x5': 0,
    'Blatt 87': 0,
    'Detail Geländerfuss': 0,
    'Detail Geländerpfosten M 1 10': 0,
    'F1. 120×10': 0,
    }

    label_a8_b1 = {
    'Ansicht Sud': 4,
    'Endgültige Abmessungen nach statischen,': 0,
    'Hinterfüllung gem. [Was 7': 0,
    'J    Treppenanlage I H': 0,
    'J    Treppenanlage I H_1': 0,
    'Regelquerschnitt M. 1 50': 3,
    'Schnitt A-A_M. 1 100': 4,
    }
    
    label_a8_b2 = {
    "Datum Zeichen": 0,
    "Detail 1": 0,
    "Detail 2 M. 1 25": 0,
    "Detail 3 M. 1 50": 0,
    "Detail 4": 0,
    "Detail 5": 0,
    "Detail 6": 0,
    "Dieser Plan gilt nur in Verbindung mit Blatt-Nr . 1": 0,
    "Endgültige Abmessungen nach statischen,": 0,
    "Hartschaumplatte": 0,
    "Schnitt B-B_M. 1 100": 0,
    "Schnitt C-C_M. 1 100": 0,
    "Schnitt D-D": 0,
    "Schnitt E-E M. 1 50": 0,
    "Schnitt F-F M. 1 50": 0,
    "Schnitt G - G": 0,
    "Schnitt H - H M. 1 100": 0,
    "Schnitt J - J M. 1 100": 0,
    "Schnitt M - M": 0,
    "Stützwand Block 1 bis 3": 0,
    "Widerlager Achse 50": 0
    }


    dic = {
        '32': label_32,
        '33': label_33,
        '34': label_34,
        '35': label_35,
        '36': label_36,
        '37': label_37,
        '38': label_38,
        '39': label_39,
        '40': label_40,
        '41': label_41,
        '42': label_42,
        '43': label_43,
        '44': label_44,
        '45': label_45,
        '46': label_46,
        '47': label_a8_b1, # Anlage 8 Blatt 1
        '48': label_a8_b2, # Anlage 8 Blatt 2
    }
    return dic[key]

if __name__ == '__main__':
    scan_number = 'Entwurf_Anlage_8_Blatt_2'
    print(scan_number, ':')
    # for name in os.listdir(os.path.join(scans_folder, scan_number)):
    #     if name.endswith('.jpg') or name.endswith('.png'):
    #         print(name[:-4])
    for name in os.listdir(wireframes_folder):
        if name.endswith('.jpg') or name.endswith('.png'):
            print(name[:-4])