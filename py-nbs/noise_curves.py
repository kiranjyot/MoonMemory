# # Author: Kiranjyot Gill (2024)

import numpy as np
import os

# Load in GW noise curves 
# sourced from 
# https://dcc.ligo.org/LIGO-T2000012/public
# https://github.com/janosch314/GWFish/tree/main/GWFish/detector_psd 
# https://www.vanderbilt.edu/lunarlabs/lila/
# https://github.com/robsci/GWplotter/blob/master/data/detectors/)
Asharp = np.genfromtxt('Data/Sensitivity Curves/Asharp.txt')
LILA = np.genfromtxt('Data/Sensitivity Curves/LILA.txt')
LIGO_O3b_1262178304 = np.genfromtxt('Data/Sensitivity Curves/O3b_L1_1262178304.txt')
AdVirgo = np.genfromtxt('Data/Sensitivity Curves/avirgo_O5high_NEW.txt')
LIGO_O3_L1 = np.genfromtxt('Data/Sensitivity Curves/aligo_O3actual_L1.txt')
LIGO_O3_H1 = np.genfromtxt('Data/Sensitivity Curves/aligo_O3actual_H1.txt')
ALIGO_O4 = np.genfromtxt('Data/Sensitivity Curves/aligo_O4high.txt')
CE = np.genfromtxt('Data/Sensitivity Curves/CE1_PSD.txt')
ET = np.genfromtxt('Data/Sensitivity Curves/ET_D.txt')
LGWAnb = np.genfromtxt('Data/Sensitivity Curves/LGWANb.txt')
LGWAsi = np.genfromtxt('Data/Sensitivity Curves/LGWASi.txt')
GLOCc = np.genfromtxt('Data/Sensitivity Curves/GLOC_conservative.txt')
GLOCo = np.genfromtxt('Data/Sensitivity Curves/GLOC_optimal.txt') 
LISA = np.genfromtxt('Data/Sensitivity Curves/LISA.txt')
DECIGO = np.genfromtxt('Data/Sensitivity Curves/DECIGO1.txt')
BDECIGO = np.genfromtxt('Data/Sensitivity Curves/BDECIGO1.txt')
ALIA = np.genfromtxt('Data/Sensitivity Curves/ALIA.txt')
DO = np.genfromtxt('Data/Sensitivity Curves/DeciHzobs.txt')
BBO = np.genfromtxt('Data/Sensitivity Curves/BBO.txt')
TianGO = np.genfromtxt('Data/Sensitivity Curves/TianGO.txt')
atom = np.genfromtxt('Data/Sensitivity Curves/ATOMifo.txt')


noise_curves = {
    'LIGO_O3b_1262178304': LIGO_O3b_1262178304,
    'LIGO_O3_L1': LIGO_O3_L1, 
    'LIGO_O3_H1': LIGO_O3_H1,
    'AdVirgo': AdVirgo,
    'ALIGO_O4': ALIGO_O4,
    'Asharp': Asharp,
    'Cosmic': CE,
    'ET': ET,
    'LILA': LILA, 
    'LGWAnb': LGWAnb,
    'LGWAsi': LGWAsi,
    'GLOCc': GLOCc,
    'GLOCo': GLOCo,
    'LISA': LISA,
    'DECIGO': DECIGO,
    'BDECIGO':BDECIGO,
    'ALIA':ALIA,
    'DO':DO,
    'TianGO':TianGO,
    'BBO':BBO,
    'atom':atom
}


