import xml.etree.ElementTree as ET
from colorama import Fore, Style
from tkinter import filedialog
from tqdm import tqdm
import yaml
import glob
import os

print(Fore.YELLOW+Style.BRIGHT+"\n\nSelect in-XML-Content Directory"+Fore.RESET)
inDir = filedialog.askdirectory()
if not os.path.isdir("dataYAMLnTXT"):
   os.makedirs("dataYAMLnTXT")

clsSet = set()
files = glob.glob(inDir+"/*.xml")
for f in tqdm(files, desc = "Fetching Class Names Yo!"):
    tree = ET.parse(f)
    root = tree.getroot()
    for i in root.iter('name'):
        clsSet.add(i.text)

clsList = list(clsSet)
clsList.sort()
print(clsList)
clsNum = [x for x in range(0, len(clsList))]
clsDict = dict(zip(clsList, clsNum))
nameDict = dict(zip(clsNum, clsList))
f = open("dataYAMLnTXT/numClasses.txt", 'w')
f.write(str(clsDict))
f.close()

yamlSegmentDict = dict(
    train = "../train/images",
    val = "../valid/images",
    nc = len(clsNum),
    names = str(clsList)
)
f = open("dataYAMLnTXT/segmentationData.yaml", 'w')
yaml.dump(yamlSegmentDict, f, default_flow_style=False)
f.close()

yamlDetectionDict = dict(
    train = "../train/images",
    val = "../valid/images",
    names = nameDict
)
f = open("dataYAMLnTXT/detectionData.yaml", 'w')
yaml.dump(yamlDetectionDict, f, default_flow_style=False)
f.close()