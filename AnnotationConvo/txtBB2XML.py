import xml.etree.ElementTree as ET
from colorama import Fore, Style
from tkinter import filedialog
from tqdm import tqdm
import glob
import cv2
import os

print(Fore.YELLOW+Style.BRIGHT+"\n\nSelect in-TxT-Content Directory"+Fore.RESET)
inTxTDir = filedialog.askdirectory()
print(Fore.YELLOW+Style.BRIGHT+"\n\nSelect LabellediMG-Content Directory"+Fore.RESET)
inImGDir = filedialog.askdirectory()
print(Fore.BLUE+Style.BRIGHT+"\n\nSelect the numClasses Text File Yo!"+Fore.RESET)
inTxt = filedialog.askopenfilename(filetypes=[("NumClass TextFile", "*.txt")])
print(Fore.CYAN+Style.BRIGHT+"\n\nand Come On! Select outPut XML Directory"+Fore.RESET)
outDir = filedialog.askdirectory()
if "\\" in outDir:
    slashStuff = '\\'
else:
    slashStuff = '/'

f = open(inTxt, 'r')
clsNames = f.read().replace(", ", ',').split(',')
f.close()
clsDict = dict(zip(list(range(len(clsNames))), clsNames))

files = glob.glob(inTxTDir+"/*.txt")
for f in tqdm(files, desc = "Gettin' the Txt File out Yo!"):
    f_ = f.replace("\\", "~").replace("/", "~")
    outName = (f_.split("~")[-1]).split('.')[0] + '.xml'
    if os.path.isfile(inImGDir + slashStuff + outName.replace(".xml", ".jpg")) or os.path.isfile(inImGDir + slashStuff + outName.replace(".xml", ".png")):
        try:
            img = cv2.imread(inImGDir + slashStuff + outName.replace(".xml", ".jpg"))
        except:
            img = cv2.imread(inImGDir + slashStuff + outName.replace(".xml", ".png"))
        imgWidth = img.shape[1]
        imgHeight = img.shape[0]
        outFile = outDir + slashStuff + outName
        txtFile = open(f, 'r')
        txtContents = txtFile.read().split('\n')
        txtFile.close()

        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = "images"
        ET.SubElement(root, "filename").text = outName.replace(".xml", ".jpg")
        ET.SubElement(root, "path").text = "..//images//{}".format(outName.replace(".xml", ".jpg"))
        
        sizeElement = ET.SubElement(root, "size")
        ET.SubElement(sizeElement, "width").text = str(imgWidth)
        ET.SubElement(sizeElement, "height").text = str(imgHeight)
        ET.SubElement(sizeElement, "depth").text = '3'
        
        for i in txtContents:
            annoData = [eval(x) for x in i.split()]
            objectElement = ET.SubElement(root, "object")
            ET.SubElement(objectElement, "name").text = clsDict[annoData[0]]
            
            xCen = annoData[1]
            yCen = annoData[2]
            clsWidth = annoData[3]
            clsHeight = annoData[4]
            xUnits = clsWidth * imgWidth
            yUnits = clsHeight * imgHeight
            xMin = ((xCen * imgWidth * 2) - xUnits) / 2
            yMin = ((yCen * imgHeight * 2) - yUnits) / 2
            xMax = xMin + xUnits
            yMax = yMin + yUnits
            bBoxElement = ET.SubElement(objectElement, "bndbox")
            ET.SubElement(bBoxElement, "xmin").text = str(xMin)
            ET.SubElement(bBoxElement, "ymin").text = str(yMin)
            ET.SubElement(bBoxElement, "xmax").text = str(xMax)
            ET.SubElement(bBoxElement, "ymax").text = str(yMax)
        tree = ET.ElementTree(root)
        ET.indent(tree, space="\t", level=0)
        tree.write(outFile, encoding='utf-8')   