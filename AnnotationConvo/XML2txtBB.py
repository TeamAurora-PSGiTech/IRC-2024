import xml.etree.ElementTree as ET
from colorama import Fore, Style
from tkinter import filedialog
from tqdm import tqdm
import glob

print(Fore.YELLOW+Style.BRIGHT+"\n\nSelect in-XML-Content Directory"+Fore.RESET)
inDir = filedialog.askdirectory()
print(Fore.BLUE+Style.BRIGHT+"\n\nSelect the numClasses Text File Yo!"+Fore.RESET)
inTxt = filedialog.askopenfilename(filetypes=[("NumClass TextFile", "*.txt")])
print(Fore.CYAN+Style.BRIGHT+"\n\nand Come On! Select outPut TXT Directory"+Fore.RESET)
outDir = filedialog.askdirectory()
if "\\" in outDir:
    outDir = outDir + '\\'
else:
    outDir = outDir + '/'

f = open(inTxt, 'r')
clsDict = eval(f.read())
f.close()

print(clsDict)
files = glob.glob(inDir+"/*.xml")


for f in tqdm(files, desc = "Gettin' the Txt File out Yo!"):
    f_ = f.replace("\\", "~").replace("/", "~")
    outFile = (f_.split("~")[-1]).split('.')[0] + '.txt'
    outFile = outDir + outFile
    tree = ET.parse(f)
    root = tree.getroot()

    sizeElement = root.find('size')
    widthElement = sizeElement.find('width')
    heightElement = sizeElement.find('height')
    xDiv = int(widthElement.text)
    yDiv = int(heightElement.text)
    
    labelData = []
    sizeElement = root.find('size')
    imgWidth = int(sizeElement.find('width').text)
    imgHeight = int(sizeElement.find('height').text)
    objectElements = root.findall('object')
    for objectElement in objectElements:
        nameElement = objectElement.find('name')
        clsName = nameElement.text
        pnts = []
        nameElement = objectElement.find('name')
        bbxElement = objectElement.find('bndbox')
        xMin = int(float(bbxElement.find('xmin').text))
        yMin = int(float(bbxElement.find('ymin').text))
        xMax = int(float(bbxElement.find('xmax').text))
        yMax = int(float(bbxElement.find('ymax').text))

        xCen = str((xMin + ((xMax-xMin)/2))/imgWidth)
        yCen = str((yMin + ((yMax-yMin)/2))/imgHeight)
        clsWidth = str((xMax - xMin)/imgWidth)
        clsHeight = str((yMax - yMin)/imgHeight )
        pnts = [xCen, yCen, clsWidth, clsHeight]

        loadData = ' '.join([str(clsDict[clsName])] + pnts)
        labelData.append(loadData)
    txtData = '\n'.join(labelData)
    f_ = open(outFile, 'w')
    f_.write(txtData)
    f_.close()

clsTxt = ', '.join(list(clsDict.keys()))
f = open(outDir+'classes.txt', 'w')
f.write(clsTxt)
f.close()