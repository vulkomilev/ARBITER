from PIL import Image, ImageDraw, ImageFont,ImageOps
import random
from pathlib import Path
import threading
import csv
import time
from xml.etree import cElementTree as ET
from dicttoxml import dicttoxml

class XMLToDictionary(dict):
    def __init__(self, parentElement):
        self.parentElement = parentElement
        for child in list(parentElement):
            child.text = child.text if (child.text != None) else  ' '
            if len(child) == 0:
                self.update(self._addToDict(key= child.tag, value = child.text.strip(), dict = self))
            else:
                innerChild = XMLToDictionary(parentElement=child)
                self.update(self._addToDict(key=innerChild.parentElement.tag, value=innerChild, dict=self))

    def getDict(self):
        return {self.parentElement.tag: self}

    class _addToDict(dict):
        def __init__(self, key, value, dict):
            if not key in dict:
                self.update({key: value})
            else:
                identical = dict[key] if type(dict[key]) == list else [dict[key]]
                self.update({key: identical + [value]})

textArray = 'QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm123456789 '

def generate_text(length=10):
    local_word = ''
    for i in range(length):
        random_letter = random.choice(textArray)
        local_word += random_letter
    return local_word
IMAGE_HEIGHT = 1080
IMAGE_WIDTH = 1920
GLOBAL_DIR = './textDataset/'
TARGERT_DIR = 'dataset/sites-mini/'
RANDOM_POS_X = 400
RANDOM_POS_Y = 200
RANDOM_SIZE = 90
RANDOM_ROTATE = 45
RANDOM_COLOR_R = 1
RANDOM_COLOR_B = 1
RANDOM_COLOR_G = 1
RANDOM_TRIES_MAX = 100
RANDOM_TEXT_LENGTH_MIN = 10
RANDOM_TEXT_LENGTH_MAX = 1000
RECTANGLE_SIZE = 0
PREFIX_COUNT = 0
RANDOM_FONTS_SET = ["UbuntuMono-R.ttf",'ArialCEBold.ttf',
                    "ArialCE.ttf",'Times_New_Roman.ttf','DejaVuSansMono.ttf','Times_New_Roman_Bold.ttf']
#["UbuntuMono-R.ttf", 'ArialCEBold.ttf', 'ArialCEItalic.ttf',
# "ArialCE.ttf", 'Times_New_Roman.ttf', 'DejaVuSansMono.ttf', 'Times_New_Roman_Bold.ttf',
# 'Times_New_Roman_Italic.ttf', 'Times_New_Roman_Bold_Italic.ttf']
template_label = {'name': '1', 'pose': 'Unspecified', 'truncated': '0', 'difficult': '0', 'bndbox':
    {'xmin': '482', 'ymin': '562', 'xmax': '496', 'ymax': '578'}}
template_annotation = \
    {'annotation': {'folder': TARGERT_DIR, 'filename': 'test.png', 'path': 'autoCdp/labelImg/tests/test.png',
                    'source': {'database': 'Unknown'}, 'size': {'width': '1920', 'height': '1080', 'depth': '3'},
                    'segmented': '0', 'object': []}}
template_annotation = \
    {'annotation': {'folder': 'dataset/sites/', 'filename': 'test.png', 'path': 'autoCdp/dataset/sites/0.png',
                    'source': {'database': 'Unknown'}, 'size': {'width': '1920', 'height': '1080', 'depth': '3'},
                    'segmented': '0', 'object': {'item': [{'name': '1', 'pose': 'Unspecified', 'truncated': '0', 'difficult': '0', 'bndbox': {'xmin': '202', 'ymin': '86', 'xmax': '202', 'ymax': '86'}}, {'name': '1', 'pose': 'Unspecified', 'truncated': '0', 'difficult': '0', 'bndbox': {'xmin': '202', 'ymin': '86', 'xmax': '202', 'ymax': '86'}}, {'name': '1', 'pose': 'Unspecified', 'truncated': '0', 'difficult': '0', 'bndbox': {'xmin': '202', 'ymin': '86', 'xmax': '202', 'ymax': '86'}}, {'name': '1', 'pose': 'Unspecified', 'truncated': '0', 'difficult': '0', 'bndbox': {'xmin': '202', 'ymin': '86', 'xmax': '202', 'ymax': '86'}}, {'name': '1', 'pose': 'Unspecified', 'truncated': '0', 'difficult': '0', 'bndbox': {'xmin': '202', 'ymin': '86', 'xmax': '202', 'ymax': '86'}}, {'name': '1', 'pose': 'Unspecified', 'truncated': '0', 'difficult': '0', 'bndbox': {'xmin': '202', 'ymin': '86', 'xmax': '202', 'ymax': '86'}}, {'name': '1', 'pose': 'Unspecified', 'truncated': '0', 'difficult': '0', 'bndbox': {'xmin': '202', 'ymin': '86', 'xmax': '202', 'ymax': '86'}}, {'name': '1', 'pose': 'Unspecified', 'truncated': '0', 'difficult': '0', 'bndbox': {'xmin': '202', 'ymin': '86', 'xmax': '202', 'ymax': '86'}}, {'name': '1', 'pose': 'Unspecified', 'truncated': '0', 'difficult': '0', 'bndbox': {'xmin': '202', 'ymin': '86', 'xmax': '202', 'ymax': '86'}}, {'name': '1', 'pose': 'Unspecified', 'truncated': '0', 'difficult': '0', 'bndbox': {'xmin': '202', 'ymin': '86', 'xmax': '202', 'ymax': '86'}}]}}}
template_text_1 = """
<annotation>
	<folder>tests</folder>
	<filename>test.png</filename>
	<path>/autoCdp/labelImg/tests/test.png</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>1522</width>
		<height>875</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
"""

template_text_2="""

</annotation>"""
csv_dict = {'ids':[]}
class myThread (threading.Thread):
   def __init__(self, start, end):
      threading.Thread.__init__(self)
      self.start_pos = int(start)
      self.end_pos = int(end)
   def run(self):
    for i in range(self.start_pos,self.end_pos):
        print(str(i)+'/250000')
        Path("./"+TARGERT_DIR).mkdir(parents=True, exist_ok=True)
        random_image_heigth = random.randint(100,400)
        random_image_width = random.randint(100,800)
        RANDOM_POS_X = random_image_width
        RANDOM_POS_Y = random_image_heigth
        IMAGE_HEIGHT = random_image_heigth
        IMAGE_WIDTH = random_image_width
        # 1920x1080# Image.open('templateSite.png')
        image = Image.new("RGB", ( random_image_width,random_image_heigth), (255, 255, 255))

        for j in range(PREFIX_COUNT):
            image_second = Image.open('./prefixes/paint'+str(random.randint(1,3))+'.png')
            img_w, img_h = image_second.size
            img_w = img_w / float(random.randint(1,1000))
            img_h = img_h / float(random.randint(1,1000))
            offset = (int((img_w) // 2), int((img_h) // 2))
            image.paste(image_second, offset)
        for j in range(RECTANGLE_SIZE):
            draw = ImageDraw.Draw(image)
            img_w, img_h = image.size
            random_r = random.randint(0,RANDOM_COLOR_R)
            random_g = random.randint(0,RANDOM_COLOR_G)
            random_b = random.randint(0,RANDOM_COLOR_B)
            draw.rectangle([random.randint(0,img_w),random.randint(0,img_w),
                            random.randint(0,img_h),random.randint(0,img_h)], fill=(random_r,random_g,random_b,255))
        local_template = template_annotation#copy
        local_template['annotation']['path'] = GLOBAL_DIR+TARGERT_DIR+str(i)+'.png'
        width, height = image.size

        font = ImageFont.truetype(random.choice(RANDOM_FONTS_SET), random.randint(10,RANDOM_SIZE), encoding="unic")
        margin = 50
        previous_locations = []
        previous_size = []
        local_object = ""
        local_dict = ""
        for n in range(random.randint(10,15)):
            local_text = generate_text(random.randint(RANDOM_TEXT_LENGTH_MIN, RANDOM_TEXT_LENGTH_MAX))
            new_location = True
            local_try = 0
            while new_location:
                x =  margin + random.randint(0, RANDOM_POS_X)
                y =  margin - 20 + random.randint(0, RANDOM_POS_Y)
                new_location = False
                local_try += 1
                for location,size in zip(previous_locations,previous_size):
                    if y > location[1]-50 \
                            and y < location[1]+size[1]+30:
                        new_location = True
                        break
                if local_try > RANDOM_TRIES_MAX:
                    break
            if local_try > RANDOM_TRIES_MAX:
                continue
            #print("----------------------------")
            #print(x,font.getsize(local_text)[0])
            #print(y, font.getsize(local_text)[1])
            #print(local_text)
            previous_locations.append([x,y])
            previous_size.append([font.getsize(local_text)[0], font.getsize(local_text)[1]])
            x_local = x
            draw = ImageDraw.Draw(image)
            total_textwidth = 0
            textwidth = None
            total_textheight = 0
            textheight = None
            for j in local_text:
                 local_textwidth, local_textheight = font.getsize(j)
                 if textwidth == None and textheight == None:
                    textwidth  = local_textwidth
                    textheight = local_textheight
                 if local_textheight > textheight:
                     textheight = local_textheight
                 local_object += """
                    <object>
                    <name>
                    """+j+\
                    """</name>
                    <pose>Unspecified</pose>
                    <truncated>0</truncated>
                    <difficult>0</difficult>
                    <bndbox>
                        <xmin>"""+str(x_local)+"""</xmin>
                        <ymin>"""+str(y+textheight)+"""</ymin>
                        <xmax>"""+str(x_local+textwidth)+\
                            """</xmax>
                        <ymax>"""+str(y+20)+"""</ymax>
                    </bndbox>
                    </object>
                 """
                 total_textwidth += local_textwidth
                 #x_local += textwidth
            local_dict += str('1') + " " + str(((x) / IMAGE_WIDTH) +(total_textwidth / float(IMAGE_WIDTH))/2.0 ) \
                          + " " + str((y + textheight / 2.0) / IMAGE_HEIGHT) + " " + \
                          str(total_textwidth / float(IMAGE_WIDTH)) + " " + str(
                textheight / float(IMAGE_HEIGHT)) + " \n"
            random_r = random.randint(0,RANDOM_COLOR_R)
            random_g = random.randint(0,RANDOM_COLOR_G)
            random_b = random.randint(0,RANDOM_COLOR_B)
            draw.text((x, y), local_text, font=font, fill=(random_r,random_g,random_b,255))
            #image = ImageOps.invert(image)
        #image = image.rotate(random.randint(0, RANDOM_ROTATE))

        # optional parameters like optimize and quality

        #f = open("new_test_"+str(i)+".xml", "a")
        #f.write(template_text_1+local_object+template_text_2)
        #f.close()
        f = open("./" + TARGERT_DIR + "/img"+str(i)+".txt", "a")
        f.write(local_dict)
        f.close()
        csv_dict['ids'].append("img"+str(i))
        image.save("./" + TARGERT_DIR + "/img" + str(i) + '.png', optimize=True, quality=50)
'''


for text in textArray:

    for i in range(250):
        image = Image.open('templateLetter.png')
        width, height = image.size

        draw = ImageDraw.Draw(image)
        textwidth, textheight = draw.textsize(text)

        margin = 50
        x = width - textwidth - margin + random.randint(-RANDOM_POS,RANDOM_POS)
        y = height - textheight - margin - 20 + random.randint(-RANDOM_POS,RANDOM_POS)
        font = ImageFont.truetype("UbuntuMono-BI.ttf", 90, encoding="unic")

        draw.text((x, y), text,font=font, fill="#000")

        image = image.rotate(random.randint(0,RANDOM_ROTATE))
        # optional parameters like optimize and quality
        image.save("./"+TARGERT_DIR+"/"+text+'/'+str(i)+'.png', optimize=True, quality=50)
'''


#tree = ET.parse('./new_test.xml')
#root = tree.getroot()
#parseredDict = XMLToDictionary(root).getDict()
#print(parseredDict)

threads = []

# Create new threads
batch_size = 10000/25
for i in range(25):
    thread1 = myThread(i*batch_size,  (i+1)*batch_size)
    thread1.start()
    threads.append(thread1)

# Wait for all threads to complete
for t in threads:
    t.join()
with open("./" + TARGERT_DIR + "/"+'train.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    titles = []
    for key, value in csv_dict.items():
       titles.append(key)
    writer.writerow(titles)
    for key, value in csv_dict.items():
        for val in csv_dict[key]:
            writer.writerow([val])