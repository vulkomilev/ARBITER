import csv
import os
captcha_arr = []
for path, subdirs, files in os.walk('./'):
    if path == './':
        continue
    for name in files:
            captcha_arr.append(name[:-4])


with open('train.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['name','letter'])
    for captcha in captcha_arr:
        writer.writerow([captcha+'.png','"'+captcha+'"'])
