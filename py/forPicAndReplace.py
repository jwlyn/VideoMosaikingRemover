import os
from PIL import Image

path = r"image"
os.chdir(path)
files = os.listdir(".") #或者"."改为os.getcwd()，请不要直接使用os.listdir(path)
#打开目录下所有文件进行处理（所以只能放图片）
for file in files :
    print(file)
    #也可以通过file[-3:len(file)]判断，只处理图像格式
    img = Image.open(file)
    new_img = img.convert('L') #转换为灰度

    file=os.path.join(file) #必须拼接完整文件名
    if os.path.isfile(file) and file.find(".jpg")>0:
        os.remove(file)
        print(file+" remove succeeded")

    #修改后名称为源名称中间加.bak,file[0:-3]是取文件名（包含最后的.）
    new_img.save(file)