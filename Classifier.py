from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
import cv2
import sys

model = load_model('./models/model_01_2class.h5')

image_url = input("path=")
#./cat/cat1.jpg
image=cv2.imread(image_url)

#画像の読込
img = img_to_array(load_img(image_url, target_size=(224,224)))
#0-1に変換
img_nad = img_to_array(img)/255
#4次元配列に
img_nad = img_nad[None, ...]

classes = ["Dog","Cat"]

#判別
pred = model.predict(img_nad, batch_size=1, verbose=0)
#判別結果で最も高い数値を抜き出し
score = np.max(pred)
#判別結果の配列から最も高いところを抜きだし、そのクラス名をpred_labelへ
pred_label = classes[np.argmax(pred[0])]

result =  pred_label + "  " + str(score*100) +"%"

def detect_face(image):
    img = cv2.resize(image,(224,224))
    img=np.expand_dims(img,axis=0)
    cv2.putText(image,result,(0,50),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)
    return image


#if __name__ == '__main__':


if image is None:
    print("Not open:")
b,g,r = cv2.split(image)
image = cv2.merge([r,g,b])
whoImage=detect_face(image)

plt.imshow(whoImage)
plt.show()
