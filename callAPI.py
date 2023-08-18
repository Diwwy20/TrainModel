import os
import cv2
import base64
import requests
import pickle

def imgtovec(img):
    try:
        resized_img = cv2.resize(img, (128, 128), cv2.INTER_AREA)
        v, buffer = cv2.imencode(".jpg", resized_img)
        img_str = base64.b64encode(buffer).decode('utf-8')
        image_data_string = "data:image/jpeg;base64," + img_str

        url = "http://localhost:8080/api/genhog"
        params = {"img_base64": image_data_string}

        response = requests.get(url, json=params)

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"เรียก API ไม่สำเร็จ API CODE : {response.status_code}"}
    except Exception as ex:
        return {"error": f"เกิดข้อผิดพลาด: {str(ex)}"}
    
# path = "Cars Dataset/train/Audi/1.jpg"
# img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# print(imgtovec(img))

path = "Cars Dataset/train" # Cars Dataset/train & Cats Dataset/test
list_x = []
list_y = []
# label_mapping = {}
# label_number = 0

# อ่านภาพไฟล์เป็น base64 และ นำไปเก็บไว้ใส่ตัวแปร list_x #
# อ่านภาพโฟลเดอร์ที่ชื่อยี่ห้อรถยนต์ของแต่ละภาพ เก็บไว้ใน list_y #
for sub in os.listdir(path):
    for fn in os.listdir(os.path.join(path, sub)):
        path_file_img = os.path.join(path, sub, fn)
        readImage = cv2.imread(path_file_img, cv2.IMREAD_GRAYSCALE)
        list_x.append(readImage)
        list_y.append(sub)
        # if sub not in label_mapping:
        #     label_mapping[sub] = label_number
        #     label_number += 1

# print(len(list_x))
# print(len(label_mapping))

# ส่งภาพ base64 ไปให้ api แปลงกลับมาเป็น hog vector #
hogvectors = []
for i in range(len(list_x)):
    res = imgtovec(list_x[i])
    vec = list(res["HOG"])
    vec.append(list_y[i])
    hogvectors.append(vec)

# print(hogvectors)

# เขียน hogvectors_train.pkl ขึ้นมา #
write_path = "hogvectors_train.pkl"
pickle.dump(hogvectors, open(write_path, "wb"))
print("data preparation is done")

# เขียน hogvectors_test.pkl ขึ้นมา #
# write_path = "hogvectors_test.pkl"
# pickle.dump(hogvectors, open(write_path, "wb"))
# print("data preparation is done")