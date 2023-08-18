from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle
import xgboost as xgb
from sklearn.ensemble import VotingClassifier

hogvectors_train = pickle.load(open('hogvectors_train.pkl', 'rb'))
hogvectors_test = pickle.load(open('hogvectors_test.pkl', 'rb'))

# ดึงข้อมูลทุกแถว และ คอลัมน์ที่ 0-8099 มาเป็น feature 
X_train_data = [hogfeature_Xtrain[0:8100] for hogfeature_Xtrain in hogvectors_train]
X_test_data = [hogfeature_Xtest[0:8100] for hogfeature_Xtest in hogvectors_test]

# ดึงข้อมูลทุกแถว แต่เอาแค่คอลัมน์สุดท้าย มาเป็น class
Y_train_data = [hogfeature_Ytrain[-1] for hogfeature_Ytrain in hogvectors_train]
Y_test_data = [hogfeature_Ytest[-1] for hogfeature_Ytest in hogvectors_test]

label_encoder = LabelEncoder() # สร้าง object label_encoder จาก Class LabelEncoder เพราะต้องใช้ในการแปลงชื่อยี่ห้อรถยนต์เป็นตัวเลข
y_cls_train = label_encoder.fit(Y_train_data) # ใช้ .fit(Y_train_data) เพื่อใช้ในการเรียนรู้ว่า ชื่อยี่ห้อรถยนต์จะถูกแทนด้วยตัวเลขอะไรบ้าง ( ทำ mapping )
y_labelNum_train = label_encoder.transform(Y_train_data) # หลังจากใช้ .fit เพื่ออบรมแล้วจะเป็นตัวเลขตาม mapping ที่ถูกสร้างไว้
y_cls_test = label_encoder.fit(Y_test_data)
y_labelNum_test = label_encoder.transform(Y_test_data)
# # แปลงชุดข้อมูล สตริง ยี่ห้อรถจาก Test เป็น TableIndex

# print(len(y_labelNum_train))
# print(len(label_encoder.classes_))
# print(len(set(y_labelNum_train)))

# สร้าง object จาก model DecisionTree
clf = DecisionTreeClassifier(random_state=42)

# สร้าง object จาก modelXGBoost
xgb_model = xgb.XGBClassifier(objective="multi:softmax",num_class=len(label_encoder.classes_), random_state=42)

# ทำการรวม 2โมเดล ที่สร้างไว้มารวมด้วยกัน 
ensemble_model = VotingClassifier(estimators=[('DecisionTree', clf), ('XGBoost', xgb_model)], voting='hard',weights=[1, 4])

# .fit() X_train_data เป็นการให้โมเดลมันเรียนรู้ข้อมูลที่เหมาะสมกับข้อมูล ส่วน Y_train_data เป็นคำตอบที่ควรจะได้จากการเรียนรู้
ensemble_model.fit(X_train_data, y_labelNum_train) 

# ใช้ข้อมูลทดสอบ X_test_data เพื่อไว้ทำนายผลลัพธ์ที่ได้จากโมเดลนี้ของข้อมูลทดสอบ
y_pred = ensemble_model.predict(X_test_data) 

# y_labelNum_test เป็นข้อมูลที่ถูกต้องจริงของข้อมูลทดสอบ ส่วน y_pred เป็นข้อมูลคำนวณความแม่นยำของโมเดลที่ได้นายผลลัพธ์จากข้อมูลทดสอบ
# คือมันจะทำการเปรียบเทียบ โดยเอาข้อมูลที่ทำนาย y_pred ที่ได้จากการ predict ข้อมูลทดสอบ โดยเปรียบเทียบว่าจำนวนคำตอบที่ถูกต้องที่โมเดลทำนายตรงกับ
# y_labelNum_test เป็นกี่เปอร์เซ็นต์ของจำนวนข้อมูลทดสอบ
accuracy = accuracy_score(y_labelNum_test, y_pred)
confusionMatrix = confusion_matrix(y_labelNum_test, y_pred)
print("Accuracy: ", accuracy)
print("Confusion Matrix: ", confusionMatrix)

path_model = 'model_genhog.pkl'
pickle.dump(ensemble_model, open(path_model, 'wb'))