import cv2
import os
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Flatten,Dense,MaxPooling2D,Dropout
import matplotlib.pyplot as plt 
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.utils import to_categorical
Test1_DIR="Testing/glioma_tumor"
Test2_DIR="Testing/meningioma_tumor"
Test3_DIR="Testing/pituitary_tumor"
Test4_DIR="Testing/no_tumor"
Train1_DIR="Training/glioma_tumor"
Train2_DIR="Training/meningioma_tumor"
Train3_DIR="Training/pituitary_tumor"
Train4_DIR="Training/no_tumor"



INPUT_SHAPE=[(350,350,1),(400,400,1)]
layer1_n_filters=[16,32]
layer1_kernel_sizes=[(9,9),(11,11)]
layer2_strides=[(1,1),(2,2)]
N_SAMPLE=10
images_Testing1_name=os.listdir(Test1_DIR)
images_Testing2_name=os.listdir(Test2_DIR)
images_Testing3_name=os.listdir(Test3_DIR)
images_Testing4_name=os.listdir(Test4_DIR)
images_Training1_name=os.listdir(Train1_DIR)
images_Training2_name=os.listdir(Train2_DIR)
images_Training3_name=os.listdir(Train3_DIR)
images_Training4_name=os.listdir(Train4_DIR)




i=0
models_results=[]
for INP_SH in INPUT_SHAPE :
    images_Testing1=[]
y=[]
for name in images_Testing1_name[ :N_SAMPLE]:
	img=cv2.imread(Test1_DIR+name)

	try:
	   img_resize=cv2.resize(img, INP_SH [ :-1])
	   img_gray=cv2.cvtColor(img_resize , cv2.COLOR_BGR2GRAY).reshape((INP_SH ))
	   images_Testing.append(img_gray)
	   y.append(0)
	except Exception as e:
	    print(e)
images_Testing1=np.array(images_Testing1)
# print(models_results)
# quit()






images_Testing2=[]
# y=[]
for name in images_Testing2_name[ :N_SAMPLE]:
	img=cv2.imread(Test2_DIR+name)

	try:
	   img_resize=cv2.resize(img, INP_SH [ :-1])
	   img_gray=cv2.cvtColor(img_resize , cv2.COLOR_BGR2GRAY).reshape((INP_SH ))
	   images_Testing.append(img_gray)
	   y.append(1)
	except Exception as e:
	    print(e)
images_Testing2=np.array(images_Testing2)
print(images_Testing2.shape)

images_Testing3=[]
# y=[]
for name in images_Testing3_name[ :N_SAMPLE]:
	img=cv2.imread(Test3_DIR+name)

	try:
	   img_resize=cv2.resize(img, INP_SH [ :-1])
	   img_gray=cv2.cvtColor(img_resize , cv2.COLOR_BGR2GRAY).reshape((INP_SH ))
	   images_Testing.append(img_gray)
	   y.append(2)
	except Exception as e:
	    print(e)
images_Testing3=np.array(images_Testing3)
print(images_Testing3.shape)

images_Testing4=[]
# y=[]
for name in images_Testing4_name[ :N_SAMPLE]:
	img=cv2.imread(Test4_DIR+name)

	try:
	   img_resize=cv2.resize(img, INP_SH [ :-1])
	   img_gray=cv2.cvtColor(img_resize , cv2.COLOR_BGR2GRAY).reshape((INP_SH ))
	   images_Testing.append(img_gray)
	   y.append(3)
	except Exception as e:
	    print(e)
images_Testing4=np.array(images_Testing4)
print(images_Testing4.shape)

images_Training1=[]
y1=[]

for name in images_Training1_name[ :N_SAMPLE]:
    img=cv2.imread(Train1_DIR+name)
    try:
    	img_resize=cv2.resize(img, INP_SH [:-1])
    	img_gray=cv2.cvtColor(img_resize,cv2.COLOR_BGR2GRAY).reshape(( INP_SH ))
    	images_Trainig.append(img_gray)
    	y1.append(4)
    except Exception as e:
    	print(e)
images_Training1=np.array(images_Training1)

images_Training2=[]
for name in images_Training2_name[ :N_SAMPLE]:
    img=cv2.imread(Train2_DIR+name)
    try:
    	img_resize=cv2.resize(img, INP_SH [:-1])
    	img_gray=cv2.cvtColor(img_resize,cv2.COLOR_BGR2GRAY).reshape(( INP_SH ))
    	images_Training.append(img_gray)
    	y1.append(5)
    except Exception as e:
    	print(e)
images_Training2=np.array(images_Training2)

images_Training3=[]
for name in images_Training3_name[ :N_SAMPLE]:
    img=cv2.imread(Train3_DIR+name)
    try:
    	img_resize=cv2.resize(img, INP_SH [:-1])
    	img_gray=cv2.cvtColor(img_resize,cv2.COLOR_BGR2GRAY).reshape(( INP_SH ))
    	images_Training.append(img_gray)
    	y1.append(6)
    except Exception as e:
    	print(e)
images_Training3=np.array(images_Training3)    	

images_Training4=[]
for name in images_Training4_name[ :N_SAMPLE]:
    img=cv2.imread(Train4_DIR+name)
    try:
    	img_resize=cv2.resize(img, INP_SH [:-1])
    	img_gray=cv2.cvtColor(img_resize,cv2.COLOR_BGR2GRAY).reshape(( INP_SH ))
    	images_Training.append(img_gray)
    	y1.append(7)
    except Exception as e:
    	print(e)
images_Training4=np.array(images_Training4)    	

images_test=np.concatenate([images_Testing1, images_Testing2 ,images_Testing3 , images_Testing4],0)
images_train=np.concatenate([images_Training1,images_Training2,images_Training3,images_Training4],0)
# images=np.concatenate([images_Testing3, images_Training3])
# images=np.concatenate([images_Testing4, images_Training4])
# print(y1)
# quit()
y_ctg2=to_categorical(y)
y=np.array(y_ctg2)


y1_ctg2=to_categorical(y1)
y1=np.array(y1_ctg2)


# print(y_ctg2[10])

# quit()
images_test_nrm=images_test/255.0
images_train_nrm=images_train/255.0
x_test=images_test_nrm
x_train=images_train_nrm
y_test=y1
# x_train=images_nrm
y_train=y
# x_train,x_test,y_train,y_test=train_test_split(images_nrm,y,train_size=0.8) 
cb=EarlyStopping(patience=5,verbose=1,restore_best_weights=True)
for l1_nf in layer1_n_filters:
        for l1_ks in layer1_kernel_sizes:
	           for l2_st in layer2_strides:
                       model=Sequential()
                       model.add(Conv2D(filters=l1_nf,kernel_size=l1_ks,strides=(1,1),activation='relu',padding="same",input_shape= INP_SH ))
                       model.add(Dropout(0.2))
                       model.add(MaxPooling2D((2,2),(2,2)))
                       model.add(Conv2D(filters=64,kernel_size=(7,7),strides=l2_st,padding="same",activation='relu'))
                       model.add(Dropout(0.2))
                       model.add(MaxPooling2D((2,2),(2,2)))
                       model.add(Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding="same",activation='relu'))
                       model.add(MaxPooling2D((2,2),(2,2)))
                       model.add(Flatten())
                       model.add(Dense(256,'relu'))
                       model.add(Dense(64,'relu'))
                       model.add(Dense(8,'relu'))
                       model.add(Dense(4,'relu'))
                       model.add(Dense(y.max()+1,'softmax'))
                       model.compile(optimizer='adam',loss='categorical_crossentropy',metrics="accuracy")
                       print(model.summary())
                       results=model.fit(x_train,y_train,epochs=100,validation_data=(x_test,y_test),callbacks=cb)
                       models_results.append(results)
                       model.save(f"model_Training_Testing{i}.h5")
                       i+=1
for(i,res) in enumerate(models_results):
        plt.plot(res.history["val_loss"],label=f"val loss{i}")
        plt.title("Loss")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
plt.show()