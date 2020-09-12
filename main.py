import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import xlwt
from fpdf import FPDF
from PIL import Image





import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression



tf.reset_default_graph()

#Test and Train data DIR set
TRAIN_DIR='E:/FIVERR/stefanovic92-corrosionDetection/scratchMetalDetector/train'
TEST_DIR='E:/FIVERR/stefanovic92-corrosionDetection/scratchMetalDetector/test'


test_data_size=10 #number of images to be tested
IMG_SIZE=60
LR=1e-3  #Learning Rate
num_of_epoch=10 #incresing epoch will increase the accuracy and increase the compilation tim(slow process)




#image placement on the PDF
convoluted_img_x=10
convoluted_img_y=8
original_img_x=60
original_img_y=8

#image name array
image_list=[]
image_name_list=[]

MODEL_NAME='Corrosion-Detection-{}-{}.model'.format(LR,'6conv-basic-net')

def label_img(img):
	#N.1.JPG or Y.1.jpg
	word_label=img.split('.')[-3]
	if word_label=='N':return [1,0]
	elif word_label=='Y': return[0,1]
 
def create_train_data():
	training_data=[]
	for img in tqdm(os.listdir(TRAIN_DIR)): 
		label=label_img(img)
		path=os.path.join(TRAIN_DIR,img)
		img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
		training_data.append([np.array(img),np.array(label)])
	shuffle(training_data)
	np.save('train_data.npy',training_data)
	return training_data

def process_test_data():
	testing_data=[]
	for img in tqdm(os.listdir(TEST_DIR)):
		path=os.path.join(TEST_DIR,img)
		image_name_list.append(img)
		img_num=img.split('.')[0]
		img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
		testing_data.append([np.array(img),img_num])
	np.save('test_data.npy',testing_data)
	return testing_data

train_data=create_train_data()


###############################start of 6ConvLayers##########################

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)




convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)


convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)




convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet,tensorboard_dir='log')

########################end of 6conv Layers##################################


if os.path.exists('{}.meta'.format(MODEL_NAME)):
	model.load(MODEL_NAME)
	print('model loaded')

train=train_data[:]
test=train_data[:]

X= np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y=[i[1] for i in train]

test_x= np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y=[i[1] for i in test]

 
model.fit({'input': X}, {'targets': Y}, n_epoch=num_of_epoch, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=1, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)

#if you you dont have file
test_data=process_test_data()

#if you have the file
# test_data=np.load('test_data.npy')



#prediction update to excel sheet
book = xlwt.Workbook(encoding="utf-8")

sheet1 = book.add_sheet("Sheet 1")
style = xlwt.XFStyle()

# font
font = xlwt.Font()
font.bold = True
style.font = font 

sheet1.write(0, 0, "Images")
sheet1.write(0, 1, "Prediction")




#Reslut show window
fig=plt.figure()
i=1
j=0
test_img_counter=0


# Instantiation of inherited class
pdf = FPDF('P','mm',(100,100))
pdf.set_font('Times', '', 12)

image_names = [f for f in os.listdir(TEST_DIR) if os.path.splitext(f)[-1] == '.jpg']

for num,data in enumerate(test_data[:test_data_size]):
	#N:[1,0]
	#Y:[0,1]
	img_num=data[1]
	img_data=data[0]
	y=fig.add_subplot(3,4,num+1)
	orig=img_data
	data=img_data.reshape(IMG_SIZE,IMG_SIZE,1)

	model_out=model.predict([data])[0]
	if np.argmax(model_out)==1: str_label='Corrode'
	else: str_label='Non Corrode'


	sheet1.write(i, 0, i)
	sheet1.write(i, 1,str_label)
	image_list.append(i)
	i=i+1
	
	y.imshow(orig,cmap='gray')
	cv2.imwrite("predictionImage"+str(i)+".png", orig)

	#pdf creation
	pdf.add_page()
	pdf.image('predictionImage'+str(i)+'.png', convoluted_img_x,convoluted_img_y , 33)
	pdf.text(20,70,"original_image_name : " + image_name_list[j])
	pdf.text(20,60, "image_number : " + str(i-1))
	pdf.text(20,50, "prediction : "+str_label)
	pdf.image("./test/"+image_name_list[j], original_img_x,original_img_y , 33)


	plt.title(str_label)
	y.axes.get_xaxis().set_visible(False)
	y.axes.get_yaxis().set_visible(False)
	test_img_counter=test_img_counter+1
	j=j+1


plt.show()

#excel and pdf saving
book.save("trial.xls")
pdf.output('prediction.pdf', 'F')







