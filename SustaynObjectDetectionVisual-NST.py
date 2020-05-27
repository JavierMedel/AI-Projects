import tensorflow as tf
tf.get_logger().setLevel('INFO')

import warnings
warnings.filterwarnings("ignore")

#Import Dependencies
import os
import cv2
import sys
import requests
import runpy
import imageio
import sqlalchemy as sa
import pandas as pd
import numpy as np
from datetime import datetime
from imageai.Detection.Custom import CustomObjectDetection
from skimage.metrics import structural_similarity as ssim
from PIL import Image, ImageSequence
from io import BytesIO

print('{}    {}'.format('' + datetime.now().strftime("%H:%M:%S.%f"), 'Object Detection Model for Visual Devices is initializing...'))

path_images = 'O:\\ObjectDetection\\Visual\\'
#path_images = 'C:\\Users\\jmedel\\OneDrive - Avangard Innovative\\coding\\Object Detection\\OB_images\\'

model = 'Visual_050120_v9_detection_model-ex-001--loss-0011.784' + '.h5'
json  = 'Visual_050120_v9_detection_config_7_objects' + '.json'

# load the mapping file
print('{}    {}'.format('' + datetime.now().strftime("%H:%M:%S.%f"), 'Model loaded.... Reading Setting...'))
detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(model)
detector.setJsonPath(json)
detector.loadModel()

print('{}     {}'.format('' + datetime.now().strftime("%H:%M:%S.%f"), model))
print('{}     {}'.format('' + datetime.now().strftime("%H:%M:%S.%f"), json))
print('{}    {}'.format('' + datetime.now().strftime("%H:%M:%S.%f"), 'Creating Object Detection Function...'))

def detect_object(input_image, path_image):
    try:
        #print("image to detect:" + path_images + input_image + ".jpg")
        detections = detector.detectObjectsFromImage(input_image = path_image + input_image + ".jpg", 
                                               output_image_path = path_image + input_image + "-output.jpg",
                                               minimum_percentage_probability = 70)
        
        img = cv2.imread(path_image + input_image + '.jpg', cv2.IMREAD_COLOR)
        height, width, depth = img.shape
        imgScale = 2
        newX , newY = img.shape[1] * imgScale, img.shape[0] * imgScale
        img = cv2.resize(img,(int(newX),int(newY)))
        
        list_probabilities = []
        object_list = []
        if len(detections) > 0:
            #iterate to create a list with all the probabilities
            for detection in detections[:]:
                list_probabilities.append(round(detection["percentage_probability"],2))
            
            # get the index with the max percentage_probability
            max_index = np.argmax(list_probabilities)
            
            object_list.append([detections[max_index]["name"]
            ,round(detections[max_index]["percentage_probability"],2)
            ,detections[max_index]["box_points"][0]
            ,detections[max_index]["box_points"][1]
            ,detections[max_index]["box_points"][2]
            ,detections[max_index]["box_points"][3]
            ,None])
            
            cv2.rectangle(img
                         ,(int(detections[max_index]["box_points"][0] * imgScale), int(detections[max_index]["box_points"][1] * imgScale))
                         ,(int(detections[max_index]["box_points"][2] * imgScale), int(detections[max_index]["box_points"][3] * imgScale))
                         ,[0, 255, 255] #define yellow color
                         , 2)
            '''cv2.putText(img
            ,detection["name"] + ': ' + str(round(detection["percentage_probability"],2))
            ,(int(detection["box_points"][0] * imgScale) , int((detection["box_points"][1] * imgScale) - 7))
            ,cv2.FONT_HERSHEY_DUPLEX, .5, (255, 255, 255), 2)'''
            
            cv2.imwrite(path_images + '{}_{}x{}.jpg'.format(input_image, str(newX), str(newY)), img)
            
            return object_list
        else:
            object_list = [['NO OBJECT',0,0,0,0,0,None]]
            return object_list
    except:
        print(sys.exc_info())
        object_list = [[None,0,0,0,0,0,'UNPROCESSED']]
        return object_list
    
print('{}    {}'.format('' + datetime.now().strftime("%H:%M:%S.%f"), 'Creating Load Image Function...'))

def load_image(image_URL, image_path, image_id):
    
    image_URL = image_URL + image_id
    
    try:
        response = requests.get(image_URL)
        img = Image.open(BytesIO(response.content))
        img.convert('RGB').save(image_path + image_id + '.jpg', 'JPEG')
        return 0
    except:
        #print(sys.exc_info())
        #print('>>>   UNPROCESSED URL : ', image_id )
        return -1

print('{}    {}'.format('' + datetime.now().strftime("%H:%M:%S.%f"), 'Creating SQL engine...'))

def sql_query(ENGINE):
    #
    SQL = '''
    SELECT 
        VI.DevicePackageID AS IMAGE_ID
        ,VI.DeviceType AS DEVICE_TYPE
        ,VI.device_id AS DEVICE_ID
        ,VI.DeviceCode AS DEVICE_CODE
        ,VI.PACKAGE_DATE AS PACKAGE_DATE
    FROM
        sustayn.v_ml_object_detection_detail OB
        RIGHT JOIN sustayn.v_ml_visual_package_data VI
        ON OB.IMAGE_ID = VI.DevicePackageID
    WHERE
        OB.IR_PROCESS_DATE IS NULL
        AND VI.PACKAGE_DATE >= DATE_SUB(CURRENT_TIMESTAMP, INTERVAL 7 DAY)
    ORDER BY 
        VI.PACKAGE_DATE DESC
    LIMIT 500;
    '''
    #
    print('--------------------------------' )
    print('query to get source images list: {}'.format( SQL ) )
    print('--------------------------------' )
    #
    return pd.read_sql_query(SQL, ENGINE)

def sql_query_previous(ENGINE, image_id):
    SQL = '''
    select 
        img_url
    from 
        sustayn.v_ml_visual_package_data
    where 
        img_url != '{}'
        and DeviceCode = (select DeviceCode
                            from sustayn.v_ml_visual_package_data
                            where img_url = '{}')
        and package_date <= (select package_date
                            from sustayn.v_ml_visual_package_data
                            where img_url = '{}')
    order by
        package_date desc
    limit 1;'''.format(image_id, image_id, image_id)

    return pd.read_sql_query(SQL, ENGINE)

def similarity_calculation(image_last, image_prev, path_image, visual_url):
   
    if load_image(visual_url, path_image, image_prev) <= -1:
        return 0
    
    try:
        img_last = cv2.imread(path_image + image_last + '.jpg')
        img_prev = cv2.imread(path_image + image_prev + '.jpg')

        ssim_lvl = ssim(img_last, img_prev, multichannel=True)
    except:
        print(sys.exc_info())
        ssim_lvl = 1
        
    return round(ssim_lvl,2)  
    
def blur_bright(image, path_image):
    img     = cv2.imread(path_image + image + '.jpg')
    blr_lvl = cv2.Laplacian(img, cv2.CV_64F).var()
    
    img = imageio.imread(path_image + image + '.jpg')
    brt_lvl = np.mean(img)

    return round(blr_lvl,2), round(brt_lvl,2)


# In[3]:


print('{}    {}'.format('' + datetime.now().strftime("%H:%M:%S.%f"), 'Reading data from SQL...'))
ENGINE = sa.create_engine('mysql+mysqldb://mercenary:Flxi8571@40.69.142.165:3306/Sustayn', pool_recycle=120) # NST02
image_IDs = sql_query(ENGINE)

#order the records by DEVICE_CODE and time created
image_IDs.sort_values(by=['DEVICE_CODE', 'PACKAGE_DATE'], ascending=True, inplace=True)

images_objests_list = []

visual_url = 'http://40.69.142.165/visual/imageAction.action?imageId=' #visual URL image
imgIDX = 0

#Iterate through df and make predictions
print('{}    {} {}'.format('' + datetime.now().strftime("%H:%M:%S.%f"), len(image_IDs), 
                           'records read... Starting Prediction cycle...'))

for j in range(len(image_IDs)):
    
    image_response = load_image(visual_url, path_images, image_IDs['IMAGE_ID'][j])
    
    blr_lvl, brt_lvl, ssim_lvl = 0, 0, 0
    
    if image_response < 0:        
        object_list = [[None,0,0,0,0,0,'NON LOADED']]
        
    else:
        #calculate the brithness and bluerness of the image
        blr_lvl, brt_lvl = blur_bright(image_IDs['IMAGE_ID'][j], path_images)
        
        #get the previous image
        prev_img = sql_query_previous(ENGINE, image_IDs['IMAGE_ID'][j]) 
        
        # calculate the similarity with the previous image
        ssim_lvl = similarity_calculation(image_IDs['IMAGE_ID'][j], prev_img['img_url'].loc[0], path_images, visual_url)
        
        # validate if the image is usefull
        if (brt_lvl <= 25) | (brt_lvl >= 145):
            object_list = [[None,0,0,0,0,0,'BRIGHTNESS']]
        elif (blr_lvl <= 25):
            object_list = [[None,0,0,0,0,0,'BLURRINESS']]
        elif (ssim_lvl >= 0.50):
            object_list = [[None,0,0,0,0,0,'SIMILARITY']]
        else:
            object_list = detect_object(image_IDs['IMAGE_ID'][j], path_images)
            
    for k in range(len(object_list)):
        images_objests_list.append([image_IDs['IMAGE_ID'][j],
                                   object_list[k][6],
                                   object_list[k][0], object_list[k][1], object_list[k][2],
                                   object_list[k][3], object_list[k][4], object_list[k][5],
                                   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                   ssim_lvl,
                                   image_IDs['DEVICE_TYPE'][j],
                                   image_IDs['DEVICE_ID'][j],
                                   image_IDs['DEVICE_CODE'][j],
                                   image_IDs['PACKAGE_DATE'][j],
                                   blr_lvl,
                                   brt_lvl])
        
        print("{}: {}| {}| {}| {}| {}| {}| {}| {}| {}| {}| {}| {}".format( imgIDX, datetime.now().strftime("%H:%M:%S.%f"),
                                                       image_IDs['IMAGE_ID'][j],
                                                       object_list[k][6],
                                                       object_list[k][0], 
                                                       ssim_lvl,
                                                       object_list[k][1], object_list[k][2], 
                                                       object_list[k][3], object_list[k][4], object_list[k][5],
                                                       blr_lvl,
                                                       brt_lvl))
    imgIDX = imgIDX + 1
    
    #os.remove('images_processed\\' + image_IDs[j] + '.jpg')
    #os.remove('images_processed\\' + image_IDs[j] + '-output.jpg')
    
print("")
print("\n{} images processed".format(imgIDX))

columns =  ['IMAGE_ID','PROCESS_MSG','OBJECT_DETECTED','IR_CONFIDENCE','XMIN','YMIN','XMAX','YMAX','IR_PROCESS_DATE','SIMILARITY','DEVICE_TYPE','DEVICE_ID','DEVICE_CODE','ORIGIN_IMG_DATE','BLURRINESS','BRIGHTNESS']

df = pd.DataFrame(images_objests_list, columns=columns)

sizes = df.groupby(['IMAGE_ID'], sort=False).size().values
df['OBJECT_INDEX'] = (np.arange(sizes.sum()) - np.repeat(sizes.cumsum() - sizes, sizes)) + 1

df = df[['IMAGE_ID','OBJECT_INDEX','OBJECT_DETECTED','IR_CONFIDENCE','XMIN','YMIN','XMAX','YMAX','IR_PROCESS_DATE','SIMILARITY','DEVICE_TYPE','DEVICE_ID','DEVICE_CODE','BLURRINESS','BRIGHTNESS','PROCESS_MSG','ORIGIN_IMG_DATE']]

df.to_sql(con=ENGINE, if_exists='append', name='ml_object_detection_detail', index=False)

path = 'E:\\Analytics\\Images_Recognition\\'
runpy.run_path(path_name = path + 'SustaynVisualEmailNotification.py')

print('{}    {}'.format('' + datetime.now().strftime("%H:%M:%S.%f"), 'Sending Image Notification Email...'))

print('{}    {}'.format('' + datetime.now().strftime("%H:%M:%S.%f"), 'Notification Email Sent!'))

print('{}    {}'.format('' + datetime.now().strftime("%H:%M:%S.%f"), 'Script Completed!!!'))
