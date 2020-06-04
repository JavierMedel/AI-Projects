import os
import cv2
import sys
import requests
import runpy
import imageio
import shutil
import sqlalchemy as sa
import pandas as pd
import numpy as np
import xml.etree.ElementTree as etree
from datetime import datetime
#from imageai.Detection.Custom import CustomObjectDetection
from skimage.metrics import structural_similarity as ssim
from PIL import Image, ImageSequence
from io import BytesIO

def similarityIteration(df_iter, devices_list, path):
    similarity_calculation = []

    for item in devices_list[:]:
        df_s = df_iter[df_iter['DEVICE_CODE'] == item]
        df_s.reset_index(inplace=True, drop=True)

        lst = similarity_image(df_s, path)
        similarity_calculation += lst
    
    return similarity_calculation

def similarityImage(df_img, path):
    
    similarity_list = []
    
    for i in range(0, len(df_img)):
        path_image         =  path + df_img['IMAGE_ID'][i] + '.jpg'
        try:
            if i == 0:
                #print('First    : --- ', str(i) , ' # ' , df['IMAGE_ID'][i])
                image_2_compare = cv2.imread(path_image)
                similarity_list.append(0)
                continue
            else:
                image_new       = cv2.imread(path_image)

                '''
                # calculate the blur level
                blr_lvl = cv2.Laplacian(image_new, cv2.CV_64F).var()
                print('blr_lvl:', blr_lvl)

                # calculate the brightness level
                img = imageio.imread(path_image)
                brt_lvl = np.mean(img)        
                print('brt_lvl: ' , brt_lvl)
                '''

                # caculate the similarity with the last image
                s_lvl = ssim(image_new, image_2_compare, multichannel=True)
                #print('s_lvl  :' , str(round(s_lvl,2)))
                similarity_list.append(s_lvl)

                image_2_compare = image_new

        except:
            #print('Remove ERROR: *** ', str(i) , ' * ', df['IMAGE_ID'][i])
            similarity_list.append(-3)
            
    return similarity_list

def loadBalerImages(path, object_type, id_image):
    url_image = 'http://40.69.142.165/baler/imageAction.action?imageId=' + id_image

    try:    
        response = requests.get(url_image)
        img = Image.open(BytesIO(response.content))
        path_class = path + '\\' + object_type + '\\' + id_image
        img.save(path_class + '.jpg', 'JPEG')
    except:
        print('Error Image')   
    
def loadVisualImages(path, object_type, id_image):
    
    url_image = 'http://40.69.142.165/visual/imageAction.action?imageId=' + id_image
    
    try:    
        response = requests.get(url_image)
        img = Image.open(BytesIO(response.content))
        path_class = path + '\\' + object_type + '\\' + id_image
    
    except:
        return -1
    
    frame_num = 0
    
    if img.format == 'GIF':
        try:
            for frame in ImageSequence.Iterator(img):
                frame.convert('RGB').save(path_class + '.jpg' , 'PNG')
                break
        except:
            print('>>>> UNPROCESSED IMAGE GIF <<<<<')
            return -1
    else:
        try:
            img.save(path_class + '.jpg', 'JPEG')
        except:
            print('>>>> UNPROCESSED IMAGE JPEG <<<<<')
            return -1
    return 0
    
    
def loadCompologyImages(path, image):
    
    return 'completed'

def loadXMLFile(path, object_type, id_image, bndbox):
    
    path_class = path + '\\' + object_type + '\\' + id_image
    #print('path_class XML:' , path_class)
    
    doc_xml = etree.parse('structure-annotation.xml')
    
    doc_xml.find("folder").text    = object_type
    doc_xml.find("filename").text  = id_image
    doc_xml.find("path").text      = path
    
    img     = cv2.imread(path_class + '.jpg')
    width   = img.shape[1]
    height  = img.shape[0]
    
    doc_xml.find("size/width").text  = str(img.shape[1])
    doc_xml.find("size/height").text = str(img.shape[0])
    
    doc_xml = addLabel(doc_xml, object_type, bndbox)
    
    doc_xml.write(path_class + '.xml') 
    
    return id_image + '.xml'    

def addLabel(doc_xml, object_type, bndbox):
    xmin   = str(bndbox[0])
    ymin   = str(bndbox[1])
    xmax   = str(bndbox[2])
    ymax   = str(bndbox[3])
    
    root   = doc_xml.getroot()
    object = etree.SubElement(root, 'object')
    etree.SubElement(object, 'name').text = object_type
    etree.SubElement(object, 'pose').text = 'Unspecified'
    etree.SubElement(object, 'truncated').text = '0'
    etree.SubElement(object, 'difficult').text = '0'

    bndbox = etree.SubElement(object, 'bndbox')
    etree.SubElement(bndbox, 'xmin').text = xmin
    etree.SubElement(bndbox, 'ymin').text = ymin
    etree.SubElement(bndbox, 'xmax').text = xmax
    etree.SubElement(bndbox, 'ymax').text = ymax
    
    return doc_xml

def resize_directory(path_src, width = 640, height = 480):
    
    list_jpgs = os.listdir(path_src + '\\')
    list_jpgs = list(filter(lambda x: x[-4:] == '.jpg', list_jpgs))
    
    for i in range(len(list_jpgs)):
        img_path_src = path_src + '\\' + list_jpgs[i]
        img_src      = cv2.imread(img_path_src)
        
        height_src, width_src, channels_src = img_src.shape
        #print(height_src, width_src, channels_src)
        
        dim = (width, height)
        
        img_src_out = cv2.resize(img_src, dim)
        cv2.imwrite(img_path_src, cv2.cvtColor(img_src_out, cv2.COLOR_RGB2BGR))    
        

def imageProcessing(path_src, path_trg, object_type, image_src):
    
    list_jpgs = os.listdir(path_trg + '\\' + object_type)
    list_jpgs = list(filter(lambda x: x[-4:] == '.jpg', list_jpgs))
    
    path_image_src = path_src + '\\' + object_type + '\\' + image_src + '.jpg'
    #print(path_image_src)
    img_src        = cv2.imread(path_image_src)
    
    # calculate the blur level
    blr_lvl = cv2.Laplacian(img_src, cv2.CV_64F).var()
    if (blr_lvl <= 35):
        print('blr_lvl:', blr_lvl)
        return -1
    
    # calculate the brightness level
    img = imageio.imread(path_image_src)
    brt_lvl = np.mean(img)
    if (brt_lvl <= 35) | (brt_lvl >= 135):
        print('brt_lvl: ' , brt_lvl)
        return -2
       
    for jpg in list_jpgs:
        path_image_trg = path_trg + '\\' + object_type + '\\' + jpg
        img_trg        = cv2.imread(path_image_trg)
        
        s_lvl = ssim(img_src, img_trg, multichannel=True)

        if s_lvl >= 0.65:
            #print(image_src, ' --- ' , jpg)
            print('s_lvl  :' , str(round(s_lvl , 2)))
            return -3
    
    return 0

def similarity_directory(path_src, path_trg, path_rep, path_opt):
    
    pass                              
    

def blur_bright_validation_directory(src_path, trg_path):
    list_jpgs = os.listdir(src_path)
    list_jpgs = list(filter(lambda x: x[-4:] == '.jpg', list_jpgs))
    
    for i in range(len(list_jpgs)):
        
        if (i % 100) == 0:
            print(round((i / len(list_jpgs)),2), ' %',)
        
        path_image_src = src_path + '\\' + list_jpgs[i]
        img_src        = cv2.imread(path_image_src, cv2.IMREAD_COLOR)
        
        '''
        cv2.imshow('image',img_src)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        #print(path_image_src)
        
        path_image_trg = trg_path + '\\' + list_jpgs[i] 
        
        # calculate blur level
        blr_lvl = cv2.Laplacian(img_src, cv2.CV_64F).var()
        if (blr_lvl <= 35):
            print(list_jpgs[i], ' - ' , 'blr_lvl:', blr_lvl)
            #return -1
            
            shutil.move(path_image_src,
                path_image_trg)
            
            continue

        # calculate brightness level
        img = imageio.imread(path_image_src)
        brt_lvl = np.mean(img)
        #print(list_jpgs[i], ' - ' , 'brt_lvl: ' , brt_lvl)
        if (brt_lvl <= 55) | (brt_lvl >= 185):
            print(list_jpgs[i], ' - ' , 'brt_lvl: ' , brt_lvl)
            #return -2
            
            shutil.move(path_image_src,
                path_image_trg)

def moveImage(path_src, path_trg, object_type, image_src):
    
    path_image_src = path_src + '\\' + object_type + '\\' + image_src + '.jpg'
    path_image_trg = path_trg + '\\' + object_type + '\\' + image_src + '.jpg'
    
    shutil.move(path_image_src,
                path_image_trg)
        
    return 0

def moveXML(path_src, path_trg, object_type, xml_src):
    
    path_xml_src = path_src + '\\' + object_type + '\\' + xml_src + '.xml'
    path_xml_trg = path_trg + '\\' + object_type + '\\' + xml_src + '.xml'
    
    shutil.move(path_xml_src,
                path_xml_trg)
        
    return 0

def copyImage(path_src, path_trg, object_type, image_src):
    
    path_image_src = path_src + '\\' + object_type + '\\' + image_src + '.jpg'
    path_image_trg = path_trg + '\\' + object_type + '\\' + image_src + '.jpg'
    
    shutil.copyfile(path_image_src,
                path_image_trg)
        
    return 0

def copyXML(path_src, path_trg, object_type, xml_src):
    
    path_xml_src = path_src + '\\' + object_type + '\\' + xml_src + '.xml'
    path_xml_trg = path_trg + '\\' + object_type + '\\' + xml_src + '.xml'
    
    shutil.copyfile(path_xml_src,
                path_xml_trg)
        
    return 0



