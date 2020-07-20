import sys
import requests
import pandas as pd
import numpy as np
import sqlalchemy as sa
from PIL import Image
from io import BytesIO
from datetime import datetime
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

def load_image(imageURL):
    action_image = ''

    try:
        response = requests.get(imageURL)
        img = Image.open(BytesIO(response.content))
        if img.format == 'GIF':
            if img.tile[0][1][-2:] == img.size:
                img = img.convert('RGB')
                img = img.resize((224, 224))
                img = img_to_array(img)
            else:
                img = np.zeros((224, 224, 3))
                action_image = 'UNPROCESSABLE IMAGE'
        else:
            img = img.convert('RGB')
            img = img.resize((224, 224))
            img = img_to_array(img)

    except Exception as spe:
        action_image = 'UNPROCESSABLE IMAGE'
        print('>>>> {} <<<<<'.format(acction_image))
        print(spe)

        img = np.zeros((224, 224, 3))
        img = img_to_array(img)
    img = img.reshape(1, 224, 224, 3)
    return img, action_image

def baler_validation(label_device, label_model):
    if label_device == label_model:
        return 'VALID'
    else:
        return 'INVALID'


print('{}    {}'.format('' + datetime.now().strftime("%H:%M:%S.%f"),
                        'SustaynMaterialClassification-NST script is initializating...'))

model_name_baler = 'Sustayn-Image-Classification-Baler-Trained_Model-061622-03-Classes-5700-Samples.h5'

model_location_baler = './' + model_name_baler

time_before_load_model = '' + datetime.now().strftime("%H:%M:%S.%f")
print('{}    {} {}'.format(time_before_load_model, 'Loading Model for Baler devices:', model_name_baler))

print(model_location_baler)

model_baler = load_model(model_location_baler)

print('{}    {}'.format('' + datetime.now().strftime("%H:%M:%S.%f"), 'Model loaded.... Reading Mapping categories...'))

categories_baler = ['BALED CARDBOARD', 'BALED EMPTY', 'BALED FILM']

# order the categories alphabetically
categories_baler = sorted(categories_baler)

SQL = '''
SELECT lower(B.devicetype) as DeviceType,
    B.img_url,
    B.device_id,
    B.DeviceCode as Device_code,
    B.material_type,
    B.MaterialDescription_FromOriginal,
    B.MaterialClass_FromOriginal,
    B.package_date
FROM sustayn.ml_material_classification A
    RIGHT OUTER JOIN sustayn.v_ml_baler_package_data B 
    ON A.IMAGE_ID = B.img_url
WHERE B.IMG_URL IS NOT NULL
    AND A.IMAGE_ID IS NULL
    AND B.deviceType IN ('Baler')
    AND B.devicetype IS NOT NULL
ORDER BY B.deviceType, B.PACKAGE_DATE DESC
LIMIT 299;
'''

print('{}    {}'.format('' + datetime.now().strftime("%H:%M:%S.%f"), 'Creating MySQL engine'))

sql_con_str = 'mysql+mysqldb://mercenary:Flxi8571@40.69.142.165:3306/Sustayn'  # NST02 / PROD
ENGINE = sa.create_engine(sql_con_str, pool_recycle = 7200)  # NST01

print('{}    {}	{}'.format('' + datetime.now().strftime("%H:%M:%S.%f"), 'Reading data from MySQL:', ENGINE))

df = pd.read_sql_query(SQL, ENGINE)  # Read SQL query into a data frame
list_df = []

print('{}    {} {}'.format('' + datetime.now().strftime("%H:%M:%S.%f"), len(df), 'records read.'))
print('{}    {} '.format(datetime.now().strftime('%H:%M:%S.%f'), 'Starting Prediction cycle...'))
print('Process Time | Check | Form | Prediction | URL | Package Date')

URL_baler = 'http://40.69.142.165/baler/imageAction.action?imageId='

for count in range(len(df)):  # Loop from zero through the length of the dataframe
    check = -1
    prediction = ''
    action_image = ''
    result_validation = ''
    confidence_val = 0
    image_url = ''
    predicted_category = ''
    time_execution = ''

    # Process images from baler
    image_url = URL_baler + str(df['img_url'].loc[count])
    img, action_image = load_image(image_url)
    result = model_baler.predict(img)
    predicted_category = categories_baler[np.argmax(result)]
    check = np.argmax(result)
    confidence_val = result[0][check]

    # execute the function to validate baler classification
    action_image = baler_validation(str(df['MaterialClass_FromOriginal'].loc[count]), predicted_category)

    time_execution = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    list_df.append([image_url[-36:],
                    str(df['device_id'].loc[count]),
                    str(df['Device_code'].loc[count]),
                    str(df['DeviceType'].loc[count]),
                    str(df['material_type'].loc[count]),
                    str(df['MaterialClass_FromOriginal'].loc[count]),
                    predicted_category,
                    confidence_val,
                    time_execution,
                    action_image])

    print('{}| {}| {}| {} | {} |{}'.format(datetime.now().strftime('%H:%M:%S.%f'),
                                           count,
                                           df['DeviceType'].loc[count],
                                           predicted_category,
                                           image_url,
                                           df['package_date'].loc[count]))

try:
    print('{}| {} '.format(datetime.now().strftime('%H:%M:%S.%f'), 'Creating Data frame...'))

    columns_fields = ['IMAGE_ID', 'DEVICE_ID', 'DEVICE_CODE', 'DEVICE_TYPE', 'ORIGINAL_MATERIAL_TYPE', 'ORIGINAL_CLASS',
                      'IR_CLASS', 'IR_CONFIDENCE', 'IR_PROCESS_DATE', 'IR_ACTION']

    pd_response = pd.DataFrame(list_df, columns = columns_fields)
    pd_response.to_sql('ml_material_classification', ENGINE, if_exists = 'append', index = False)
    print('{}| {} {}'.format(datetime.now().strftime("%H:%M:%S.%f"), len(pd_response),
                             ' Records inserted in the Database '))
except Exception as spe:
    print('{}| {} {}'.format(datetime.now().strftime("%H:%M:%S.%f"), 'Error while trying to insert into the Database',
                             spe))

print('{}| {}'.format(datetime.now().strftime("%H:%M:%S.%f"),
                      'Python script execution completed...'))  # end of the execution