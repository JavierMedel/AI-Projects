import pickle
import mysql.connector
import pandas as pd
import numpy as np
import sqlalchemy as sa
from datetime import datetime


# function to audit duplicates in a group
def find_duplicates_in_groups(temp_df):
    temp_df['duplicate_parent_id'] = 'unique'
    temp_df['audit_status_group'] = np.where(
        (temp_df['audit_status_pred'] == 'A') & (temp_df['audit_status_imgs'] == 'A'), 'A', 'R')

    if len(temp_df) == 1:
        return temp_df

    print('---------------------- ', temp_df['device_id'].unique(), ' -- ', len(temp_df))

    group_flag = False
    lower_pointer = 0
    upper_pointer = 0
    second_val = True

    for i in range(len(temp_df)):
        if temp_df['audit_duplicates_groups'].iloc[i] == 0:
            if group_flag == False:
                group_flag = True
                lower_pointer = i
        else:
            if group_flag == True:
                group_flag = False
                upper_pointer = i

                temp_df[lower_pointer + 1:upper_pointer + 1]['duplicate_parent_id'] = temp_df['id'].iloc[lower_pointer]
                temp_df[lower_pointer: lower_pointer + 1]['duplicate_parent_id'] = 'parent'
                temp_df[lower_pointer: upper_pointer + 1]['count_duplicates_groups'] = len(
                    temp_df[lower_pointer: upper_pointer + 1])

                print('lower_pointer', lower_pointer, ': upper_pointer', upper_pointer)
                print('Number of items duplicated: ', len(temp_df[lower_pointer: upper_pointer + 1]))
                print(temp_df[lower_pointer: upper_pointer + 1][['id',
                                                                 'audit_status_pred',
                                                                 'audit_status_imgs',
                                                                 'audit_status_group',
                                                                 'package_date_f',
                                                                 'time_delta_f']])

                audit_pre = list(temp_df[lower_pointer:upper_pointer + 1]['audit_status_pred'])
                audit_img = list(temp_df[lower_pointer:upper_pointer + 1]['audit_status_imgs'])

                if ('A' in audit_pre) & ('A' in audit_img):

                    print('Find the best candidate in the group of records.')

                    # --------------------
                    # Scenario 1:
                    # The record masked as 'A' in the audit_pred is the best candidate
                    # --------------------
                    print('Trying 1 scenario...')
                    for j in range(lower_pointer, upper_pointer + 1):
                        if ((temp_df.iloc[j]['audit_status_pred'] == 'A') &
                                (temp_df.iloc[j]['label_f'] == 1) &
                                (temp_df.iloc[j]['image_f'] == 1) &
                                (temp_df.iloc[j]['standard_weight_f'] >= 80)):
                            print('Just accept one record... Scenario 1')
                            temp_df.iloc[lower_pointer:upper_pointer + 1, -1] = 'D'
                            temp_df.iloc[j, -1] = 'A'
                            second_val = False
                            break

                    # --------------------
                    # Scenario 2:
                    # When in the audit_pred there is not a good candiate
                    # and we need to select one from the audit_img list
                    # --------------------
                    if second_val == True:
                        print('Trying 2 scenario...')
                        for j in range(lower_pointer, upper_pointer + 1):
                            if ((temp_df.iloc[j]['label_f'] == 1) &
                                    (temp_df.iloc[j]['image_f'] == 1) &
                                    (temp_df.iloc[j]['standard_weight_f'] >= 80)):
                                print('Just accept one record... Scenario 2')
                                temp_df.iloc[lower_pointer:upper_pointer + 1, -1] = 'D'
                                temp_df.iloc[j, -1] = 'A'
                                break

                    print(temp_df[lower_pointer: upper_pointer + 1][['id',
                                                                     'audit_status_pred',
                                                                     'audit_status_imgs',
                                                                     'audit_status_group',
                                                                     'package_date_f',
                                                                     'time_delta_f']])
                else:

                    # --------------------
                    # Scenario 3:
                    # If there is not any valid record from the ML model and Image Classification Model
                    # then mask as reject all the records 'R'
                    # --------------------
                    print('Reject all the records and marks a duplicates...')
                    temp_df.iloc[lower_pointer:upper_pointer + 1, -1] = 'D'
                    temp_df.iloc[lower_pointer, -1] = 'R'

                print('**********************\n')

    return temp_df


# # Get the Dataset
sql_con_str = 'mysql+mysqldb://mercenary:Flxi8571@40.69.142.165:3306/Sustayn'  # NST02 / PRO
# sql_con_str = 'mysql+mysqldb://mercenary:Flxi8571@52.173.202.38:3306/Sustayn'  # NST01 / DEV
ENGINE = sa.create_engine(sql_con_str, pool_recycle=3600)

# Stract the data from the DataBase


# Query for first execution

SQL1 = """
SELECT b.*
FROM sustayn.v_ml_baler_productor_history b 
WHERE b.package_date >= '2020-01-01 00:00:00'
AND b.ir_class IS NOT NULL
UNION
SELECT DeviceType, Null, Null, max(package_date), device_id, Null, Null, Null, Null, NUll, Null, NUll, 'KG', 'AAA', Null, Null, Null, Null
FROM sustayn.v_ml_baler_productor_history
WHERE 1 = 1
and package_date <= '2020-01-01 00:00:00'
GROUP BY DeviceType, device_id;"""

# Query for daily execution

SQL = """
SELECT b.*
FROM sustayn.v_ml_baler_productor_history b
LEFT JOIN sustayn.ml_baler_auto_audit a ON b.id = a.id 
WHERE b.package_date >= '2020-01-01 00:00:00'
AND b.ir_class IS NOT NULL
AND a.id IS NULL
UNION
SELECT 'Baler', Null, Null, max(package_date), device_id, Null, Null, Null, Null, NUll, Null, NUll, 'KG', 'AAA', Null, Null, Null, Null
FROM sustayn.ml_baler_auto_audit
GROUP BY device_id;
"""

# read data from db to DataFrame
df = pd.read_sql_query(SQL1, ENGINE)
print(df.shape)
# # Analyse Dataset

# filling the missing values
df.fillna(0, inplace=True)

# Convert package_date to date time to order for the most recent
df['package_date'] = df['package_date'].astype('datetime64')

# sort dataframe by device_code and package_date
df.sort_values(by=['device_id', 'package_date'], ascending=[True, True], inplace=True, )

# reset index with the new order
df.reset_index(inplace=True, drop=True)


# # Create extra labels
# ## material_description_from_original
# Redefine MaterialDescription to group in 3 categories fewer categories
def material_description(material):
    if 'OCC' in material:
        return 'BALED CARDBOARD'
    elif 'CARTON' in material:
        return 'BALED CARDBOARD'
    elif 'CARDBOARD' in material:
        return 'BALED CARDBOARD'

    elif 'FILM' in material:
        return 'BALED FILM'
    elif 'LDP' in material:
        return 'BALED FILM'
    elif 'PLAYO' in material:
        return 'BALED FILM'
    elif material == 'PLMX':
        return 'BALED FILM'
    elif material == 'SHRINK WRAP':
        return 'BALED FILM'
    elif material == 'BOMA':
        return 'BALED FILM'

    else:
        return 'BALED OTHER'


df['material_description_f'] = df['material_description_from_original'].astype(str).str.upper().apply(
    material_description)

# ## material_description_prev

df['material_description_prev'] = df.groupby(by=['device_id'])['material_description_f'].shift(-1)

df['material_description_prev'] = df['material_description_prev'].fillna(df['material_description_f'])

# ## material_description_after

df['material_description_after'] = df.groupby(by=['device_id'])['material_description_f'].shift(1)

df['material_description_after'] = df['material_description_after'].fillna(df['material_description_f'])

# ## package_date

# Create a new field from the package_date but truncating the secs
df['package_date_f'] = df['package_date'].dt.strftime("%m/%d/%Y %H:%M").astype('datetime64')


# Create a time interval delta
def diff_func(df):
    return abs(df.diff().dt.total_seconds() / 60)


# Now call the function using .apply
df['time_delta_f'] = df.groupby(['device_id'])['package_date'].apply(diff_func)

# Fill in any NaN values
df['time_delta_f'].fillna('0', inplace=True)

# Convert the output into a float
df['time_delta_f'] = pd.to_numeric(df['time_delta_f']).astype(int)


# ## standard_weight

# Standardize all the weights by making everything KGs
def standard_weight(row):
    if row['unit'] == 'LB':
        return round(row['net_weight'] * 0.453592, 0)
    else:
        return row['net_weight']


df['standard_weight_f'] = df.apply(standard_weight, axis=1).astype(int)

# ## barcode to label

# Create an additional field that contains if the bale has a barcode or not.
df['barcode_f'] = np.where(df['barcode'] == 'null', 0, df['barcode'])

df['label_f'] = np.where(df['barcode_f'] == 0, 0, 1)

# ## img_url to image

df['image_f'] = np.where(df['img_url'] == 0, 0, 1)

df.drop(columns=['DeviceType', 'DeviceCode', 'package_id', 'material_type', 'barcode', 'audit_date', 'audit_userid'],
        inplace=True)

# ## duplicates in manual audit

# Duplicates are marked from zero to one
df['audit_duplicates_f'] = df.groupby(by=['device_id'])['time_delta_f'].shift(-1, fill_value=60)
df['audit_duplicates_f'] = np.where(df['audit_duplicates_f'] <= 10, 0, 1)

# ## eliminate records audited

df.drop(df[df['audit_status'] == 'AAA'].index, inplace=True)
df.shape

# ## distinct the group with similar items

df['audit_duplicates_groups'] = np.where(df['material_description_prev'] != df['material_description_f'], 1,
                                         df['audit_duplicates_f'])

# ## create the duplicates_groups_count

df['count_duplicates_groups'] = 1

# # Validad the records with the Image Classification Model

df['audit_status_imgs'] = np.where(df['ir_class'] != df['material_description_f'], 'R', 'A')
df['audit_status_imgs'].value_counts()

# # Load Model
pickle_in = open('./Baler_062920_ML_Classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

# # Dataset to predict
df_src = df[['device_id',
             'id',
             'label_f',
             'image_f',
             'time_delta_f',
             'standard_weight_f',
             'material_description_f']]

df_model = pd.get_dummies(df_src,
                          columns=["material_description_f"],
                          drop_first=False)

try:
    df_model.drop(columns=['material_description_f_BALED OTHER'],
                  inplace=True)
except:

    columns_list = df_model.columns

    column_cardboard = 'material_description_f_BALED CARDBOARD'
    column_film = 'material_description_f_BALED FILM'

    if column_cardboard not in columns_list:
        df_model[column_cardboard] = 0

    if column_film not in columns_list:
        df_model[column_film] = 0

# Define the X and Y variables
X = df_model[['label_f',
              'image_f',
              'time_delta_f',
              'standard_weight_f',
              'material_description_f_BALED CARDBOARD',
              'material_description_f_BALED FILM']]

if X.shape[0] < 1:
    print(X.shape)
    print('Do not execute the records')
else:
    print(X.shape)
    pred_y = classifier.predict(X)
    df['audit_status_pred'] = pred_y

    # ## audit result of the group of duplicates
    df = df.groupby(by=['device_id']).apply(find_duplicates_in_groups)

    df['audit_status_group'].value_counts()

    # # Identify the records that could be 'Changed' by human validation
    # Copy the audit_status columns to generated the final prediction
    df['audit_status_valid'] = df['audit_status_group']

    # --------------------
    # Records that could be re-audit will be mark as P
    # --------------------

    df.loc[((df['audit_status_group'] == 'R') &
            (df['audit_status_pred'] == 'A')), 'audit_status_valid'] = 'P'

    # --------------------
    # Records that could be re-audit will be mark as P, second pass
    # --------------------

    df.loc[((df['audit_status_group'] == 'R') &
            (df['audit_status_imgs'] == 'A')), 'audit_status_valid'] = 'P'

    df['audit_status_valid'].value_counts()

    df['audit_userid'] = 'AutoAI'
    df['audit_status_nst'] = df['audit_status_valid']
    df['audit_reason'] = None

    df['audit_status_nst'] = np.where(df['audit_status_valid'] == 'D', 'R', df['audit_status_valid'])

    df['audit_reason'] = np.where(df['audit_status_valid'] == 'D', '8', df['audit_reason'])
    df['audit_reason'] = np.where(df['audit_status_valid'] == 'R', '9', df['audit_reason'])
    df['audit_reason'] = np.where((df['audit_status_valid'] == 'R') & (df['standard_weight_f'] <= 80), '4',
                                  df['audit_reason'])

    df['process_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Reorder the order of the columns
    df = df[['id', 'img_url', 'device_id', 'package_date',
             'package_date_f', 'time_delta_f',
             'material_description_from_original', 'material_description_f',
             'material_description_prev', 'material_description_after',
             'ir_original_class', 'ir_class', 'ir_confidence',
             'unit', 'net_weight', 'standard_weight_f',
             'barcode_f', 'label_f', 'image_f',
             'audit_status_imgs', 'audit_status_pred', 'audit_duplicates_f',
             'audit_duplicates_groups', 'count_duplicates_groups',
             'duplicate_parent_id', 'audit_status_group', 'audit_status_valid',
             'audit_userid', 'audit_status_nst', 'audit_reason', 'process_date',
             'audit_status']]

    df.to_sql(name='ml_baler_auto_audit', con=ENGINE, if_exists='append', index=False)

    print('Execution completed, {} records inserted!'.format(len(df)))
