# Run this script to update mask column in metadata_withmasks.csv file.

import pandas as pd
import os

# Create pandas dataframe from raw metadata file
df = pd.read_csv('..' + os.sep + 'data' + os.sep + 'meta_data' + os.sep + 'raw' + os.sep + 'metadata.csv')

# List of mask names
mask_ids = os.listdir('..' + os.sep + 'data' + os.sep + 'images' + os.sep + 'mask' + os.sep)

# Update dataframe from masks in mask_ids
for i, row in df.iterrows():
    im_id = row['img_id']
    mask_id = im_id[:-4] + '_mask.png'
    if mask_id in mask_ids:
        df.at[i, 'mask'] = 1
    else:
        df.at[i, 'mask'] = 0

df['mask']=df['mask'].astype('int')

# Remove old metadata_withmasks.csv file and write new one
os.remove('..' + os.sep + 'data' + os.sep + 'meta_data' + os.sep + 'metadata_withmasks.csv')
df.to_csv('..' + os.sep + 'data' + os.sep + 'meta_data' + os.sep + 'metadata_withmasks.csv')
