# MLOnSparkProject

## generate_image_input.py

Generates a csv file readable by Spark. Each row represents an image with associated meta-data. 
One image is stored per row. The image is assumed to be 64x64 pixels with one channel (grayscale). Pixel intensity is uint8 type (0-255). The pixels in the 2d image are unwrapped in order to write them to one row. Use nparray.reshape((64,64)) on the appropriate columns to regenerate the 2d array.

Columns in file train_input.csv

* 0. User Id. Corresponds to folder name (1-500 for train) (501-700 for validation)
* 1. Slice Id. The number in sax_# folder
* 2. Time Id. 1-30. The time index associated with the image
* 3. Age
* 4. Gender
* 5. Slice location
* 6. Slice thickness
* 7. Area Multiplier (= mm^2/pixel^2)
* 8-4103. Pixel intensity (0-255 grayscale) 
* 4103. Systole volume
* 4104. Diastole volume

Columns in file train_input_meta_only.csv

* 0. User Id. Corresponds to folder name (1-500 for train) (501-700 for validation)
* 1. Slice Id. The number in sax_# folder
* 2. Time Id. 1-30. The time index associated with the image
* 3. Age
* 4. Gender
* 5. Slice location
* 6. Slice thickness
* 7. Systole volume (label)
* 8. Diastole volume (label)

Columns in file centers_area.csv

* 0. User Id. Corresponds to folder name (1-500 for train) (501-700 for validation)
* 1. Slice Id. The number in sax_# folder
* 2. Time Id. 1-30. The time index associated with the image
* 3. Area prediction of image (units of pixel^2)
* 4. X coordinate of left ventricle center (in pixels)
* 5. X coordinate of left ventricle center (in pixels)
* 6. Radius of circular region of interest surrounding left ventricle (in pixels)

