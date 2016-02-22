# MLOnSparkProject

## generate_image_input.py

Generates a csv file readable by Spark. Each row represents an image with associated meta-data. 
One image is stored per row. The image is assumed to be 64x64 pixels with one channel (grayscale). Pixel intensity is uint8 type (0-255). The pixels in the 2d image are unwrapped in order to write them to one row. Use nparray.reshape((64,64)) on the appropriate columns to regenerate the 2d array.

Columns

* 0. User Id. Corresponds to folder name (1-500 for train) (501-700 for validation)
* 1. Slice Id. The number in sax_### folder
* 2. Time Id. 1-30. The time index associated with the image
* 3. Age
* 4. Gender
* 5. Slice location
* 6. Slice thickness
* 7-4102. Pixel intensity (0-255 grayscale) 
