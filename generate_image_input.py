import os
import csv
import sys
import numpy as np
import dicom
from skimage import transform, img_as_ubyte, exposure
from skimage.morphology import disk
from skimage.filters import rank
from skimage.filters.rank import enhance_contrast
from skimage.restoration import denoise_tv_chambolle
from joblib import Parallel, delayed
from scipy import misc
import math
import json


def mkdir(fname):
   try:
       os.mkdir(fname)
   except:
       pass


def get_file_structure(root_path):
   """Return a JSON with user/slice/filepath hierarchy"""

   def is_number(s):
       try:
           float(s)
           return True
       except ValueError:
           return False

   output = []

   currentId = 0
   for root, dirnames, files in os.walk(root_path):
     currentdir = root.replace("\\","/").rsplit("/", 1)[1]

     if is_number(currentdir):     
       currentId = int(float(currentdir))
       print("Walking files for user {0}".format(currentId))
       userdict = {}
       userdict["UserId"] = str(currentId)
       userdict["Slices"] = []
       output.append(userdict)

     if ("sax" in currentdir):
       slicedict = {}
       slicedict["SliceId"] = currentdir.split("_")[1]
       slicedict["Files"] = sorted([x for x in files if ".dcm" in x], key = lambda f: int(float(f.split("-")[2].split(".")[0])))
       userdict["Slices"].append(slicedict)     

   output = sorted(output, key = lambda user: int(float(user["UserId"])))

   for userdict in output:
     userdict["Slices"] = sorted(userdict["Slices"], key = lambda slice: int(float(slice["SliceId"])))    

   with open("{0}/file_structure.json".format(root_path),"w") as f:
     json.dump(output, f, indent=2)

   return output  


def generate_csv(userdict_array):

   image_info = Parallel()(delayed(generate_rows_for_user)(user_dict) for user_dict in userdict_array[:NUM_USERS])

   print("All finished, %d images in total" % len(image_info))


def generate_rows_for_user(user_dict):

   userId = user_dict["UserId"]

   print ("UserId: {0}".format(userId))

   for slice_dict in user_dict["Slices"]:

       sliceId = slice_dict["SliceId"]

       for file in slice_dict["Files"]:
          dcm_path = "{0}/{1}/study/sax_{2}/{3}".format(folderpath, userId, slice_dict["SliceId"], file)

          timeId = file.split("-")[2].split(".")[0]

          dcm_img = dicom.read_file(dcm_path)

          img = preprocess(dcm_img)

          # optionally save image
          #save_img(img, dcm_path)

          #unwrap the numpy array into a 1-d array
          img = img.reshape(img.size)

          # DICOM file gives age as ###Letter, where Letter=Y, M, or W
          # eg. 050Y = 50 years old. 048M = 48 months old
          age = int(dcm_img.PatientAge[:3])
          age_unit = dcm_img.PatientAge[-1]
          
          # if M or W, convert to age to years
          if age_unit == "M":
            age = age/12
          elif age_unit == "W":
            age = age/52

          # user meta-data
          row_data = []
          row_data.append(userId)
          row_data.append(sliceId)
          row_data.append(int(timeId.lstrip("0")))  # eg. convert 0012 to 12
          row_data.append(age)
          row_data.append(dcm_img.PatientSex)
          row_data.append(int(math.floor(dcm_img.SliceLocation)))
          row_data.append(dcm_img.SliceThickness)
          areamultiplier = float(dcm_img.PixelSpacing[0])*float(dcm_img.PixelSpacing[1])
          row_data.append(np.round(areamultiplier,2))

          # actual image data as an unwrapped 64x64 array (0-255 grayscale)
          row_data.extend(img)

          # write out the labels
          row_data.append(label_dict[userId][0])
          row_data.append(label_dict[userId][1])

          foutput.write(','.join(str(i) for i in row_data))

   return row_data


def save_img(img, dcm_path):
    jpg_path = dcm_path.rsplit(".", 1)[0] + ".64x64.jpg"     
    misc.imsave(jpg_path, img)
    print ("Saved {0}".format(jpg_path))


def crop(img):
   short_egde = min(img.shape[:2])
   yy = int((img.shape[0] - short_egde) / 2)
   xx = int((img.shape[1] - short_egde) / 2)
   crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
   return crop_img


def rescale(img, dcm_img):
   xscale = dcm_img.PixelSpacing[0]   
   yscale = dcm_img.PixelSpacing[1]   

   rescaled_img = transform.rescale(img, (xscale, yscale) )
   return rescaled_img


def preprocess(dcm_img):
   '''Take the DICOM file and convert it to a numpy array'''

   # generate a numpy array with range 0-1 pixel intensities
   img = dcm_img.pixel_array.astype(float) / np.max(dcm_img.pixel_array)

   # some images have height < width. 
   #in this case, rotate counter-clockwise so all images have same orientation
   if img.shape[0] < img.shape[1]:
     img = np.rot90(img)

   # crop a square image from the center
   img = crop(img)

   # scale the image by the pixel spacing given in the DICOM file
   img = rescale(img, dcm_img)

   # resize the image based on given number of pixels
   img = transform.resize(img, (NUM_PIXELS, NUM_PIXELS))

   # optionally denoise the image
   #img = denoise_tv_chambolle(img, weight=0.05, multichannel=False)

   # enhance the contrast of the image to make regions more distinct
   selem = disk(30)
   img = rank.equalize(img, selem=selem)
   img = img_as_ubyte(img) # converts 0-1 range to 0-255 range
   img = exposure.rescale_intensity(img)

   return img


def get_labels(file):
    label_dict = {}
    with open(file,"r") as f:
        next(f)
        for line in f: 
            tokens = line.split(",")
            userid = tokens[0]
            systole = tokens[1]
            diastole = tokens[2]
            label_dict[userid] = (systole, diastole)
    return label_dict
            

if __name__ == "__main__":

    label_dict = get_labels("/data/data/train.csv")

    folderpath = "/data/data/train"

    NUM_USERS = 50
    NUM_PIXELS = 64

    # e.g. look for the train folder at /data/data/train
    # file_structure is a JSON-like object with user/slice/path_to_DICOM_file hierarchy
    file_structure = get_file_structure(folderpath)

    # write out the csv to foutpath
    foutpath = "{0}/train_input.csv".format(folderpath)
    with open(foutpath,"w") as foutput:
      # crawl through the file_structure and process the DICOM files
      generate_csv(file_structure)

