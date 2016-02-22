import os
import csv
import sys
import random
import numpy as np
import dicom
from skimage import transform, img_as_ubyte, feature, exposure, segmentation, color
from skimage.future import graph
from skimage.morphology import disk
from skimage.filters import rank
from skimage.filters.rank import enhance_contrast
from skimage.restoration import denoise_tv_chambolle
from joblib import Parallel, delayed
import dill
import pickle
import boto
from scipy import misc
import math
import json


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def mkdir(fname):
   try:
       os.mkdir(fname)
   except:
       pass


def get_file_structure(root_path):
   """Return a JSON with user/slice/filepath hierarchy"""

   output = []

   currentId = 0
   for root, dirnames, files in os.walk(root_path):
     currentdir = root.replace("\\","/").rsplit("/", 1)[1]

     if is_number(currentdir):     
       currentId = int(float(currentdir))
       print("Walking user {0}".format(currentId))
       userdict = {}
       userdict["UserId"] = str(currentId)
       userdict["Slices"] = []
       output.append(userdict)

     if ("sax" in currentdir):
     #if ("sax" in currentdir and len(userdict["Slices"]) < 10):
       slicedict = {}
       slicedict["SliceId"] = currentdir.split("_")[1]
       slicedict["Files"] = sorted([x for x in files if ".dcm" in x], key = lambda f: int(float(f.split("-")[2].split(".")[0])))
       userdict["Slices"].append(slicedict)     

   output = sorted(output, key = lambda user: int(float(user["UserId"])))

   for userdict in output:
     userdict["Slices"] = sorted(userdict["Slices"], key = lambda slice: int(float(slice["SliceId"])))    

     #while (len(userdict["Slices"]) < 10):
     #  userdict["Slices"].append(userdict["Slices"][-1])

     #for slicedict in userdict["Slices"]:
     #  while (len(slicedict["Files"]) < 30):
     #      slicedict["Files"].append(slicedict["Files"][-1])

   return output  


def process_sequences(userdict_array, preproc):

   image_info = Parallel()(delayed(get_user_data)(user_dict,preproc,idx) for idx, user_dict in enumerate(userdict_array))

   print("All finished, %d images in total" % len(image_info))


def get_user_data(user_dict, preproc, idx):
   print ("UserId: {0}".format(idx))

   userId = user_dict["UserId"]

   user_output = []

   for slice_dict in user_dict["Slices"]:

       sliceId = slice_dict["SliceId"]

       for file in slice_dict["Files"]:
          dcm_path = "/data/data/{0}/{1}/study/sax_{2}/{3}".format(file_ind,userId,slice_dict["SliceId"],file)

          timeId = file.split("-")[2].split(".")[0]

          dcm_img = dicom.read_file(dcm_path)
          img = preproc(dcm_img)

          img = img.reshape(img.size)

          # 050Y = 50 years old. 048M = 48 months old
          age = int(dcm_img.PatientAge[:3])
          age_unit = dcm_img.PatientAge[-1]
          
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

          # actual image data as an unwrapped 64x64 array (0-255 grayscale)
          row_data.extend(img)

          foutput.write(','.join(str(i) for i in row_data)+'\n')

   return row_data


def get_circle_params(slice_dict, userId):
    key = str(userId) + "-" + str(slice_dict["SliceId"])
    print (key)
    if key in centers:
       circle_params = centers[key]
       print("found: {0}".format(circle_params[0]))
    else:
       circle_params = avg_params
       print("not found: {0}".format(circle_params[0]))


def get_img_from_dcm(dcm_img):
  img = dcm_img.pixel_array.astype(float) / np.max(dcm_img.pixel_array)
  return img


def rotate90counter(img):
  if img.shape[0] < img.shape[1]:
       img = np.rot90(img)
  return img


def crop(img):
   short_egde = min(img.shape[:2])
   yy = int((img.shape[0] - short_egde) / 2)
   xx = int((img.shape[1] - short_egde) / 2)
   crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
   return crop_img


def crop2(img, center_params):
   radius_scaled = 1.5*center_params[2]
   ylow = max(0, center_params[1]-radius_scaled)
   yhigh = min(img.shape[0]-1, center_params[1]+radius_scaled)
   xlow = max(0, center_params[0]-radius_scaled)
   xhigh = min(img.shape[1]-1, center_params[0]+radius_scaled)
   crop_img = img[ylow: yhigh , xlow : xhigh]
   return crop_img


def rescale(img, dcm_img):
   xscale = 1/dcm_img.PixelSpacing[0]   
   yscale = 1/dcm_img.PixelSpacing[1]   

   rescaled_img = transform.rescale(img, (xscale, yscale) )
   return rescaled_img


def resize(img, size):
   resized_img = transform.resize(img, (size, size))
   return resized_img


def convertdtype(img):
  return img_as_ubyte(img)


def cannyedge(img, s):
  return feature.canny(img, sigma=s)


def preprocess_func(dcm_img):

   img = get_img_from_dcm(dcm_img)

   img = crop(img)

   img = rescale(img, dcm_img)
   img = resize(img, 64)

   img = denoise_tv_chambolle(img, weight=0.05, multichannel=False)
   selem = disk(30)
   img = rank.equalize(img, selem=selem)
   img = exposure.rescale_intensity(convertdtype(img))

   img = rotate90counter(img)

   return img


def get_inputs():
    mode_param = sys.argv[1]

    if (mode_param != 'train' and mode_param != 'validate'):
      print("Invalid mode")
      exit()
 
    train = mode_param == 'train'

    if (train):
      file_ind = 'train'
    else:
      file_ind = 'validate'

    return file_ind


def get_centers(file):
    centers = {}
    x = []
    y = []
    r = []
    with open(file) as f:
      for line in f:
        items = line.split(",")
        key = items[0] + "-" + items[1].strip()
        val = (int(float(items[2])), int(float(items[3])), int(float(items[4])))
        x.append(int(float(items[2])))
        y.append(int(float(items[3])))
        r.append(int(float(items[4])))
        centers[key] = val

    avg_params = (int(np.array(x).mean()), int(np.array(y).mean()), int(np.array(r).mean()))

    return centers, avg_params


if __name__ == "__main__":

    random.seed(10)

    file_ind = get_inputs()

    file_structure = get_file_structure("/data/data/{0}".format(file_ind))

    with open("/data/staging/test.csv","w") as foutput:
      process_sequences(file_structure, preprocess_func)

