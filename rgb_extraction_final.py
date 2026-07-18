# rgb_extraction_final.py
import cv2
import numpy as np
import pandas as pd
import glob, os
from pathlib import Path

IMAGE_FOLDER='arsenic_samples'
OUTPUT_CSV="rgb_results.csv"
IMAGE_EXTENSIONS=("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff")
FIXED_ROI=None

def get_images(folder):
    files=[]
    for e in IMAGE_EXTENSIONS:
        files+=glob.glob(os.path.join(folder,e))
        files+=glob.glob(os.path.join(folder,e.upper()))
    return sorted(files)

def select_roi(img_path):
    img=cv2.imread(img_path)
    scale=1.0
    disp=img
    if img.shape[1]>1000:
        scale=1000/img.shape[1]
        disp=cv2.resize(img,None,fx=scale,fy=scale)
    roi=cv2.selectROI("Select ROI",disp,False)
    cv2.destroyAllWindows()
    x,y,w,h=[int(v/scale) for v in roi]
    return (x,y,w,h)

def features(img_path,roi):
    img=cv2.imread(img_path)
    x,y,w,h=roi
    crop=img[y:y+h,x:x+w]
    if crop.size==0:
        raise ValueError("Invalid ROI")
    rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    R,G,B=[rgb[:,:,i].astype(np.float32) for i in range(3)]
    H,S,V=[hsv[:,:,i].astype(np.float32) for i in range(3)]
    mR,mG,mB=R.mean(),G.mean(),B.mean()
    sR,sG,sB=R.std(),G.std(),B.std()
    total=mR+mG+mB
    if total==0: total=1
    rn,gn,bn=mR/total,mG/total,mB/total
    intensity=total/3
    flag=[]
    for n,v in [("R",mR),("G",mG),("B",mB)]:
        if v>=250: flag.append(f"{n}_HIGH")
        if v<=5: flag.append(f"{n}_LOW")
    return {
        "Filename":Path(img_path).name,
        "Mean_R":round(mR,2),"Mean_G":round(mG,2),"Mean_B":round(mB,2),
        "Std_R":round(sR,2),"Std_G":round(sG,2),"Std_B":round(sB,2),
        "Hue":round(H.mean(),2),"Saturation":round(S.mean(),2),"Value":round(V.mean(),2),
        "R_norm":round(rn,4),"G_norm":round(gn,4),"B_norm":round(bn,4),
        "Intensity":round(intensity,2),
        "Pixel_Count":crop.shape[0]*crop.shape[1],
        "Saturation_Flag":";".join(flag)
    }

def main():
    imgs=get_images(IMAGE_FOLDER)
    if not imgs:
        print("No images found.")
        return
    roi=FIXED_ROI if FIXED_ROI else select_roi(imgs[0])
    rows=[]
    for i,p in enumerate(imgs,1):
        try:
            d=features(p,roi)
            rows.append(d)
            print(f"[{i}/{len(imgs)}] {d['Filename']} -> R={d['Mean_R']} G={d['Mean_G']} B={d['Mean_B']}")
        except Exception as e:
            print("Error:",p,e)
    pd.DataFrame(rows).to_csv(OUTPUT_CSV,index=False)
    print("Saved",OUTPUT_CSV)
    print("ROI:",roi)

if __name__=="__main__":
    main()
