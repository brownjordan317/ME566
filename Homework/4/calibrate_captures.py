"""Calibrate camera intrinsics from the HW4 9x6 inner-corner captures."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();obj=np.zeros((54,3),np.float32)
    obj[:,:2]=np.mgrid[0:9,0:6].T.reshape(-1,2)
    objects=[];corners=[];accepted=[];rejected=[];size=None
    for path in sorted(a.input.glob('*.jpg')):
        image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if image is None:rejected.append(str(path));continue
        if size is None:size=image.shape[::-1]
        if image.shape[::-1]!=size:raise ValueError('Mixed calibration resolutions')
        ok,points=cv2.findChessboardCornersSB(image,(9,6),cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:objects.append(obj.copy());corners.append(points);accepted.append(str(path.resolve()))
        else:rejected.append(str(path.resolve()))
    if len(objects)<10:raise ValueError('Need at least 10 detected views')
    rms,k,d,rs,ts=cv2.calibrateCamera(objects,corners,size,None,None)
    errors=[float(np.sqrt(np.mean(np.sum((cv2.projectPoints(obj,r,t,k,d)[0].reshape(-1,2)-q.reshape(-1,2))**2,axis=1)))) for r,t,q in zip(rs,ts,corners)]
    report={'camera_matrix':k.tolist(),'dist_coeffs':d.tolist(),'image_size_wh':list(size),
            'rms_error_px':rms,'inner_corners_wh':[9,6],
            'checkerboard_square_units':'unit squares; physical size unknown; only intrinsics used',
            'accepted_images':accepted,'rejected_images':rejected,'per_image_rmse_px':errors}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(f'{len(accepted)} accepted; {len(rejected)} without detected grid; RMS={rms:.6f} px')
if __name__=='__main__':main()
