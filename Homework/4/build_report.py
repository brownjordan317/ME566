"""Generate measured tables, figures, and conditional pose covariance for HW4."""
import argparse,importlib,json,platform,sys
from pathlib import Path
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'latex');a=p.parse_args()
 root=a.root.resolve();hw=root/'Homework/4';out=a.output;out.mkdir(parents=True,exist_ok=True);fig=out/'figures';fig.mkdir(exist_ok=True)
 sys.path.insert(0,str(root));D=importlib.import_module('Homework.4.aruco.detection').MarkerDetector
 shared=importlib.import_module('Homework.4.shared_orientation').fit_shared_rotation
 motion=json.loads((hw/'outputs/captures_3inch_shared/motion.json').read_text());cal=json.loads((hw/'camera_cal.json').read_text())
 k=np.array(cal['camera_matrix']);d=np.array(cal['dist_coeffs']);det=D(k,d,.10)
 paths=sorted((hw/'hw4_captures/aruco_caps').glob('*.jpg'));obs=[]
 for path in paths:
  _,_,poses=det.detect(cv2.imread(str(path)));obs.append(poses[0].corners)
 r,t,res=shared(det.object_points,obs,k,d);n=len(t);pos=t[0]-t
 np.testing.assert_allclose(pos,np.array([f['position_m'] for f in motion['frames']]),atol=1e-8)
 J=np.zeros((8*n,3+3*n))
 for i in range(n):
  _,jac=cv2.projectPoints(det.object_points,r,t[i],k,d);J[i*8:(i+1)*8,:3]=jac[:,:3];J[i*8:(i+1)*8,3+3*i:6+3*i]=jac[:,3:6]
 rank=int(np.linalg.matrix_rank(J));dof=8*n-rank;sigma2=float(np.sum(res**2)/dof);cov=sigma2*np.linalg.inv(J.T@J)
 pc=[];distance_sd=[];step_sd=[]
 for i in range(n):
  A=np.zeros((3,3+3*n));A[:,3:6]+=np.eye(3);A[:,3+3*i:6+3*i]-=np.eye(3);c=A@cov@A.T;pc.append(c)
  dist=np.linalg.norm(pos[i]);distance_sd.append(0. if dist==0 else float(np.sqrt(pos[i]@c@pos[i]/dist**2)))
  if i:
   B=np.zeros_like(A);B[:,3+3*(i-1):6+3*(i-1)]+=np.eye(3);B[:,3+3*i:6+3*i]-=np.eye(3);v=pos[i]-pos[i-1];step_sd.append(float(np.sqrt(v@(B@cov@B.T)@v/(v@v))))
 steps=np.linalg.norm(np.diff(pos,axis=0),axis=1);dist=np.linalg.norm(pos,axis=1);expected=np.arange(n)*.0762
 stats={'frames':n,'rank':rank,'dof':dof,'sigma_px':np.sqrt(sigma2),'fit_rmse_px':motion['fit_rmse_px'],'mean_step_cm':float(steps.mean()*100),'step_rmse_cm':float(np.sqrt(np.mean((steps-.0762)**2))*100),'endpoint_cm':float(dist[-1]*100),'expected_endpoint_cm':float(expected[-1]*100),'endpoint_error_cm':float((dist[-1]-expected[-1])*100),'cumulative_rmse_cm':float(np.sqrt(np.mean((dist-expected)**2))*100),'position_std_cm':(np.sqrt(np.maximum(np.diagonal(np.array(pc),axis1=1,axis2=2),0))*100).tolist(),'distance_std_cm':(np.array(distance_sd)*100).tolist(),'step_std_cm':(np.array(step_sd)*100).tolist(),'shared_rodrigues_std_rad':np.sqrt(np.diag(cov)[:3]).tolist(),'conditional_on':'fixed K/distortion, exact 0.10m marker, zero relative rotation, independent equal-variance pixel residuals','versions':{'python':platform.python_version(),'opencv':cv2.__version__,'numpy':np.__version__}}
 (out/'uncertainty.json').write_text(json.dumps(stats,indent=2)+'\n');np.savez(out/'pose_covariance.npz',parameter_covariance=cov,relative_position_covariance=np.array(pc),jacobian=J)
 lines=[]
 for i in range(n):
  step='--' if i==0 else f'{steps[i-1]*100:.2f}'
  lines.append(f'{i} & {paths[i].stem[-6:]} & {pos[i,0]*100:.2f} & {pos[i,1]*100:.2f} & {pos[i,2]*100:.2f} & {dist[i]*100:.2f} & {expected[i]*100:.2f} & {step} & {1.96*distance_sd[i]*100:.2f} '+r'\\')
 (out/'measurements.tex').write_text('\n'.join(lines)+'\n')
 macros={'Endpoint':stats['endpoint_cm'],'EndpointError':stats['endpoint_error_cm'],'CumulativeRMSE':stats['cumulative_rmse_cm'],'SigmaPixel':stats['sigma_px'],'EndpointCI':1.96*distance_sd[-1]*100,'SharedRxSD':np.rad2deg(np.sqrt(cov[0,0])),'SharedRySD':np.rad2deg(np.sqrt(cov[1,1])),'SharedRzSD':np.rad2deg(np.sqrt(cov[2,2]))}
 (out/'numbers.tex').write_text('\n'.join('\\newcommand{\\'+key+'}{'+f'{value:.3f}'+'}' for key,value in macros.items()))
 plt.rcParams.update({'font.size':10})
 f,axs=plt.subplots(2,1,figsize=(7,6),layout='constrained')
 axs[0].plot(range(n),expected*100,'k--',label='Expected, 7.62 cm per interval');axs[0].errorbar(range(n),dist*100,yerr=np.array(distance_sd)*196,fmt='o-',label='Estimated distance, conditional 95% interval');axs[0].set(ylabel='Distance from first frame (cm)',xlabel='Frame index');axs[0].legend(fontsize=8);axs[0].grid(alpha=.3)
 axs[1].errorbar(range(1,n),steps*100,yerr=np.array(step_sd)*196,fmt='o-',label='Estimated interval');axs[1].axhline(7.62,c='k',ls='--',label='Expected 7.62 cm');axs[1].set(xlabel='Destination frame index',ylabel='Consecutive translation (cm)');axs[1].legend(fontsize=8);axs[1].grid(alpha=.3);f.savefig(fig/'motion_comparison.png',dpi=200);plt.close(f)
 f,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
 for ax,i in zip(axs,[0,5,10]):
  im=cv2.imread(motion['frames'][i]['annotated_image']);ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));ax.set_title(f'Frame {i}: {dist[i]*100:.1f} cm');ax.axis('off')
 f.savefig(fig/'annotated_sequence.png',dpi=200);plt.close(f)
 f,axs=plt.subplots(1,2,figsize=(8,4),layout='constrained');im=cv2.imread(str(paths[0]));corners,ids,poses=det.detect(im);xy=poses[0].corners;lo=np.floor(xy.min(0)-100).astype(int);hi=np.ceil(xy.max(0)+100).astype(int);ann=cv2.imread(motion['frames'][0]['annotated_image']);axs[0].imshow(cv2.cvtColor(ann[max(0,lo[1]):hi[1],max(0,lo[0]):hi[0]],cv2.COLOR_BGR2RGB));axs[0].set_title('Marker outline, ID, and XYZ axes');axs[0].axis('off')
 cf=Path(cal['accepted_images'][0]);im=cv2.imread(str(cf));ok,cs=cv2.findChessboardCornersSB(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(9,6),cv2.CALIB_CB_NORMALIZE_IMAGE);cv2.drawChessboardCorners(im,(9,6),cs,ok);axs[1].imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));axs[1].set_title('9 × 6 calibration inner corners');axs[1].axis('off');f.savefig(fig/'marker_and_calibration.png',dpi=200);plt.close(f)
 print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
