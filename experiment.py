"""Run beside data/ containing the three named CSVs extracted from the supplied RAR."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binom, beta
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

specs = [('APS','aps_failure_reduced_30k','class'),('AI4I','ai4i2020_preprocessed','Machine failure'),('Conveyor','pre_conveyor_fault','Fault')]
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
output = Path("runs") / str(args.seed)
output.mkdir(parents=True, exist_ok=True)
rows, predictions = [], []
fig, axes = plt.subplots(1,3,figsize=(11,3.3),sharey=True)
for ax,(name,file,target) in zip(axes,specs):
    frame = pd.read_csv(Path('data')/(file+'.csv'))
    y = frame.pop(target).to_numpy(dtype=int)
    X = frame.drop(columns=['TWF','HDF','PWF','OSF','RNF'],errors='ignore')
    train, hold = train_test_split(np.arange(len(y)),test_size=.4,stratify=y,random_state=args.seed)
    cal,test = train_test_split(hold,test_size=.5,stratify=y[hold],random_state=args.seed+1)
    models = {'LR':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=3000)),
              'RF':make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=200,min_samples_leaf=2,random_state=42,n_jobs=4))}
    for model,est in models.items():
        est.fit(X.iloc[train],y[train])
        sc,st = [est.predict_proba(X.iloc[idx])[:,1] for idx in (cal,test)]
        for split,idx,score in [('cal',cal,sc),('test',test,st)]:
            predictions.append(pd.DataFrame(dict(dataset=name,model=model,split=split,row=idx,label=y[idx],score=score)))
        faults = np.sort(sc[y[cal]==1])
        precision,recall,thr = precision_recall_curve(y[cal],sc)
        f2 = 5*precision[:-1]*recall[:-1]/np.maximum(4*precision[:-1]+recall[:-1],1e-15)
        choices = {'Default':.5,'F2':thr[np.flatnonzero(f2==max(f2))[-1]]}
        for r in [.8,.9]:
            ranks=np.arange(1,len(faults)+1)
            valid=ranks[binom.sf(ranks-1,len(faults),1-r)>=.95]
            choices['R'+str(round(100*r))]=faults[valid[-1]-1] if len(valid) else -np.inf
        order=np.argsort(-st,kind='stable')
        budget=int(np.floor(.05*len(test)))
        for rule,t in list(choices.items())+[('Top5%',np.nan)]:
            flag=st>=t
            if rule=='Top5%':flag[order[:budget]]=True
            n=len(test); pos=sum(y[test]); tp=sum(y[test][flag]); alerts=sum(flag)
            rows.append(dict(dataset=name,model=model,rule=rule,threshold=t,test_n=n,positives=pos,cal_faults=len(faults),alerts=alerts,tp=tp,fp=alerts-tp,fn=pos-tp,per1000=1000*alerts/n,recall=tp/pos,lo=beta.ppf(.025,tp,pos-tp+1) if tp else 0,hi=beta.ppf(.975,tp+1,pos-tp) if tp<pos else 1))
        ax.step(100*np.arange(1,len(test)+1)/len(test),np.cumsum(y[test][order])/sum(y[test]),label=model,where='post')
    ax.axvline(5,ls=':',color='black');ax.axhline(.8,ls='--',color='gray');ax.set(xlabel='Records inspected (%)',title=name,xlim=(0,50),ylim=(0,1.04))
axes[0].set_ylabel('Fraction of faults found');axes[0].legend();fig.tight_layout();fig.savefig(output/'workload.png',dpi=300)
pd.concat(predictions).to_csv(output/'scores.csv.gz',index=False)
pd.DataFrame(rows).to_csv(output/'results.csv',index=False)
