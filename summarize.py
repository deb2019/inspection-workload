"""Summarize 10 repeated holdouts and revise the seed-42 manuscript."""
from pathlib import Path
import numpy as np
import pandas as pd

runs=[]
for seed in range(42,52):
    frame=pd.read_csv(f'runs/{seed}/results.csv')
    assert len(frame)==30 and not frame.duplicated(['dataset','model','rule']).any()
    assert ((frame.tp+frame.fp==frame.alerts)&(frame.tp+frame.fn==frame.positives)).all()
    assert (frame.query("rule=='Top5%'").eval('alerts == test_n // 20')).all()
    runs.append(frame.assign(seed=seed))
all_rows=pd.concat(runs,ignore_index=True)
all_rows['both']=((all_rows.recall>=.8)&(all_rows.alerts<=all_rows.test_n//20)).astype(int)
all_rows.to_csv('split_results.csv',index=False)
summary=all_rows.groupby(['dataset','model','rule']).agg(n=('seed','size'),rec_med=('recall','median'),rec_min=('recall','min'),rec_max=('recall','max'),load_med=('per1000','median'),load_min=('per1000','min'),load_max=('per1000','max'),hit80=('recall',lambda x:(x>=.8).sum()),hit90=('recall',lambda x:(x>=.9).sum()),both=('both','sum')).reset_index()
summary.to_csv('stability_summary.csv',index=False)
