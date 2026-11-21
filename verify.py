from pathlib import Path
import numpy as np
import pandas as pd
for seed in range(42,52):
    root=Path('runs')/str(seed)
    scores=pd.read_csv(root/'scores.csv.gz',float_precision='round_trip')
    rules=pd.read_csv(root/'results.csv',float_precision='round_trip')
    for record in rules.itertuples():
        subset=scores[(scores.dataset==record.dataset)&(scores.model==record.model)]
        test=subset[subset.split=='test']
        assert len(set(test.row)&set(subset[subset.split=='cal'].row))==0
        if record.rule=='Top5%':
            selected=test.sort_values('score',ascending=False,kind='stable').head(len(test)//20)
        else:
            selected=test[test.score>=record.threshold]
        assert (len(selected),sum(selected.label))==(record.alerts,record.tp)
print('Verified 300 policy evaluations against saved scores.')
