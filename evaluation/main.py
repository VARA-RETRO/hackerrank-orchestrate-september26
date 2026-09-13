#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

EXPECTED=['request_id','amount_safe_to_pay','affordability_status','recommended_payment_method','payment_plan','earliest_date_for_full_payment','spending_changes_needed','decision_explanation']

def main():
    root=Path(__file__).resolve().parents[1]
    out=root/'output.csv'; req=root/'dataset'/'requests.csv'
    df=pd.read_csv(out); r=pd.read_csv(req)
    checks={
        'rows_match':len(df)==len(r),
        'columns_exact':list(df.columns)==EXPECTED,
        'request_ids_match':set(df.request_id)==set(r.request_id),
        'amount_bounds':all((df.amount_safe_to_pay>=0) & (df.amount_safe_to_pay<=r.set_index('request_id').loc[df.request_id,'requested_amount'].to_numpy()+1e-9)),
        'valid_status':df.affordability_status.isin(['affordable_now','affordable_with_plan','affordable_later','not_affordable']).all(),
        'valid_method':df.recommended_payment_method.isin(['full_payment','partial_payment','installments','wait','not_recommended']).all(),
    }
    print(pd.Series(checks))
    raise SystemExit(0 if all(checks.values()) else 1)

if __name__=='__main__': main()
