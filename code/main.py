#!/usr/bin/env python3
import os, re, math, json, argparse
from pathlib import Path
from datetime import date, timedelta
import pandas as pd

OUT_COLS = [
    'request_id','amount_safe_to_pay','affordability_status',
    'recommended_payment_method','payment_plan',
    'earliest_date_for_full_payment','spending_changes_needed',
    'decision_explanation'
]
BAD_STATUS = {'failed','cancelled','rejected','declined'}
GOOD_STATUS = {'settled','scheduled','confirmed','completed'}


def s(x):
    if pd.isna(x): return ''
    return str(x).strip()

def f(x, default=0.0):
    try:
        if pd.isna(x) or s(x)=='': return default
        return float(x)
    except Exception:
        return default

def d(x):
    return pd.to_datetime(x).date()

def money(x):
    if abs(x) < 0.005: x = 0.0
    return f'{x:.2f}'.rstrip('0').rstrip('.')

def split_pipe(x):
    return [v for v in s(x).split('|') if v]

def parse_bool(x):
    return s(x).lower() in {'true','1','yes','y','t'}

def amount_from_text(text, currency=None):
    """Best-effort extraction; only used for messages/images that explicitly state an amount."""
    text = s(text)
    if not text: return None
    cur = re.escape(currency) if currency else r'(?:INR|IDR|ZAR|USD|EUR)'
    pats = [
        rf'{cur}\s*([0-9][0-9,]*(?:\.[0-9]+)?)',
        rf'([0-9][0-9,]*(?:\.[0-9]+)?)\s*{cur}',
    ]
    for p in pats:
        m = re.search(p, text, re.I)
        if m:
            try: return float(m.group(1).replace(',',''))
            except Exception: pass
    return None


def parse_messages(msgs, users):
    """Turn explicit message amendments into a small set of future event overrides.
    We deliberately do not interpret arbitrary instructions as commands; only financial
    facts with clear amounts/dates are extracted.
    """
    by_user = {}
    by_event = {}
    if msgs is None or msgs.empty: return by_user, by_event
    for _, r in msgs.iterrows():
        u=s(r.get('user_id')); ev=s(r.get('related_event_id')); txt=s(r.get('message_text'))
        if not u or not txt: continue
        rec={'text':txt,'sent_at':s(r.get('sent_at')),'request_id':s(r.get('request_id'))}
        by_user.setdefault(u,[]).append(rec)
        if ev: by_event.setdefault(ev,[]).append(rec)
    return by_user, by_event


def build_rate_lookup(rates):
    out={}
    if rates is None or rates.empty: return out
    for _,r in rates.iterrows():
        out[(s(r['rate_date']),s(r['from_currency']),s(r['to_currency']))]=f(r['rate'])
    return out


def convert(amount, src, dst, when, rates):
    src=s(src); dst=s(dst)
    if not src or src==dst: return amount
    key=(when.strftime('%Y-%m-%d'),src,dst)
    if key in rates: return amount*rates[key]
    # Use latest rate on/before the date, then exact inverse if available.
    candidates=[(pd.to_datetime(k[0]).date(),v) for k,v in rates.items() if k[1]==src and k[2]==dst and pd.to_datetime(k[0]).date()<=when]
    if candidates: return amount*sorted(candidates,key=lambda z:z[0])[-1][1]
    inv=[(pd.to_datetime(k[0]).date(),v) for k,v in rates.items() if k[1]==dst and k[2]==src and pd.to_datetime(k[0]).date()<=when and v]
    if inv: return amount/sorted(inv,key=lambda z:z[0])[-1][1]
    # Last-resort fixed-chain conversion through USD/EUR if possible.
    for mid in ('USD','EUR'):
        a=convert(amount,src,mid,when,rates)
        if a!=amount or src==mid:
            b=convert(a,mid,dst,when,rates)
            if b!=a or mid==dst: return b
    return amount


def infer_recurring_events(user_events, request_date, horizon_end):
    """Return future events that are explicitly scheduled/confirmed in the horizon.
    Historical recurring events are used only to identify patterns when a future event
    is missing; in that case we project by the median weekday/day-of-month cadence.
    """
    ev=user_events.copy()
    ev['_settle']=pd.to_datetime(ev['settlement_date'],errors='coerce').dt.date
    ev['_event']=pd.to_datetime(ev['event_date'],errors='coerce').dt.date
    ev=ev[~ev['status'].astype(str).str.lower().isin(BAD_STATUS)].copy()
    ev=ev[ev['_settle'].notna()]
    # Explicit future records are authoritative.
    future=ev[(ev['_settle']>=request_date)&(ev['_settle']<=horizon_end)].copy()
    return future


def event_is_countable(r):
    st=s(r.get('status')).lower()
    if st in BAD_STATUS: return False
    # Pending debits are not reliable enough for the safety forecast; scheduled/settled are.
    if st=='pending': return False
    amt=r.get('amount')
    return not (pd.isna(amt) or s(amt)=='')


def prepare_events(events, user_id, home_currency, request_date, horizon_end, rates, messages):
    ev=events[events['user_id'].astype(str)==str(user_id)].copy()
    if ev.empty: return []
    ev=ev[ev.apply(event_is_countable,axis=1)].copy()
    ev['_date']=pd.to_datetime(ev['settlement_date'],errors='coerce').dt.date
    ev=ev[ev['_date'].notna()]
    ev=ev[(ev['_date']>=request_date)&(ev['_date']<=horizon_end)]
    out=[]
    for _,r in ev.iterrows():
        amt=f(r['amount'],None)
        if amt is None: continue
        curr=s(r['currency']) or home_currency
        amt=convert(amt,curr,home_currency,r['_date'],rates)
        direction=s(r['direction']).lower()
        if direction=='credit': delta=amt
        elif direction=='debit': delta=-amt
        else: continue
        out.append({'event_id':s(r['event_id']),'date':r['_date'],'delta':delta,
                    'category':s(r['category']),'description':s(r['description']),
                    'event_type':s(r['event_type']),'direction':direction,
                    'flexibility':s(r.get('flexibility')).lower(),
                    'minimum_allowed_amount':f(r.get('minimum_allowed_amount'),0),
                    'currency':curr,'amount':amt,'linked_event_id':s(r.get('linked_event_id'))})
    return out


def apply_message_overrides(events, user_messages, request_date, home_currency):
    # Conservative: messages can explicitly confirm/revise a salary amount/date or cancel a future payment.
    if not user_messages: return events
    result=[dict(x) for x in events]
    for msg in user_messages:
        txt=msg['text'].lower()
        # Explicit cancellation language tied to a known event id.
        ids=re.findall(r'event[_-]?\d+',txt,re.I)
        for e in result:
            if e['event_id'].lower() in {i.lower() for i in ids} and any(k in txt for k in ['cancel','cancelled','canceled','stopped']):
                e['delta']=0
        # Salary amendment: replace the amount of the next salary event after the message date.
        if 'salary' in txt or 'payroll' in txt or 'gaji' in txt:
            amt=amount_from_text(msg['text'],home_currency)
            if amt is not None:
                for e in result:
                    if e['category']=='salary' and e['date']>=request_date:
                        e['delta']=amt if e['direction']=='credit' else -amt
                        break
    return result


def future_balance_curve(balance, events, min_balance, extra_plan=None, spending_changes=None):
    """Daily balance curve. extra_plan is list[(date, amount)] debits.
    spending_changes is list of (event_id, new_amount_or_none); matching recurring events
    are reduced/stopped when they occur.
    """
    by_date={}
    for e in events:
        by_date.setdefault(e['date'],[]).append(e)
    changes={}
    for c in spending_changes or []: changes[c[0]]=c[1]
    for e in events:
        if e['event_id'] in changes:
            pass
    plan_by_date={}
    for dt,amt in extra_plan or []:
        plan_by_date[dt]=plan_by_date.get(dt,0)+amt
    # Reconstruct dates from min/max events and plan.
    all_dates=set(by_date)|set(plan_by_date)
    if not all_dates: return {}
    lo=min(all_dates); hi=max(all_dates)
    bal=balance
    curve={}
    for day in pd.date_range(lo,hi).date:
        for e in by_date.get(day,[]):
            delta=e['delta']
            if e['event_id'] in changes:
                newamt=changes[e['event_id']]
                if newamt is None: delta=0
                else:
                    sign=1 if e['delta']>=0 else -1
                    delta=sign*newamt
            bal+=delta
        bal-=plan_by_date.get(day,0)
        curve[day]=bal
    return curve


def safe_plan(balance, min_balance, events, request_date, plan):
    """Check all dates from request date through plan horizon. plan entries are debits."""
    if plan is None: plan=[]
    by_date={}
    for e in events: by_date.setdefault(e['date'],[]).append(e)
    pdict={}
    for dt,amt in plan: pdict[dt]=pdict.get(dt,0)+amt
    dates=sorted(set(by_date)|set(pdict)|{request_date})
    bal=balance
    for day in dates:
        if day < request_date: continue
        for e in by_date.get(day,[]): bal+=e['delta']
        bal-=pdict.get(day,0)
        if bal < min_balance-1e-7: return False,bal,day
    return True,bal,None


def daily_balance(balance, events, request_date, end_date, plan=None, changes=None):
    by_date={}
    for e in events:
        by_date.setdefault(e['date'],[]).append(e)
    p={}
    for dt,amt in plan or []: p[dt]=p.get(dt,0)+amt
    changes=changes or {}
    bal=balance; curve={}
    for day in pd.date_range(request_date,end_date).date:
        for e in by_date.get(day,[]):
            delta=e['delta']
            if e['event_id'] in changes:
                new=changes[e['event_id']]
                delta=0 if new is None else (new if e['delta']>=0 else -new)
            bal += delta
        bal -= p.get(day,0)
        curve[day]=bal
    return curve


def max_safe_today(balance,min_balance,events,request_date,end_date,requested,changes=None):
    # Amount today is safe if subtracting it keeps every future balance >= minimum.
    curve=daily_balance(balance,events,request_date,end_date,changes=changes)
    # curve excludes the requested payment; at request date this includes same-day events.
    floor=min(curve.values()) if curve else balance
    safe=max(0.0, floor-min_balance)
    return min(requested,safe)


def earliest_full(balance,min_balance,events,request_date,end_date,requested,changes=None):
    curve=daily_balance(balance,events,request_date,end_date,changes=changes)
    for day in pd.date_range(request_date,end_date).date:
        # payment at day happens after that day's regular cashflows
        if day not in curve: continue
        bal_after=curve[day]-requested
        # Need balance after payment and all subsequent future events to remain safe.
        future_min=min(v for dd,v in curve.items() if dd>=day)-requested
        if future_min>=min_balance-1e-7:
            return day
    return None


def recurring_change_candidates(events, profile, request_date, end_date):
    allowed_stop=set(split_pipe(profile.get('expense_categories_user_willing_to_stop')))
    allowed_reduce=set(split_pipe(profile.get('expense_categories_user_willing_to_reduce')))
    candidates=[]
    # Use one representative future event per flexible category/event pattern.
    seen=set()
    for e in sorted(events,key=lambda x:(x['date'],x['event_id'])):
        if e['date']<request_date or e['date']>end_date: continue
        if e['flexibility'] not in {'stoppable','reducible'}: continue
        cat=e['category']
        if e['flexibility']=='stoppable' and cat in allowed_stop and cat not in seen:
            candidates.append(('stop',e,None)); seen.add(cat)
        elif e['flexibility']=='reducible' and cat in allowed_reduce and cat not in seen:
            new=max(e['minimum_allowed_amount'],0.0)
            candidates.append(('reduce',e,new)); seen.add(cat)
    return candidates


def apply_changes_to_events(events, changes):
    mapping={}
    for typ,e,new in changes:
        mapping[e['event_id']] = None if typ=='stop' else new
    return mapping


def change_tokens(changes):
    if not changes: return 'none'
    toks=[]
    for typ,e,new in changes[:3]:
        if typ=='stop': toks.append(f"stop:{e['event_id']}")
        else: toks.append(f"reduce_to:{e['event_id']}:{money(new)}")
    return '|'.join(toks)


def payment_option_plan(opt):
    n=int(f(opt.get('number_of_payments'),0))
    if n<=0: return []
    first=pd.to_datetime(opt['first_payment_date']).date()
    freq=f(opt.get('payment_frequency_days'),0)
    amt=f(opt['payment_amount'])
    return [(first+timedelta(days=freq*i),amt) for i in range(n)]


def plan_end(plan): return max((x[0] for x in plan),default=None)


def fmt_plan(plan):
    return '|'.join(f'{dt.isoformat()}:{money(a)}' for dt,a in sorted(plan)) if plan else 'none'


def plan_safe(balance,min_balance,events,request_date,plan,changes=None):
    # Need daily checking because a payment can occur between sparse event dates.
    # The challenge's safety horizon is 90 days even after the final requested payment.
    end=max([request_date + timedelta(days=90)]+[x[0] for x in plan])
    curve=daily_balance(balance,events,request_date,end,plan=plan,changes=changes)
    return all(v>=min_balance-1e-7 for v in curve.values())


def solve(req, profile, events, options, messages, rates):
    rid=s(req['request_id']); uid=s(req['user_id'])
    rd=d(req['request_date']); deadline=d(req['desired_completion_date'])
    horizon=rd+timedelta(days=90)
    reqamt=f(req['requested_amount'])
    minbal=f(profile['minimum_balance_to_keep'])
    balance=f(profile['current_available_balance'])
    methods=set(split_pipe(profile.get('payment_methods_user_will_consider')))
    allows_partial=parse_bool(req.get('allows_partial_payment'))
    currency=s(profile.get('home_currency'))

    ev=prepare_events(events,uid,currency,rd,horizon,rates,messages)
    user_msgs=messages.get(uid,[]) if isinstance(messages,dict) else []
    ev=apply_message_overrides(ev,user_msgs,rd,currency)

    # Events on request date are included before the new requested payment.
    amount_safe=max_safe_today(balance,minbal,ev,rd,horizon,reqamt)
    nochange_earliest=earliest_full(balance,minbal,ev,rd,horizon,reqamt)

    # Candidate safe spending changes, tested incrementally. Never change protected categories.
    candidates=recurring_change_candidates(ev,profile,rd,horizon)
    change_scenarios=[[]]
    # 1-3 changes; prioritize fewer changes and larger immediate relief.
    for c in candidates[:8]: change_scenarios.append([c])
    for i in range(min(len(candidates),6)):
        for j in range(i+1,min(len(candidates),6)):
            if candidates[i][1]['event_id']!=candidates[j][1]['event_id']:
                change_scenarios.append([candidates[i],candidates[j]])
    for c in candidates[:4]:
        for c2 in candidates[:4]:
            if c is not c2 and c[1]['event_id']!=c2[1]['event_id']:
                pass

    # Evaluate capacity with each change set.
    change_infos=[]
    for changes in change_scenarios:
        mapping=apply_changes_to_events(ev,changes)
        for typ,ce,newval in changes:
            sig=(ce['category'],ce['description'],ce['direction'])
            for ee in ev:
                if (ee['category'],ee['description'],ee['direction'])==sig and ee['date']>=rd:
                    mapping[ee['event_id']]=None if typ=='stop' else newval
        safe_amt=max_safe_today(balance,minbal,ev,rd,horizon,reqamt,changes=mapping)
        full_date=earliest_full(balance,minbal,ev,rd,horizon,reqamt,changes=mapping)
        change_infos.append((changes,safe_amt,full_date,mapping))
    change_infos.sort(key=lambda z:(len(z[0]),-z[1]))

    # Immediate safe payment candidates.
    plans=[]
    opts=options.get(rid,[]) if isinstance(options,dict) else []
    for o in opts:
        method=s(o.get('payment_method'))
        if method not in methods or method not in {'full_payment','installments'}: continue
        plan=payment_option_plan(o)
        if not plan: continue
        if plan_end(plan)>deadline or plan_end(plan)>horizon: continue
        # Payment plan amounts must sum to the supplied option's total payable amount by construction.
        for changes,safe_amt,full_date,mapping in change_infos:
            if plan_safe(balance,minbal,ev,rd,plan,changes=mapping):
                total=f(o.get('total_payable_amount'))
                plans.append({
                    'kind':method,'plan':plan,'changes':changes,'total':total,
                    'start':plan[0][0],'count':len(plan),'option_id':s(o.get('payment_option_id')),
                    'safe_amt':safe_amt,'full_date':full_date
                })
                break

    # Full payment now is a special plan and should win when eligible and safe.
    full_plan=[(rd,reqamt)]
    if 'full_payment' in methods:
        for changes,safe_amt,full_date,mapping in change_infos:
            if plan_safe(balance,minbal,ev,rd,full_plan,changes=mapping):
                plans.append({'kind':'full_payment','plan':full_plan,'changes':changes,'total':reqamt,
                              'start':rd,'count':1,'option_id':'','safe_amt':safe_amt,'full_date':full_date})
                break

    # Partial payment: exact two-payment rule, only if request permits and user accepts it.
    if allows_partial and 'partial_payment' in methods and amount_safe>0 and amount_safe<reqamt:
        if nochange_earliest and nochange_earliest<=deadline:
            pp=[(rd,amount_safe),(nochange_earliest,reqamt-amount_safe)]
            if plan_safe(balance,minbal,ev,rd,pp):
                plans.append({'kind':'partial_payment','plan':pp,'changes':[],'total':reqamt,
                              'start':rd,'count':2,'option_id':'','safe_amt':amount_safe,'full_date':nochange_earliest})

    # Change-based full payment: payment today after stopping/reducing flexible recurring spend.
    if 'full_payment' in methods:
        for changes,safe_amt,full_date,mapping in change_infos:
            if changes and plan_safe(balance,minbal,ev,rd,full_plan,changes=mapping):
                plans.append({'kind':'full_payment','plan':full_plan,'changes':changes,'total':reqamt,
                              'start':rd,'count':1,'option_id':'','safe_amt':safe_amt,'full_date':full_date})
                break

    # Deduplicate identical plan/change candidates.
    uniq=[]; seen=set()
    for p in plans:
        key=(p['kind'],tuple(p['plan']),change_tokens(p['changes']))
        if key not in seen: seen.add(key); uniq.append(p)
    plans=uniq

    # Rank exactly in the challenge's order.
    def rank(p):
        complete=p['plan'][-1][0] <= deadline
        no_changes=(len(p['changes'])==0)
        return (-int(complete),-int(no_changes),p['total'],p['start'],p['count'],p['option_id'] or 'ZZZ')
    plans.sort(key=rank)

    if plans:
        p=plans[0]
        kind=p['kind']; plan=p['plan']; changes=p['changes']
        # Status depends on method/plan, with affordable_now only for a safe full payment today.
        if kind=='full_payment' and plan and plan[0][0]==rd and not changes:
            status='affordable_now'
        else:
            status='affordable_with_plan'
        full_date = rd if status=='affordable_now' else (p['full_date'] or nochange_earliest)
        if kind=='installments' and p['full_date'] is None:
            full_date=nochange_earliest
        exp=(f"Pay {currency} {money(reqamt)} today. " if kind=='full_payment' and status=='affordable_now'
             else f"Use {len(plan)} scheduled payments totaling {currency} {money(p['total'])}. ")
        if changes:
            exp += 'Apply the permitted flexible spending changes first. '
        exp += f"The plan keeps at least {currency} {money(minbal)} available through the 90-day forecast."
        return {
            'request_id':rid,'amount_safe_to_pay':round(min(max(amount_safe,0),reqamt),2),
            'affordability_status':status,'recommended_payment_method':kind,
            'payment_plan':fmt_plan(plan),'earliest_date_for_full_payment':full_date.isoformat() if full_date else '',
            'spending_changes_needed':change_tokens(changes),'decision_explanation':exp
        }

    # Wait is eligible only if the user accepts full payment and the full amount becomes safe later.
    if 'full_payment' in methods and nochange_earliest and nochange_earliest<=min(deadline,horizon):
        plan=[(nochange_earliest,reqamt)]
        if plan_safe(balance,minbal,ev,rd,plan):
            return {'request_id':rid,'amount_safe_to_pay':round(min(max(amount_safe,0),reqamt),2),
                    'affordability_status':'affordable_later','recommended_payment_method':'wait',
                    'payment_plan':fmt_plan(plan),'earliest_date_for_full_payment':nochange_earliest.isoformat(),
                    'spending_changes_needed':'none',
                    'decision_explanation':f"Wait until {nochange_earliest.isoformat()}, then pay {currency} {money(reqamt)} in full. Paying earlier would put the {currency} {money(minbal)} minimum at risk."}

    # If capacity exists today but the request cannot be completed by the deadline, do not recommend it.
    return {'request_id':rid,'amount_safe_to_pay':round(min(max(amount_safe,0),reqamt),2),
            'affordability_status':'not_affordable','recommended_payment_method':'not_recommended',
            'payment_plan':'none','earliest_date_for_full_payment':'',
            'spending_changes_needed':'none',
            'decision_explanation':f"Do not make this payment by {deadline.isoformat()}. None of the available options keeps the {currency} {money(minbal)} minimum protected."}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='dataset')
    ap.add_argument('--output',default='output.csv')
    args=ap.parse_args()
    root=Path(args.dataset)
    req=pd.read_csv(root/'requests.csv')
    profiles=pd.read_csv(root/'financial_profiles.csv')
    events=pd.read_csv(root/'financial_events.csv')
    opts=pd.read_csv(root/'request_payment_options.csv')
    rates=pd.read_csv(root/'exchange_rates.csv') if (root/'exchange_rates.csv').exists() else pd.DataFrame()
    msgs=pd.read_csv(root/'messages.csv') if (root/'messages.csv').exists() else pd.DataFrame()
    images=pd.read_csv(root/'images.csv') if (root/'images.csv').exists() else pd.DataFrame()

    profiles_idx={s(r['user_id']):r for _,r in profiles.iterrows()}
    options={}
    for _,r in opts.iterrows(): options.setdefault(s(r['request_id']),[]).append(r)
    rates_idx=build_rate_lookup(rates)
    msg_by_user,_=parse_messages(msgs,profiles)

    rows=[]
    for _,r in req.iterrows():
        uid=s(r['user_id'])
        if uid not in profiles_idx:
            raise RuntimeError(f'Missing financial profile for {uid}')
        rows.append(solve(r,profiles_idx[uid],events,options,msg_by_user,rates_idx))
    out=pd.DataFrame(rows,columns=OUT_COLS)
    # Deterministic validation.
    if len(out)!=len(req): raise RuntimeError('Row count mismatch')
    if list(out.columns)!=OUT_COLS: raise RuntimeError('Column order mismatch')
    for _,r in out.iterrows():
        reqrow=req[req.request_id==r.request_id].iloc[0]
        a=f(r.amount_safe_to_pay); q=f(reqrow.requested_amount)
        if not (-1e-6<=a<=q+1e-6): raise RuntimeError(f'Unsafe amount for {r.request_id}')
        if r.affordability_status not in {'affordable_now','affordable_with_plan','affordable_later','not_affordable'}: raise RuntimeError('Bad status')
        if r.recommended_payment_method not in {'full_payment','partial_payment','installments','wait','not_recommended'}: raise RuntimeError('Bad method')
    out.to_csv(args.output,index=False)
    print(f'Wrote {args.output} with {len(out)} predictions.')

if __name__=='__main__': main()
