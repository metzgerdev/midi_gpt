import json, numpy as np
M=json.load(open("/tmp/ckpt_measure.json")); S=json.load(open("/tmp/se.json"))
ROLES=[("bass","Bass"),("arp","Arp")]; STAGES=["ft1","ft2","ft3"]
def row(r,s): return next(x for x in M[r] if x["stage"]==s)
def mse(v): a=np.array(v); return float(a.mean()), float(a.std(ddof=1)/np.sqrt(len(a)))

lossd={r:[(row(r,s)["corpus_loss"]/row(r,"base")["corpus_loss"]-1)*100 for s in STAGES] for r,_ in ROLES}
kl   ={r:[row(r,s)["kl_from_base"] for s in STAGES] for r,_ in ROLES}
LMAX=max(max(abs(v) for v in x) for x in lossd.values()); KMAX=max(max(x) for x in kl.values())

def deltas(metric):
    d={}
    for r,_ in ROLES:
        bm,bs=mse(S[r]["base"][metric]); out=[]
        for s in STAGES:
            m,se=mse(S[r][s][metric]); dif=m-bm; sed=(se**2+bs**2)**.5
            out.append((dif,sed,abs(dif)>2*sed))
        d[r]=out
    return d
FIT=deltas("fit"); LOCK=deltas("lock")
DMAXV=max(max(abs(v)+e for v,e,_ in x) for d in (FIT,LOCK) for x in d.values())

def dbars(data,vmax):
    rows=[]
    for i,s in enumerate(STAGES):
        cells=[]
        for r,rl in ROLES:
            v,se,sig=data[r][i]
            w=abs(v)/vmax*50; left=50 if v>=0 else 50-w
            e0=(50+(v-se)/vmax*50); e1=(50+(v+se)/vmax*50)
            ns="" if sig else " ns"
            cells.append(
              f'<div class="bl" data-tip="{rl} · {s} · {v:+.3f} ± {se:.3f} SE'
              f'{" · resolved" if sig else " · within noise"}">'
              f'<span class="bn">{rl}</span><span class="bt">'
              f'<i class="bar{ns} s-{r}" style="left:{left:.2f}%;width:{max(w,0.4):.2f}%"></i>'
              f'<i class="err" style="left:{e0:.2f}%;width:{max(e1-e0,0.3):.2f}%"></i>'
              f'</span><span class="bv">{v:+.3f}{"" if sig else " ns"}</span></div>')
        rows.append(f'<div class="grp"><span class="gl">{s}</span><div class="gb">{"".join(cells)}</div></div>')
    return f'<div class="chart"><span class="zero"></span>{"".join(rows)}</div>'

def bars(data,vmax,fmt,signed,guard=None):
    rows=[]
    for i,s in enumerate(STAGES):
        cells=[]
        for r,rl in ROLES:
            v=data[r][i]
            if signed: w=abs(v)/vmax*50; left=50 if v>=0 else 50-w
            else:      w=v/vmax*100; left=0
            cells.append(f'<div class="bl" data-tip="{rl} · {s} · {fmt(v)}"><span class="bn">{rl}</span>'
              f'<span class="bt"><i class="bar s-{r}" style="left:{left:.2f}%;width:{max(w,0.4):.2f}%"></i></span>'
              f'<span class="bv">{fmt(v)}</span></div>')
        rows.append(f'<div class="grp"><span class="gl">{s}</span><div class="gb">{"".join(cells)}</div></div>')
    z='<span class="zero"></span>' if signed else ""
    g=(f'<span class="guard" style="left:calc(28px + 12px + 38px + 9px + '
       f'(100% - 28px - 12px - 38px - 9px - 62px - 9px)*{(50+guard/vmax*50)/100:.4f})">'
       f'<em>SFT guard +15%</em></span>') if guard else ""
    return f'<div class="chart">{z}{g}{"".join(rows)}</div>'

LAYERS=["tok_emb","pos_emb","grid_proj","chord_proj","cond_proj","trf_blocks.0","trf_blocks.1",
        "trf_blocks.2","final_norm","out_head"]
LN={"trf_blocks.0":"block 0","trf_blocks.1":"block 1","trf_blocks.2":"block 2"}
WMAX=max(row(r,s)["drift"][l] for r,_ in ROLES for s in STAGES for l in LAYERS)
heat=[]
for r,rl in ROLES:
    heat.append(f'<div class="hrole">{rl}</div><div class="hgrid">')
    heat.append('<div></div>'+"".join(f'<div class="hcol">{LN.get(l,l)}</div>' for l in LAYERS))
    for s in STAGES:
        heat.append(f'<div class="hrow">{s}</div>')
        for l in LAYERS:
            v=row(r,s)["drift"][l]
            heat.append(f'<div class="hc{" dead" if v==0 else ""}" style="--m:{v/WMAX:.4f}" '
                        f'data-tip="{rl} · {s} · {LN.get(l,l)} · {v:.4f}">{"0" if v==0 else ""}</div>')
    heat.append('</div>')

tr=[]
for r,rl in ROLES:
    bm_f,bs_f=mse(S[r]["base"]["fit"]); bm_l,bs_l=mse(S[r]["base"]["lock"])
    for s in ["base"]+STAGES:
        d=row(r,s); mf,sf=mse(S[r][s]["fit"]); ml,sl=mse(S[r][s]["lock"])
        vb="—" if s=="base" else f'{(d["corpus_loss"]/row(r,"base")["corpus_loss"]-1)*100:+.1f}%'
        tr.append(f"<tr><td>{rl}</td><td>{s}</td><td>{d['method']}</td><td>{d['n_data'] or '—'}</td>"
                  f"<td>{d['corpus_loss']:.4f}</td><td>{vb}</td><td>{d['kl_from_base']:.3f}</td>"
                  f"<td>{mf:.3f} ±{sf:.3f}</td><td>{ml:.3f} ±{sl:.3f}</td></tr>")

nsig=sum(1 for d in (FIT,LOCK) for r,_ in ROLES for v,e,sg in d[r] if sg)
json.dump(dict(LOSS=bars(lossd,LMAX*1.15,lambda v:f"{v:+.1f}%",True,guard=15.0),
               KL=bars(kl,KMAX*1.12,lambda v:f"{v:.3f}",False),
               FIT=dbars(FIT,DMAXV*1.15), LOCK=dbars(LOCK,DMAXV*1.15),
               HEAT="".join(heat), TABLE="".join(tr), WMAX=f"{WMAX:.3f}",
               ARP3=f"{lossd['arp'][2]:+.1f}%", BASS3=f"{lossd['bass'][2]:+.1f}%",
               ARPKL=f"{kl['arp'][2]:.2f}", BASSKL=f"{kl['bass'][2]:.2f}",
               NSIG=str(nsig)), open("/tmp/chart_ctx.json","w"))
print("resolved (>2 SE) of 12 quality deltas:", nsig)
for nm,d in (("fit",FIT),("lock",LOCK)):
    for r,_ in ROLES:
        print(f"  {nm:5}{r:5}", [f"{v:+.3f}±{e:.3f}{'*' if sg else ''}" for v,e,sg in d[r]])
