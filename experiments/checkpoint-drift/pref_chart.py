import json
P=json.load(open("/tmp/pref.json")); H=json.load(open("/tmp/heldout.json"))
ST=["base","ft1","ft2","ft3"]; LBL={"base":"base","ft1":"ft1","ft2":"ft2","ft3":"ft3"}
SUB={"base":"0 edits","ft1":"SFT · 4","ft2":"SFT · 10","ft3":"DPO"}
W,HH=470,352; L,R,T,B=76,20,50,78
YLO,YHI=-215.0,6.0
def X(i): return L+(W-L-R)*(i/(len(ST)-1))
def Y(v): return T+(HH-T-B)*(1-(v-YLO)/(YHI-YLO))

def panel(data, title, note):
    ch=[data[s]["chosen"] for s in ST]; rj=[data[s]["rejected"] for s in ST]
    segs=[]
    for i in range(len(ST)-1):
        d0,d1=ch[i]-rj[i], ch[i+1]-rj[i+1]
        pts=[(X(i),ch[i],rj[i])]
        if (d0<0)!=(d1<0):
            t=d0/(d0-d1); xc=X(i)+(X(i+1)-X(i))*t
            yc=ch[i]+(ch[i+1]-ch[i])*t
            pts.append((xc,yc,yc))
        pts.append((X(i+1),ch[i+1],rj[i+1]))
        for a,b in zip(pts,pts[1:]):
            up=((a[1]+b[1])/2)>((a[2]+b[2])/2)
            poly=f"{a[0]:.1f},{Y(a[1]):.1f} {b[0]:.1f},{Y(b[1]):.1f} {b[0]:.1f},{Y(b[2]):.1f} {a[0]:.1f},{Y(a[2]):.1f}"
            segs.append(f'<polygon points="{poly}" class="{"gap-up" if up else "gap-dn"}"/>')
    grid="".join(f'<line class="grid" x1="{L}" y1="{Y(v):.1f}" x2="{W-R}" y2="{Y(v):.1f}"/>'
                 f'<text class="ytick" x="{L-8}" y="{Y(v)+3.5:.1f}">{v}</text>'
                 for v in (0,-50,-100,-150,-200))
    xs="".join(f'<text class="xtick" x="{X(i):.1f}" y="{HH-B+20}">{LBL[s]}</text>'
               f'<text class="xsub" x="{X(i):.1f}" y="{HH-B+33}">{SUB[s]}</text>' for i,s in enumerate(ST))
    div=f'<line class="divider" x1="{(X(2)+X(3))/2:.1f}" y1="{T}" x2="{(X(2)+X(3))/2:.1f}" y2="{HH-B}"/>'
    def line(vals,cls):
        d=" ".join(f"{'M' if i==0 else 'L'}{X(i):.1f},{Y(v):.1f}" for i,v in enumerate(vals))
        dots="".join(f'<circle class="dot {cls}" cx="{X(i):.1f}" cy="{Y(v):.1f}" r="4.5">'
                     f'<title>{ST[i]}: {v:.1f}</title></circle>' for i,v in enumerate(vals))
        return f'<path class="ln {cls}" d="{d}"/>{dots}'
    acc="".join(f'<text class="acc" x="{X(i):.1f}" y="{T-9}">{data[s]["acc"]*100:.0f}%</text>'
                for i,s in enumerate(ST))
    acclbl=(f'<text class="acclbl" x="{L}" y="{T-24}">share of pairs where my edit wins</text>')
    cy=T+(HH-T-B)/2
    ytitle=(f'<text class="axtitle" transform="rotate(-90)" x="{-cy:.1f}" y="15" '
            f'text-anchor="middle">mean log P(clip) &mdash; higher = more likely</text>')
    xtitle=(f'<text class="axtitle" x="{(L+W-R)/2:.1f}" y="{HH-B+62}" '
            f'text-anchor="middle">fine-tuning stage &mdash; method &middot; edits used</text>')
    return f'''<figure class="pf">
  <figcaption><b>{title}</b><span>{note}</span></figcaption>
  <svg viewBox="0 0 {W} {HH}" role="img" aria-label="{title}">
    {grid}{"".join(segs)}{div}
    {line(rj,"s-orig")}{line(ch,"s-edit")}
    {acc}
    {xs}{acclbl}{ytitle}{xtitle}
  </svg></figure>'''

CSS="""
:root{--ground:#DDE3E1;--panel:#F6F8F7;--panel2:#EAEFED;--ink:#16201F;--muted:#63706E;
 --faint:#8A9694;--rule:#C3CCCA;--rule2:#D5DCDA;--edit:#26699F;--orig:#C2542B;
 --shadow:0 1px 2px rgba(20,32,31,.10),0 8px 26px -14px rgba(20,32,31,.30)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --ground:#101514;--panel:#1A2120;--panel2:#222A29;--ink:#E7ECEA;--muted:#94A29F;
 --faint:#77837F;--rule:#333D3C;--rule2:#2A3332;--edit:#4E93C8;--orig:#DA6733;
 --shadow:0 1px 2px rgba(0,0,0,.5),0 8px 26px -14px rgba(0,0,0,.8)}}
:root[data-theme="dark"]{--ground:#101514;--panel:#1A2120;--panel2:#222A29;--ink:#E7ECEA;
 --muted:#94A29F;--faint:#77837F;--rule:#333D3C;--rule2:#2A3332;--edit:#4E93C8;--orig:#DA6733;
 --shadow:0 1px 2px rgba(0,0,0,.5),0 8px 26px -14px rgba(0,0,0,.8)}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:Archivo,"Helvetica Neue",Arial,sans-serif;
 font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased;padding:clamp(20px,4vw,56px) clamp(14px,3vw,32px)}
.plate{max-width:1040px;margin:0 auto;display:flex;flex-direction:column;gap:22px}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px;letter-spacing:.16em;
 text-transform:uppercase;color:var(--muted);margin:0}
h1{font-size:clamp(26px,4.2vw,40px);line-height:1.05;letter-spacing:-.022em;font-weight:700;margin:8px 0 0;text-wrap:balance}
.lede{margin:10px 0 0;color:var(--muted);max-width:68ch;font-size:15.5px}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:3px;box-shadow:var(--shadow);
 padding:clamp(16px,2.4vw,24px)}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-family:"IBM Plex Mono",monospace;font-size:10.5px;
 letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-bottom:6px}
.lgi{display:flex;align-items:center;gap:7px}
.sw{width:16px;height:3px;border-radius:2px;display:block}
.sw.e{background:var(--edit)}.sw.o{background:var(--orig)}
.sw.g{height:11px;width:14px;border-radius:2px;opacity:.9}
.sw.gu{background:var(--edit);opacity:.16}.sw.gd{background:var(--orig);opacity:.16}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:20px}
.pf{margin:0}
figcaption{display:flex;flex-direction:column;gap:2px;margin-bottom:6px}
figcaption b{font-size:13.5px;font-weight:600}
figcaption span{font-size:12px;color:var(--muted)}
svg{width:100%;height:auto;display:block;overflow:visible}
.grid{stroke:var(--rule2);stroke-width:1}
.ytick{fill:var(--faint);font-size:9px;font-family:"IBM Plex Mono",monospace;text-anchor:end}
.xtick{fill:var(--muted);font-size:10.5px;font-family:"IBM Plex Mono",monospace;text-anchor:middle}
.xsub{fill:var(--faint);font-size:8.5px;font-family:"IBM Plex Mono",monospace;text-anchor:middle}
.divider{stroke:var(--rule);stroke-width:1}
.ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.ln.s-edit{stroke:var(--edit)}.ln.s-orig{stroke:var(--orig)}
.dot{stroke:var(--panel);stroke-width:2}
.dot.s-edit{fill:var(--edit)}.dot.s-orig{fill:var(--orig)}
.gap-up{fill:var(--edit);opacity:.16}.gap-dn{fill:var(--orig);opacity:.16}
.acc{fill:var(--faint);font-size:9px;font-family:"IBM Plex Mono",monospace;text-anchor:middle}
.acclbl{fill:var(--faint);font-size:8.5px;font-family:"IBM Plex Mono",monospace;text-anchor:start;
 letter-spacing:.08em;text-transform:uppercase}
.axtitle{fill:var(--muted);font-size:9.5px;font-family:"IBM Plex Mono",monospace;
 letter-spacing:.07em;text-transform:uppercase}
.axl{font-family:"IBM Plex Mono",monospace;font-size:10px;color:var(--faint);text-transform:uppercase;
 letter-spacing:.08em;margin:0 0 12px}
.cap{margin:0;font-size:12.5px;color:var(--muted);max-width:84ch}
.cap code{font-family:"IBM Plex Mono",monospace;font-size:11.5px;background:var(--panel2);padding:1px 5px;border-radius:2px}
.warn{border-left:3px solid var(--orig);padding-left:14px}
footer{display:block;font-size:11.5px;color:var(--muted);font-family:"IBM Plex Mono",monospace;
 border-top:1px solid var(--rule);padding-top:14px}
"""
HTML=f"""<title>Did SFT Learn My Taste</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>{CSS}</style>
<div class="plate">
  <header>
    <p class="eyebrow">Midi GPT &middot; bass &middot; preference</p>
    <h1>Did fine-tuning move toward my edits?</h1>
    <p class="lede">For each checkpoint, the log-probability it assigns to the clip I kept
      and to the one I replaced. When the blue line sits above the orange one, the model
      prefers my edit. Higher is more likely; both panels share one scale.</p>
  </header>
  <div class="card">
    <div class="legend">
      <span class="lgi"><span class="sw e"></span>My edit (chosen)</span>
      <span class="lgi"><span class="sw o"></span>Its original (rejected)</span>
      <span class="lgi"><span class="sw g gu"></span>prefers my edit</span>
      <span class="lgi"><span class="sw g gd"></span>prefers the original</span>
    </div>
    <p class="axl">each clip is 65 tokens &middot; log-probabilities are negative, so nearer the top means the model finds that clip more likely</p>
    <div class="panels">
      {panel(P["bass"],"The 10 edits it trained on","ft2 and ft3 were fit on exactly these")}
      {panel(H,"One edit it never saw","held out — not in any training run")}
    </div>
  </div>
  <p class="cap warn"><b>The two panels do not agree.</b> On its own training edits the model flips
    from preferring the original (23% of pairs) to preferring my edit (95% at ft2, 100% at ft3) — SFT
    moved the distribution a long way, and more edits moved it further. On the one edit held out, the
    orange line stays on top at every stage and preference accuracy is <b>0% throughout</b>. What the
    left panel measures is absorption of ten specific clips, not a learned taste that transfers.
    One held-out section is weak evidence — but it is the only held-out edit that exists.</p>
  <p class="cap"><b>How the margin grew matters too.</b> SFT (base &rarr; ft2) raised it by lifting my
    edit from &minus;132.9 to &minus;6.9 while leaving the original roughly where it was. DPO (ft2 &rarr; ft3)
    barely moved my edit (&minus;6.9 &rarr; &minus;11.2) and widened the gap mainly by pushing the original
    <em>down</em>, &minus;43.9 &rarr; &minus;82.1. That is the suppression case the DPO loss splits its
    margin in half to detect, visible here as the orange line falling rather than the blue one rising.</p>
  <footer>ft1 and ft2 both branch from base &middot; ft3 is DPO on ft2 &middot; 12 key-transpositions per section</footer>
</div>
"""
open("preference-shift.html","w").write(HTML)
head,rest=HTML.split("<style>",1); css,body=rest.split("</style>",1)
for t in ("light","dark"):
    open(f"preference-shift.{t}.html","w").write(
      f'<!doctype html>\n<html lang="en" data-theme="{t}">\n<head>\n<meta charset="utf-8">\n'
      '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
      +head+"<style>"+css+"</style>\n</head>\n<body>\n"+body.strip()+"\n</body>\n</html>\n")
print("wrote preference-shift.html")
