import json
C = json.load(open("/tmp/chart_ctx.json"))
CSS = """
:root{
  --ground:#DDE3E1; --panel:#F6F8F7; --panel2:#EAEFED;
  --ink:#16201F; --muted:#63706E; --faint:#8A9694; --rule:#C3CCCA; --rule2:#D5DCDA;
  --s-bass:#26699F; --s-arp:#C2542B;
  --heat-lo:#EAF0ED; --heat-hi:#1F4A4E; --dead:#B24A2E;
  --warn:#C2542B;
  --shadow:0 1px 2px rgba(20,32,31,.10),0 8px 26px -14px rgba(20,32,31,.30);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#101514; --panel:#1A2120; --panel2:#222A29;
  --ink:#E7ECEA; --muted:#94A29F; --faint:#77837F; --rule:#333D3C; --rule2:#2A3332;
  --s-bass:#4E93C8; --s-arp:#DA6733;
  --heat-lo:#222A29; --heat-hi:#7FCBB4; --dead:#E2703F;
  --warn:#E2703F;
  --shadow:0 1px 2px rgba(0,0,0,.5),0 8px 26px -14px rgba(0,0,0,.8);
}}
:root[data-theme="dark"]{
  --ground:#101514; --panel:#1A2120; --panel2:#222A29;
  --ink:#E7ECEA; --muted:#94A29F; --faint:#77837F; --rule:#333D3C; --rule2:#2A3332;
  --s-bass:#4E93C8; --s-arp:#DA6733;
  --heat-lo:#222A29; --heat-hi:#7FCBB4; --dead:#E2703F;
  --warn:#E2703F;
  --shadow:0 1px 2px rgba(0,0,0,.5),0 8px 26px -14px rgba(0,0,0,.8);
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font-family:Archivo,"Helvetica Neue",Arial,sans-serif;font-size:15px;line-height:1.55;
  -webkit-font-smoothing:antialiased;padding:clamp(20px,4vw,56px) clamp(14px,3vw,32px)}
.plate{max-width:1140px;margin:0 auto;display:flex;flex-direction:column;gap:22px}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--muted);margin:0}
h1{font-size:clamp(27px,4.4vw,42px);line-height:1.05;letter-spacing:-.022em;font-weight:700;
  margin:8px 0 0;text-wrap:balance}
.lede{margin:10px 0 0;color:var(--muted);max-width:66ch;font-size:15.5px}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:3px;box-shadow:var(--shadow)}
.pad{padding:clamp(16px,2.4vw,24px)}
h2{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);
  font-family:"IBM Plex Mono",monospace;font-weight:500;margin:0 0 4px}
.sub{font-size:12.5px;color:var(--faint);margin:0 0 16px}
.mono{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}

.lineage{display:flex;flex-wrap:wrap;gap:9px;align-items:center;font-family:"IBM Plex Mono",monospace;font-size:11.5px}
.node{padding:5px 11px;border:1px solid var(--rule);border-radius:2px;background:var(--panel2)}
.node b{font-weight:600}.node em{font-style:normal;color:var(--muted)}
.node.dead{opacity:.6;border-style:dashed}
.arrow{color:var(--faint)}

.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:0;
  border:1px solid var(--rule);border-radius:3px;overflow:hidden;background:var(--panel)}
.tile{padding:15px 17px;border-right:1px solid var(--rule2);display:flex;flex-direction:column;gap:2px}
.tile:last-child{border-right:0}
.tn{font-size:25px;font-weight:700;letter-spacing:-.02em;line-height:1.1}
.tn.warn{color:var(--warn)}
.tl{font-family:"IBM Plex Mono",monospace;font-size:10px;letter-spacing:.12em;
  text-transform:uppercase;color:var(--muted)}
.td{font-size:12.5px;color:var(--muted);line-height:1.4}

.legend{display:flex;gap:15px;flex-wrap:wrap;font-family:"IBM Plex Mono",monospace;
  font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-bottom:14px}
.lgi{display:flex;align-items:center;gap:6px}
.sw{width:12px;height:12px;border-radius:2px;display:block}
.sw.s-bass{background:var(--s-bass)}.sw.s-arp{background:var(--s-arp)}

.chart{position:relative;display:flex;flex-direction:column;gap:13px;padding:6px 0 2px}
.grp{display:flex;gap:12px;align-items:center}
.gl{font-family:"IBM Plex Mono",monospace;font-size:11.5px;color:var(--muted);width:28px;flex-shrink:0}
.gb{flex:1;display:flex;flex-direction:column;gap:4px;min-width:0}
.bl{display:grid;grid-template-columns:38px 1fr 62px;gap:9px;align-items:center}
.bn{font-family:"IBM Plex Mono",monospace;font-size:10px;color:var(--faint);text-transform:uppercase;letter-spacing:.05em}
.bt{position:relative;height:15px;display:block}
.bar{position:absolute;top:0;height:15px;border-radius:2px;display:block;min-width:2px}
.bar.s-bass{background:var(--s-bass)}.bar.s-arp{background:var(--s-arp)}
.bar.ns{opacity:.34}
.err{position:absolute;top:7px;height:1px;background:var(--ink);opacity:.6;display:block}
.err::before,.err::after{content:"";position:absolute;top:-4px;width:1px;height:9px;background:var(--ink);opacity:.6}
.err::before{left:0}.err::after{right:0}
.bv{white-space:nowrap}
.bv{font-family:"IBM Plex Mono",monospace;font-size:12px;text-align:right;font-variant-numeric:tabular-nums}
.zero,.guard{position:absolute;top:0;bottom:2px;width:1px;pointer-events:none}
.zero{left:calc(28px + 12px + 38px + 9px + (100% - 28px - 12px - 38px - 9px - 62px - 9px)*0.5);
  background:var(--ink);opacity:.45}
.guard{background:var(--warn);opacity:.5}
.guard em{position:absolute;top:-4px;left:6px;font-style:normal;font-family:"IBM Plex Mono",monospace;
  font-size:9.5px;letter-spacing:.05em;color:var(--warn);white-space:nowrap;text-transform:uppercase}

.dots{display:flex;flex-direction:column;gap:12px;padding:16px 0 4px}
.dp{display:grid;grid-template-columns:38px 1fr 78px;gap:10px;align-items:center}
.dpl{font-family:"IBM Plex Mono",monospace;font-size:10px;color:var(--faint);text-transform:uppercase}
.dpt{position:relative;height:22px;display:block;border-bottom:1px solid var(--rule2)}
.baseline{position:absolute;top:2px;bottom:0;width:1px;background:var(--ink);opacity:.4;display:block}
.dot{position:absolute;top:5px;width:9px;height:9px;border-radius:50%;margin-left:-4.5px;display:block;
  box-shadow:0 0 0 2px var(--panel)}
.dot.s-bass{background:var(--s-bass)}.dot.s-arp{background:var(--s-arp)}
.dot em{position:absolute;top:11px;left:50%;transform:translateX(-50%);font-style:normal;
  font-family:"IBM Plex Mono",monospace;font-size:8.5px;color:var(--faint)}
.dpv{font-family:"IBM Plex Mono",monospace;font-size:10.5px;color:var(--muted);text-align:right}

.hrole{font-family:"IBM Plex Mono",monospace;font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--muted);margin:14px 0 6px}
.hgrid{display:grid;grid-template-columns:34px repeat(10,1fr);gap:2px;min-width:760px}
.hcorner{}
.hcol{font-family:"IBM Plex Mono",monospace;font-size:8.5px;color:var(--faint);text-align:center;
  padding-bottom:3px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.hrow{font-family:"IBM Plex Mono",monospace;font-size:10.5px;color:var(--muted);
  display:flex;align-items:center}
.hc{height:26px;border-radius:2px;background:color-mix(in oklab,var(--heat-hi) calc(var(--m)*100%),var(--heat-lo));
  display:flex;align-items:center;justify-content:center;font-family:"IBM Plex Mono",monospace;
  font-size:9px;color:var(--dead);font-weight:600}
.hc.dead{outline:1px dashed var(--dead);outline-offset:-1px}
.hscale{display:flex;align-items:center;gap:8px;margin-top:12px;font-family:"IBM Plex Mono",monospace;
  font-size:9.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.hbar{width:110px;height:9px;border-radius:1px;border:1px solid var(--rule);
  background:linear-gradient(90deg,var(--heat-lo),var(--heat-hi))}

.scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-family:"IBM Plex Mono",monospace;font-size:11.5px;
  font-variant-numeric:tabular-nums;min-width:720px}
th{text-align:right;font-weight:500;font-size:9px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--muted);padding:0 0 7px 12px;border-bottom:1px solid var(--rule);white-space:nowrap}
th:first-child,th:nth-child(2),th:nth-child(3){text-align:left}
td{padding:6px 0 6px 12px;border-bottom:1px solid var(--rule2);text-align:right;white-space:nowrap}
td:first-child,td:nth-child(2),td:nth-child(3){text-align:left}
tr:last-child td{border-bottom:0}
.cap{margin:0;font-size:12.5px;color:var(--muted);max-width:82ch}
.cap code{font-family:"IBM Plex Mono",monospace;font-size:11.5px;background:var(--panel2);padding:1px 5px;border-radius:2px}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}
footer{display:block;font-size:11.5px;color:var(--muted);font-family:"IBM Plex Mono",monospace;
  border-top:1px solid var(--rule);padding-top:14px}
#tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .1s;background:var(--ink);
  color:var(--panel);font-family:"IBM Plex Mono",monospace;font-size:11px;padding:5px 9px;
  border-radius:3px;z-index:99;white-space:nowrap;transform:translate(-50%,-140%)}
:focus-visible{outline:2px solid var(--s-arp);outline-offset:2px}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""
HTML = f"""<title>Fine-Tune Checkpoint Drift</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>{CSS}</style>
<div id="tip"></div>
<div class="plate">
  <header>
    <p class="eyebrow">Midi GPT &middot; checkpoint measurement</p>
    <h1>What fine-tuning did to the weights</h1>
    <p class="lede">No training history was ever logged for these runs, so nothing here is a
      training curve. Every number was measured now, by loading the four shipped checkpoints
      per role and re-running them. The distribution moved a long way; the two musical
      quality proxies did not follow it.</p>
  </header>

  <div class="card pad">
    <h2>Lineage</h2>
    <p class="sub">Read from the checkpoints' own metadata. <b>ft1 and ft2 both branch from base</b>
      &mdash; ft2 did not build on ft1, it replaced it.</p>
    <div class="lineage">
      <span class="node"><b>base</b> <em>&middot; from scratch</em></span>
      <span class="arrow">&rarr;</span>
      <span class="node dead"><b>ft1</b> <em>&middot; SFT, 4 edits &middot; superseded</em></span>
      <span class="arrow" style="margin-left:14px">base &rarr;</span>
      <span class="node"><b>ft2</b> <em>&middot; SFT, 10 edits</em></span>
      <span class="arrow">&rarr;</span>
      <span class="node"><b>ft3</b> <em>&middot; DPO, &beta;=0.1</em></span>
    </div>
  </div>

  <div class="tiles">
    <div class="tile"><div class="tn warn">{C['ARP3']}</div><div class="tl">arp ft3 corpus loss</div>
      <div class="td">Past the +15% regression guard &mdash; but that guard only runs in SFT, and ft3 is DPO.</div></div>
    <div class="tile"><div class="tn">{C['BASS3']}</div><div class="tl">bass ft3 corpus loss</div>
      <div class="td">Same chain, just under the threshold.</div></div>
    <div class="tile"><div class="tn">{C['ARPKL']}</div><div class="tl">arp KL from base</div>
      <div class="td">Roughly 3&times; the bass drift ({C['BASSKL']}) for fewer preference pairs.</div></div>
    <div class="tile"><div class="tn">0.0000</div><div class="tl">cond_proj drift</div>
      <div class="td">Identical to base at every stage, both roles. The CLAP layer never trained.</div></div>
    <div class="tile"><div class="tn">0 / 12</div><div class="tl">quality deltas resolved</div>
      <div class="td">No chord-fit or kick-lock change clears 2&nbsp;SE at n=200. The drift did not reach the output.</div></div>
  </div>

  <div class="card pad">
    <h2>Corpus loss, relative to base</h2>
    <p class="sub">Weighted cross-entropy on 900 mined examples. Right is worse &mdash; the base
      distribution eroding as edits accumulate.</p>
    <div class="legend">
      <span class="lgi"><span class="sw s-bass"></span>Bass</span>
      <span class="lgi"><span class="sw s-arp"></span>Arp</span>
    </div>
    {C['LOSS']}
  </div>

  <div class="card pad">
    <h2>KL divergence from base</h2>
    <p class="sub">Mean per-token KL over the full 65-token vocabulary, on sequences drawn from
      the corpus. Distribution-space drift, which weight norms alone do not show.</p>
    <div class="legend">
      <span class="lgi"><span class="sw s-bass"></span>Bass</span>
      <span class="lgi"><span class="sw s-arp"></span>Arp</span>
    </div>
    {C['KL']}
  </div>

  <div class="cols">
    <div class="card pad">
      <h2>Chord fit &mdash; change from base</h2>
      <p class="sub">Bars are &Delta; from base; whiskers are &plusmn;1 SE of that difference over
        200 generated clips. Every interval crosses zero, so none of it is resolved
        (<span class="mono">ns</span>).</p>
      {C['FIT']}
    </div>
    <div class="card pad">
      <h2>Kick lock &mdash; change from base</h2>
      <p class="sub">Same construction. Arp ft3 is the largest move at &minus;0.050 &plusmn; 0.030,
        which is still only 1.7&nbsp;SE &mdash; suggestive, not resolved.</p>
      {C['LOCK']}
    </div>
  </div>

  <div class="card pad">
    <h2>Weight drift by layer</h2>
    <p class="sub">Relative L2 distance from the base checkpoint, &Vert;&Delta;&Vert;/&Vert;base&Vert;.
      Fine-tuning reshapes the output head far more than the embeddings.</p>
    <div class="scroll">{C['HEAT']}</div>
    <div class="hscale"><span class="hbar"></span><span>0 &rarr; {C['WMAX']}</span>
      <span style="margin-left:8px">dashed = exactly zero</span></div>
  </div>

  <div class="card pad">
    <h2>All measurements</h2>
    <div class="scroll"><table>
      <thead><tr><th>Role</th><th>Stage</th><th>Method</th><th>n</th><th>Corpus loss</th>
        <th>vs base</th><th>KL</th><th>Chord fit ±SE</th><th>Kick lock ±SE</th></tr></thead>
      <tbody>{C['TABLE']}</tbody>
    </table></div>
  </div>

  <p class="cap"><b>Two kinds of number here.</b> Corpus loss, KL and weight drift are
    <b>deterministic</b> &mdash; teacher-forced or exact, no sampling, so the differences are real as
    shown. Chord fit and kick lock are <b>sampled</b> from 200 generated clips per stage at a fixed
    seed, and carry real sampling noise; at n=40 they told a confident story that disappeared at
    n=200, which is why they are drawn with error bars and marked <span class="mono">ns</span>.
    Method: every checkpoint loaded into the same architecture, scored on a fixed 900-example sample
    of the mined corpus with the training loss weights; <code>fit_and_lock</code> is the repo's own.
    Because no run logged its history, these are endpoint measurements &mdash; where each checkpoint
    landed, not the path it took.</p>

  <footer>4 checkpoints &times; 2 roles &middot; 900 corpus examples &middot; 200 clips per stage</footer>
</div>
<script>
(function(){{
  var tip=document.getElementById('tip');
  document.addEventListener('mouseover',function(e){{
    var t=e.target.closest('[data-tip]'); if(!t)return;
    tip.textContent=t.getAttribute('data-tip'); tip.style.opacity='1';
    var r=t.getBoundingClientRect();
    tip.style.left=(r.left+r.width/2)+'px'; tip.style.top=r.top+'px';
  }});
  document.addEventListener('mouseout',function(e){{
    if(e.target.closest('[data-tip]')) tip.style.opacity='0';
  }});
}})();
</script>
"""
open("checkpoint-drift.html","w").write(HTML)
head,rest=HTML.split("<style>",1); css,body=rest.split("</style>",1)
for t in ("light","dark"):
    open(f"checkpoint-drift.{t}.html","w").write(
        f'<!doctype html>\n<html lang="en" data-theme="{t}">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        + head + "<style>" + css + "</style>\n</head>\n<body>\n" + body.strip() + "\n</body>\n</html>\n")
print(f"wrote checkpoint-drift.html ({len(HTML):,} bytes) + standalones")
