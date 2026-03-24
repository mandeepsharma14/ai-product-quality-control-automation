"""
DoorCo Manufacturing — AI Door Skin Quality Control Dashboard
Professional light theme · Program Manager portfolio project
Defects: crack, blister, crooked_corner, thin_paint,
         thick_paint, scratch, delamination
"""
import streamlit as st
import cv2, numpy as np, joblib, json, os, sys, random
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="DoorCo · AI Door Skin QC",
    page_icon="🚪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
:root {
  --bg:#F2F4F8; --surface:#FFFFFF; --border:#DDE1EA;
  --blue:#1D4ED8; --blue-lt:#EFF6FF;
  --green:#065F46; --green-lt:#D1FAE5;
  --red:#991B1B;   --red-lt:#FEE2E2;
  --amber:#92400E; --amber-lt:#FEF3C7;
  --purple:#4C1D95;--purple-lt:#EDE9FE;
  --orange:#9A3412;--orange-lt:#FFEDD5;
  --teal:#0F766E;  --teal-lt:#CCFBF1;
  --indigo:#3730A3;--indigo-lt:#E0E7FF;
  --text:#0F172A;  --text-2:#475569; --text-3:#94A3B8;
}
*,*::before,*::after{font-family:'DM Sans',system-ui,sans-serif!important;box-sizing:border-box}
[data-testid="stAppViewContainer"]{background:var(--bg)}
[data-testid="stSidebar"]{background:var(--surface);border-right:1px solid var(--border)}
[data-testid="stSidebar"]>div:first-child{padding-top:0!important}
.block-container{padding:1.8rem 2.6rem 2rem!important}

.banner{background:linear-gradient(112deg,#0C1E5B 0%,#1D4ED8 60%,#3B82F6 100%);
  border-radius:18px;padding:28px 38px;margin-bottom:28px;
  display:flex;align-items:center;justify-content:space-between;
  box-shadow:0 8px 32px rgba(29,78,216,.20)}
.banner h2{font-size:1.6rem;font-weight:700;color:#fff;margin:0 0 5px}
.banner p{font-size:.82rem;color:rgba(255,255,255,.70);margin:0}
.badge{background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.30);
  color:#fff;border-radius:20px;padding:4px 14px;font-size:.73rem;
  font-weight:600;margin:3px;display:inline-block}

.kpi{background:var(--surface);border-radius:14px;border:1px solid var(--border);
  border-top:4px solid var(--blue);padding:20px 22px;
  box-shadow:0 1px 4px rgba(0,0,0,.05)}
.kpi.g{border-top-color:#059669}.kpi.r{border-top-color:#DC2626}
.kpi.a{border-top-color:#D97706}.kpi.p{border-top-color:#7C3AED}
.kpi.o{border-top-color:#EA580C}.kpi.t{border-top-color:#0D9488}
.kpi-lbl{font-size:.66rem;font-weight:600;color:var(--text-2);
  text-transform:uppercase;letter-spacing:1px;margin:0 0 10px}
.kpi-val{font-size:1.9rem;font-weight:700;color:var(--text);margin:0;line-height:1}
.kpi.g .kpi-val{color:var(--green)}.kpi.r .kpi-val{color:var(--red)}
.kpi.a .kpi-val{color:var(--amber)}.kpi.p .kpi-val{color:var(--purple)}
.kpi.o .kpi-val{color:var(--orange)}.kpi.t .kpi-val{color:var(--teal)}
.kpi-sub{font-size:.73rem;color:var(--text-3);margin:8px 0 0}

.shdr{font-size:.75rem;font-weight:700;letter-spacing:.7px;text-transform:uppercase;
  color:var(--text-2);padding-bottom:8px;border-bottom:2px solid var(--border);
  margin:26px 0 14px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;
  padding:20px 22px;box-shadow:0 1px 4px rgba(0,0,0,.04);margin-bottom:14px}

.res-pass{background:var(--green-lt);border:2px solid #059669;border-radius:14px;
  padding:28px;text-align:center}
.res-fail{background:var(--red-lt);border:2px solid #DC2626;border-radius:14px;
  padding:28px;text-align:center}
.res-icon{font-size:2.2rem;font-weight:700;margin:0 0 8px}
.res-conf{font-size:1rem;font-weight:600;margin:0 0 4px}
.res-note{font-size:.8rem;color:var(--text-2);margin:6px 0 0}

.pb-row{display:flex;align-items:center;margin:6px 0}
.pb-name{width:148px;font-size:.78rem;color:var(--text);font-weight:500;flex-shrink:0}
.pb-rail{flex:1;height:11px;background:#F1F5F9;border-radius:6px;margin:0 10px;overflow:hidden}
.pb-fill{height:100%;border-radius:6px}
.pb-pct{width:40px;font-size:.78rem;font-weight:700;text-align:right}

[data-testid="stSidebar"] .stRadio label{font-size:.87rem!important;
  color:var(--text)!important;font-weight:500!important}
h1,h2,h3{color:var(--text)!important}
p,li,span{color:var(--text)!important}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASSES = ["good","crack","blister","crooked_corner",
           "thin_paint","thick_paint","scratch","delamination"]

CC = {
    "good":           "#059669",
    "crack":          "#DC2626",
    "blister":        "#D97706",
    "crooked_corner": "#7C3AED",
    "thin_paint":     "#0D9488",
    "thick_paint":    "#EA580C",
    "scratch":        "#1D4ED8",
    "delamination":   "#9F1239",
}
SEVERITY = {
    "good":0,"thin_paint":1,"thick_paint":1,
    "scratch":2,"blister":2,"crooked_corner":3,
    "crack":3,"delamination":3,
}
SL = {0:"None",1:"Minor",2:"Moderate",3:"Major"}
SBG= {"None":"#D1FAE5","Minor":"#FEF3C7","Moderate":"#FFEDD5","Major":"#FEE2E2"}
SFG= {"None":"#065F46","Minor":"#92400E","Moderate":"#9A3412","Major":"#991B1B"}

DINFO = {
    "good":          {"desc":"Meets all DoorCo quality standards. No defects detected.","action":"Clear for shipment","cost":0},
    "crack":         {"desc":"Structural crack in door skin. Door fails structural integrity test.","action":"Scrap — structural failure","cost":180},
    "blister":       {"desc":"Sub-surface air bubble causing paint/skin to lift.","action":"Sand and refinish or reject","cost":85},
    "crooked_corner":{"desc":"Corner geometry outside tolerance. Door will not fit frame.","action":"Scrap — dimensional reject","cost":120},
    "thin_paint":    {"desc":"Insufficient paint thickness. Fails warranty and weather resistance spec.","action":"Re-coat on paint line","cost":35},
    "thick_paint":   {"desc":"Excessive paint buildup. Causes door binding and appearance defect.","action":"Sand down or re-coat","cost":35},
    "scratch":       {"desc":"Surface scratch from handling, conveyor, or tooling contact.","action":"Touch-up or reject based on depth","cost":55},
    "delamination":  {"desc":"Door skin separating from substrate. Structural failure.","action":"Immediate scrap — safety reject","cost":200},
}
NSAMP = {"good":300,"crack":140,"blister":130,"crooked_corner":110,
         "thin_paint":130,"thick_paint":120,"scratch":130,"delamination":110}

def CL(h=310,**kw):
    base=dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#FAFBFF",
              font=dict(family="DM Sans,system-ui",color="#475569",size=11),
              margin=dict(l=8,r=8,t=38,b=8),height=h,
              xaxis=dict(gridcolor="#E8EDF2",linecolor="#E8EDF2",tickfont=dict(color="#64748B",size=10)),
              yaxis=dict(gridcolor="#E8EDF2",linecolor="#E8EDF2",tickfont=dict(color="#64748B",size=10)))
    base.update(kw); return base

# ── Model gate ─────────────────────────────────────────────────────────────────
model_ready = os.path.exists(os.path.join(BASE,"models/qc_model.pkl"))
if not model_ready:
    st.markdown("""<div style="max-width:500px;margin:100px auto;text-align:center;
      background:#fff;border:1px solid #DDE1EA;border-radius:18px;padding:48px">
      <div style="font-size:3rem;margin-bottom:16px">⚙️</div>
      <h2 style="color:#0F172A;margin:0 0 10px">Model Not Trained</h2>
      <p style="color:#64748B;font-size:.9rem;margin:0 0 24px">
        Generate 1,170 door skin images and train the classifier (~2–3 min).</p>
    </div>""", unsafe_allow_html=True)
    if st.button("🚀  Generate Data & Train Model", type="primary"):
        with st.spinner("Generating door skin defect images..."):
            import subprocess
            r1=subprocess.run([sys.executable,os.path.join(BASE,"data","generate_images.py")],
                              capture_output=True,text=True,cwd=BASE)
            if r1.returncode!=0: st.error(r1.stderr); st.stop()
        with st.spinner("Training CV classifier..."):
            r2=subprocess.run([sys.executable,os.path.join(BASE,"models","train.py")],
                              capture_output=True,text=True,cwd=BASE)
            if r2.returncode!=0: st.error(r2.stderr); st.stop()
        st.success("Ready!"); st.rerun()
    st.stop()

@st.cache_resource
def load_artifacts():
    m =joblib.load(os.path.join(BASE,"models/qc_model.pkl"))
    sc=joblib.load(os.path.join(BASE,"models/scaler.pkl"))
    with open(os.path.join(BASE,"models/model_results.json")) as f:
        r=json.load(f)
    return m,sc,r

model,scaler,results=load_artifacts()
best_name=results.get("best_model","GradientBoosting")
def _g(k,fb=0):
    if k in results: return results[k]
    return results.get("results",{}).get(best_name,{}).get(k,fb)
ACC=_g("accuracy"); F1W=_g("f1_weighted"); F1M=_g("f1_macro")

def featurize(img_bgr):
    img=cv2.resize(img_bgr,(224,224)); gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); f=[]
    hog=cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)
    f.extend(hog.compute(cv2.resize(gray,(64,64))).flatten()[::4][:200].tolist())
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    for ch,bins in zip(range(3),[18,8,8]):
        h=cv2.calcHist([hsv],[ch],None,[bins],[0,256])
        f.extend(cv2.normalize(h,h).flatten().tolist())
    lbp=np.zeros_like(gray,dtype=np.float32)
    for ai in range(24):
        a=2*np.pi*ai/24
        sh=np.roll(np.roll(gray.astype(np.float32),int(round(-3*np.sin(a))),0),int(round(3*np.cos(a))),1)
        lbp+=(sh>=gray.astype(np.float32)).astype(np.float32)*(2**ai)
    lh,_=np.histogram(lbp.flatten(),bins=32,range=(0,2**24))
    f.extend((lh/max(lh.sum(),1e-7)).tolist())
    edges=cv2.Canny(gray,50,150)
    f.extend([float(edges.mean()),float(edges.std()),float((edges>0).sum())/(224*224)])
    gx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3); gy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
    mag=np.sqrt(gx**2+gy**2)
    f.extend([float(mag.mean()),float(mag.std()),float(np.percentile(mag,90)),float(np.percentile(mag,99))])
    f.extend([float(gray.mean()),float(gray.std()),float(np.percentile(gray,10)),float(np.percentile(gray,90))])
    h3,w3=224//3,224//3
    for r in range(3):
        for c in range(3):
            z=gray[r*h3:(r+1)*h3,c*w3:(c+1)*w3]; f.extend([float(z.mean()),float(z.std())])
    _,thr=cv2.threshold(edges,10,255,cv2.THRESH_BINARY)
    cnts,_=cv2.findContours(thr,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    areas=sorted([cv2.contourArea(c) for c in cnts],reverse=True)[:5]
    while len(areas)<5: areas.append(0)
    f.extend(areas); f.append(float(len(cnts)))
    return np.array(f,dtype=np.float32)

def classify(img_bgr):
    X=scaler.transform(featurize(img_bgr).reshape(1,-1))
    proba=model.predict_proba(X)[0]
    pred=model.classes_[np.argmax(proba)]
    return pred,float(np.max(proba)),dict(zip(model.classes_,proba.tolist()))

def rand_sample(cls,split="val"):
    d=Path(os.path.join(BASE,f"data/images/{split}/{cls}"))
    if not d.exists(): return None
    files=list(d.glob("*.jpg"))
    return cv2.imread(str(random.choice(files))) if files else None

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style="background:linear-gradient(140deg,#0C1E5B,#1D4ED8);
      border-radius:14px;padding:22px 18px;margin:14px 6px 22px">
      <p style="font-size:1.5rem;text-align:center;margin:0 0 8px">🚪</p>
      <p style="font-size:.95rem;font-weight:700;color:#fff;text-align:center;margin:0 0 3px">
        DoorCo AI QC System</p>
      <p style="font-size:.71rem;color:rgba(255,255,255,.6);text-align:center;margin:0">
        Door Skin Defect Detection</p>
    </div>""", unsafe_allow_html=True)
    page=st.radio("Navigation",[
        "📊  Executive Dashboard",
        "🏭  Production Monitor",
        "🔬  Panel Inspector",
        "📈  Model Performance",
        "📚  Defect Library",
    ],label_visibility="collapsed")
    st.divider()
    st.markdown(f"""<div style="background:#F8FAFC;border:1px solid #DDE1EA;
      border-radius:12px;padding:16px">
      <p style="font-size:.63rem;font-weight:700;color:#94A3B8;text-transform:uppercase;
                letter-spacing:.9px;margin:0 0 12px">Active Model</p>
      <p style="font-size:.86rem;font-weight:700;color:#0F172A;margin:0 0 10px">
        🏆 {best_name}</p>
      {"".join([f'<div style="display:flex;justify-content:space-between;margin:5px 0">'
                f'<span style="font-size:.75rem;color:#64748B">{l}</span>'
                f'<span style="font-size:.75rem;font-weight:700;color:{vc}">{v}</span></div>'
                for l,v,vc in [("Accuracy",f"{ACC:.1%}","#059669"),
                                ("F1 Weighted",f"{F1W:.4f}","#1D4ED8"),
                                ("F1 Macro",f"{F1M:.4f}","#7C3AED"),
                                ("Defect Classes","8 types","#0F172A"),
                                ("Lines Deployed","Multi-site","#0F172A")]])}
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page=="📊  Executive Dashboard":
    st.markdown("""<div class="banner">
      <div>
        <h2>🚪 AI Door Skin Quality Control — Executive Dashboard</h2>
        <p>DoorCo Manufacturing · Computer Vision Defect Detection · Multi-Site Deployment</p>
      </div>
      <div>
        <span class="badge">✅ Live</span>
        <span class="badge">AWS + Azure</span>
        <span class="badge">8 Defect Types</span>
        <span class="badge">&lt;200ms</span>
      </div>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4,c5=st.columns(5)
    for col,(kc,lbl,val,sub) in zip([c1,c2,c3,c4,c5],[
        ("",  "Model Accuracy",  f"{ACC:.1%}",  "Gradient Boosting"),
        ("g", "F1 Weighted",     f"{F1W:.4f}",  "8 defect classes"),
        ("p", "Feature Vector",  "301 dims",    "HOG+LBP+Color+Edge"),
        ("a", "Inference",       "<200 ms",     "Per door skin"),
        ("r", "Defect Classes",  "8",           "Surface + dimensional"),
    ]):
        with col:
            st.markdown(f'<div class="kpi {kc}"><p class="kpi-lbl">{lbl}</p>'
                        f'<p class="kpi-val">{val}</p>'
                        f'<p class="kpi-sub">{sub}</p></div>',unsafe_allow_html=True)

    col1,col2=st.columns(2)
    with col1:
        st.markdown('<p class="shdr">Dataset — Samples per Defect Class</p>',unsafe_allow_html=True)
        fig=go.Figure(go.Pie(
            labels=[c.replace("_"," ").title() for c in CLASSES],
            values=[NSAMP[c] for c in CLASSES],
            marker_colors=[CC[c] for c in CLASSES],
            hole=0.55,textinfo="percent",textfont=dict(size=11),
            pull=[0.04 if c=="good" else 0 for c in CLASSES],
            hovertemplate="<b>%{label}</b><br>%{value} samples<br>%{percent}<extra></extra>",
        ))
        fig.add_annotation(text=f"<b>1,170</b><br>total",x=0.5,y=0.5,
                           font=dict(size=14,color="#0F172A"),showarrow=False)
        fig.update_layout(**CL(340,showlegend=True,
            legend=dict(orientation="v",x=1.0,y=0.5,font=dict(size=9,color="#475569"))))
        st.plotly_chart(fig,use_container_width=True)

    with col2:
        st.markdown('<p class="shdr">Business Cost per Escaped Defect (USD)</p>',unsafe_allow_html=True)
        defects=[c for c in CLASSES if c!="good"]
        fig2=go.Figure(go.Bar(
            x=[c.replace("_"," ").title() for c in defects],
            y=[DINFO[c]["cost"] for c in defects],
            marker_color=[SFG[SL[SEVERITY[c]]] for c in defects],
            marker_line_width=0,
            text=[f"${DINFO[c]['cost']}" for c in defects],
            textposition="outside",textfont=dict(size=11,color="#374151"),
        ))
        fig2.update_layout(**CL(340,showlegend=False,yaxis_title="Cost ($)",yaxis_range=[0,240]))
        fig2.update_xaxes(tickangle=-15)
        st.plotly_chart(fig2,use_container_width=True)

    st.markdown('<p class="shdr">Simulated Production Throughput — Last 8 Hours</p>',unsafe_allow_html=True)
    random.seed(42)
    hours=[f"{h:02d}:00" for h in range(8)]
    tots=[random.randint(220,270) for _ in hours]
    psd=[int(t*random.uniform(0.91,0.97)) for t in tots]
    fld=[t-p for t,p in zip(tots,psd)]
    prate=[p/t*100 for p,t in zip(psd,tots)]
    fig3=go.Figure()
    fig3.add_trace(go.Bar(name="Passed",x=hours,y=psd,marker_color="#059669",marker_line_width=0,opacity=0.88))
    fig3.add_trace(go.Bar(name="Failed",x=hours,y=fld,marker_color="#DC2626",marker_line_width=0,opacity=0.88))
    fig3.add_trace(go.Scatter(name="Pass Rate %",x=hours,y=prate,mode="lines+markers",
                              yaxis="y2",line=dict(color="#1D4ED8",width=2.5),
                              marker=dict(size=7,color="#1D4ED8",line=dict(color="white",width=1.5))))
    fig3.update_layout(**CL(280,barmode="stack",yaxis_title="Door Skins Inspected",
        yaxis2=dict(title="Pass Rate (%)",overlaying="y",side="right",range=[85,100],
                    showgrid=False,tickfont=dict(color="#1D4ED8",size=10)),
        legend=dict(orientation="h",y=1.12,font=dict(size=10))))
    st.plotly_chart(fig3,use_container_width=True)

    col1,col2=st.columns([1.3,0.7])
    with col1:
        st.markdown('<p class="shdr">Model Comparison</p>',unsafe_allow_html=True)
        all_res=results.get("results",{})
        if all_res:
            rows=[]
            for nm,m in all_res.items():
                rows.append({"Model":nm,"Accuracy":f"{m.get('accuracy',0):.1%}",
                             "F1 Weighted":f"{m.get('f1_weighted',0):.4f}",
                             "F1 Macro":f"{m.get('f1_macro',0):.4f}",
                             "Train (s)":m.get("train_time_s","—"),
                             "Status":"🏆 Active" if nm==best_name else "—"})
            st.dataframe(pd.DataFrame(rows)
                .style.apply(lambda r:["background:#D1FAE5;font-weight:600"
                             if r["Status"].startswith("🏆") else ""]*len(r),axis=1)
                .set_properties(**{"font-size":"12.5px"}),
                use_container_width=True,hide_index=True)
    with col2:
        st.markdown('<p class="shdr">Severity Distribution</p>',unsafe_allow_html=True)
        stots={}
        for c in CLASSES:
            s=SL[SEVERITY[c]]; stots[s]=stots.get(s,0)+NSAMP[c]
        sord=["None","Minor","Moderate","Major"]
        fig4=go.Figure(go.Pie(
            labels=[s for s in sord if s in stots],
            values=[stots[s] for s in sord if s in stots],
            marker_colors=[SFG[s] for s in sord if s in stots],
            hole=0.52,textinfo="percent+label",textfont=dict(size=10),
        ))
        fig4.update_layout(**CL(240,showlegend=False,margin=dict(l=0,r=0,t=20,b=0)))
        st.plotly_chart(fig4,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PRODUCTION MONITOR
# ══════════════════════════════════════════════════════════════════════════════
elif page=="🏭  Production Monitor":
    st.markdown("""<div class="banner">
      <div><h2>🏭 Production Line Monitor</h2>
      <p>Batch inspection · Real-time scoring · DoorCo Manufacturing</p></div>
      <div><span class="badge">🔄 Simulation</span></div>
    </div>""", unsafe_allow_html=True)

    items=[]
    for cls in CLASSES:
        d=Path(os.path.join(BASE,f"data/images/val/{cls}"))
        if d.exists():
            files=list(d.glob("*.jpg"))
            for fp in random.sample(files,min(2,len(files))):
                items.append({"path":str(fp),"cls":cls})
    random.shuffle(items)

    bres=[]
    for it in items:
        img=cv2.imread(it["path"])
        if img is None: continue
        pred,conf,probs=classify(img)
        bres.append({"img":img,"true":it["cls"],"pred":pred,"conf":conf,"probs":probs})

    n_tot=len(bres); n_pass=sum(1 for r in bres if r["pred"]=="good")
    n_fail=n_tot-n_pass; pp=n_pass/max(n_tot,1)
    dl=[r["pred"] for r in bres if r["pred"]!="good"]
    top=max(set(dl),key=dl.count) if dl else "none"

    c1,c2,c3,c4=st.columns(4)
    for col,(kc,lbl,val,sub) in zip([c1,c2,c3,c4],[
        ("",  "Batch Size", str(n_tot), "door skins inspected"),
        ("g", "Passed",     str(n_pass),f"{pp:.0%} pass rate"),
        ("r", "Failed",     str(n_fail),"flagged for hold"),
        ("a", "Top Defect", top.replace("_"," ").title() if top!="none" else "—","most frequent"),
    ]):
        with col:
            st.markdown(f'<div class="kpi {kc}"><p class="kpi-lbl">{lbl}</p>'
                        f'<p class="kpi-val">{val}</p>'
                        f'<p class="kpi-sub">{sub}</p></div>',unsafe_allow_html=True)

    st.markdown('<p class="shdr">Batch Inspection Grid</p>',unsafe_allow_html=True)
    for ri in range(0,len(bres),4):
        cols=st.columns(4)
        for ci,res in enumerate(bres[ri:ri+4]):
            with cols[ci]:
                ok=res["pred"]=="good"; bc=CC.get(res["pred"],"#1D4ED8")
                border="#059669" if ok else bc
                lbl="✅  PASS" if ok else f"❌  {res['pred'].replace('_',' ')}"
                sev=SL[SEVERITY.get(res["pred"],0)]
                st.markdown(f"<div style='border:2.5px solid {border};border-radius:10px;overflow:hidden'>",
                            unsafe_allow_html=True)
                st.image(cv2.cvtColor(res["img"],cv2.COLOR_BGR2RGB),use_container_width=True)
                bg=SBG.get(sev,"#F1F5F9") if not ok else "#D1FAE5"
                fg="#065F46" if ok else SFG.get(sev,"#374151")
                st.markdown(f"<div style='background:{bg};padding:5px;text-align:center'>"
                            f"<span style='color:{fg};font-size:.74rem;font-weight:700'>{lbl}</span><br>"
                            f"<span style='color:#64748B;font-size:.66rem'>{res['conf']:.0%} · {sev}</span>"
                            f"</div></div>",unsafe_allow_html=True)

    if st.button("🔄  Inspect New Batch",type="primary"): st.rerun()

    st.markdown('<p class="shdr">Defect Breakdown — This Batch</p>',unsafe_allow_html=True)
    dcnts={c:sum(1 for r in bres if r["pred"]==c) for c in CLASSES}
    fig_b=go.Figure(go.Bar(
        x=[c.replace("_"," ").title() for c in CLASSES],
        y=[dcnts[c] for c in CLASSES],
        marker_color=[CC[c] for c in CLASSES],marker_line_width=0,
        text=[dcnts[c] for c in CLASSES],textposition="outside",textfont=dict(color="#374151",size=11),
    ))
    fig_b.update_layout(**CL(250,showlegend=False,yaxis_title="Count",yaxis_range=[0,8]))
    st.plotly_chart(fig_b,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PANEL INSPECTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page=="🔬  Panel Inspector":
    st.markdown("""<div class="banner">
      <div><h2>🔬 Door Skin Inspector</h2>
      <p>Upload an image or pick a sample — full AI classification pipeline</p></div>
    </div>""", unsafe_allow_html=True)

    t1,t2=st.tabs(["📂  Upload Image","🎲  Use Dataset Sample"])
    img=None; analyze=False

    with t1:
        up=st.file_uploader("Drop a JPG or PNG door skin image",
                            type=["jpg","jpeg","png"],label_visibility="collapsed")
        if up:
            arr=np.asarray(bytearray(up.read()),dtype=np.uint8)
            img=cv2.imdecode(arr,cv2.IMREAD_COLOR); analyze=True

    with t2:
        cs=st.selectbox("Defect class",CLASSES,
                        format_func=lambda x:x.replace("_"," ").title())
        if st.button("🎲  Load Random Sample",type="secondary"):
            s=rand_sample(cs)
            if s is not None:
                st.session_state["si"]=s; st.session_state["sc"]=cs
        if "si" in st.session_state:
            img=st.session_state["si"]; analyze=True

    if analyze and img is not None:
        pred,conf,prob_dict=classify(img)
        ok=pred=="good"; sev=SL[SEVERITY.get(pred,0)]
        col_=CC.get(pred,"#1D4ED8"); info=DINFO.get(pred,{})

        L,R=st.columns([1,1])
        with L:
            st.markdown('<p class="shdr">Door Skin Image</p>',unsafe_allow_html=True)
            st.markdown(f"<div style='border:3px solid {col_};border-radius:12px;overflow:hidden'>",
                        unsafe_allow_html=True)
            st.image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),use_container_width=True)
            st.markdown("</div>",unsafe_allow_html=True)
            st.markdown('<p class="shdr" style="margin-top:22px">Canny Edge Map</p>',unsafe_allow_html=True)
            gray=cv2.cvtColor(cv2.resize(img,(224,224)),cv2.COLOR_BGR2GRAY)
            edges=cv2.Canny(gray,50,150)
            edg_col=cv2.applyColorMap(cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR),cv2.COLORMAP_CIVIDIS)
            ed=float((edges>0).mean())
            st.image(cv2.cvtColor(edg_col,cv2.COLOR_BGR2RGB),use_container_width=True,
                     caption=f"Edge density: {ed:.3f}  ·  Canny (50/150)  ·  224×224 px")

        with R:
            st.markdown('<p class="shdr">Classification Result</p>',unsafe_allow_html=True)
            if ok:
                st.markdown(f"""<div class="res-pass">
                  <p class="res-icon" style="color:#059669">✅  PASS</p>
                  <p class="res-conf" style="color:#065F46">Confidence: {conf:.1%}</p>
                  <p class="res-note">No defects detected · Clear for shipment</p>
                </div>""",unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="res-fail">
                  <p class="res-icon" style="color:#DC2626">❌  DEFECT DETECTED</p>
                  <p class="res-conf" style="color:#991B1B">
                    {pred.replace("_"," ").upper()} · {conf:.1%}</p>
                  <p class="res-note">Severity: {sev} · {info.get('action','Hold')}</p>
                </div>""",unsafe_allow_html=True)

            st.markdown(f"""<div class="card" style="border-top:4px solid {col_};margin-top:16px">
              <p style="font-size:.63rem;font-weight:700;color:#94A3B8;text-transform:uppercase;
                        letter-spacing:.9px;margin:0 0 12px">Defect Profile</p>
              <table style="width:100%;border-collapse:collapse">
                <tr><td style="font-size:.78rem;color:#64748B;padding:4px 0;width:95px;vertical-align:top">Type</td>
                    <td style="font-size:.78rem;font-weight:600;color:#0F172A">{pred.replace("_"," ").title()}</td></tr>
                <tr><td style="font-size:.78rem;color:#64748B;padding:4px 0">Severity</td>
                    <td><span style="background:{SBG.get(sev,'#F1F5F9')};color:{SFG.get(sev,'#374151')};
                      font-size:.71rem;font-weight:700;padding:2px 10px;border-radius:12px">{sev}</span></td></tr>
                <tr><td style="font-size:.78rem;color:#64748B;padding:4px 0">Cost Risk</td>
                    <td style="font-size:.78rem;font-weight:600;color:#0F172A">${info.get('cost',0)} per escape</td></tr>
                <tr><td style="font-size:.78rem;color:#64748B;padding:4px 0;vertical-align:top">Description</td>
                    <td style="font-size:.78rem;color:#374151">{info.get('desc','')}</td></tr>
              </table></div>""",unsafe_allow_html=True)

            st.markdown('<p class="shdr">Confidence by Class</p>',unsafe_allow_html=True)
            for cls_p,p in sorted(prob_dict.items(),key=lambda x:x[1],reverse=True):
                cp=CC.get(cls_p,"#1D4ED8"); pct=int(p*100)
                st.markdown(f"<div class='pb-row'>"
                            f"<span class='pb-name'>{cls_p.replace('_',' ')}</span>"
                            f"<div class='pb-rail'><div class='pb-fill' style='width:{pct}%;background:{cp}'></div></div>"
                            f"<span class='pb-pct' style='color:{cp}'>{p:.1%}</span>"
                            f"</div>",unsafe_allow_html=True)

            fig_g=go.Figure(go.Indicator(
                mode="gauge+number",value=conf*100,
                title=dict(text="Classification Confidence",font=dict(size=11,color="#64748B")),
                number=dict(suffix="%",font=dict(size=26,color="#0F172A")),
                gauge=dict(axis=dict(range=[0,100],tickfont=dict(size=8,color="#94A3B8")),
                           bar=dict(color=col_,thickness=0.26),bgcolor="#F2F4F8",borderwidth=0,
                           steps=[dict(range=[0,60],color="#FEE2E2"),
                                  dict(range=[60,80],color="#FEF3C7"),
                                  dict(range=[80,100],color="#D1FAE5")],
                           threshold=dict(line=dict(color="#0F172A",width=2.5),thickness=0.75,value=80))
            ))
            fig_g.update_layout(**CL(200,margin=dict(l=18,r=18,t=36,b=8)))
            st.plotly_chart(fig_g,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page=="📈  Model Performance":
    st.markdown(f"""<div class="banner">
      <div><h2>📈 Model Performance Analytics</h2>
      <p>Accuracy · F1 scores · Confusion matrix · Per-class breakdown</p></div>
      <div><span class="badge">🏆 {best_name}</span><span class="badge">{ACC:.1%}</span></div>
    </div>""",unsafe_allow_html=True)

    all_res=results.get("results",{}); best_r=all_res.get(best_name,{})
    per_cls=best_r.get("per_class",best_r.get("per_class_f1",{}))

    c1,c2,c3,c4=st.columns(4)
    for col,(kc,lbl,val) in zip([c1,c2,c3,c4],[
        ("",  "Accuracy",   f"{best_r.get('accuracy',ACC):.1%}"),
        ("g", "F1 Weighted",f"{best_r.get('f1_weighted',F1W):.4f}"),
        ("p", "F1 Macro",   f"{best_r.get('f1_macro',F1M):.4f}"),
        ("a", "Train Time", f"{best_r.get('train_time_s','—')}s"),
    ]):
        with col:
            st.markdown(f'<div class="kpi {kc}"><p class="kpi-lbl">{lbl}</p>'
                        f'<p class="kpi-val">{val}</p></div>',unsafe_allow_html=True)

    c1,c2=st.columns(2)
    with c1:
        st.markdown('<p class="shdr">Per-Class F1 Score</p>',unsafe_allow_html=True)
        if per_cls:
            sc=sorted(per_cls.items(),key=lambda x:x[1])
            fig_f=go.Figure(go.Bar(
                x=[v for _,v in sc],
                y=[k.replace("_"," ").title() for k,_ in sc],
                orientation="h",marker_color=[CC[k] for k,_ in sc],marker_line_width=0,
                text=[f"{v:.3f}" for _,v in sc],textposition="outside",
                textfont=dict(size=11,color="#374151"),
            ))
            fig_f.add_vline(x=0.90,line_dash="dot",line_color="#94A3B8",line_width=1.5)
            fig_f.update_layout(**CL(380,xaxis_range=[0.7,1.1],xaxis_title="F1 Score"))
            st.plotly_chart(fig_f,use_container_width=True)

    with c2:
        st.markdown('<p class="shdr">All Models — Accuracy vs F1</p>',unsafe_allow_html=True)
        if all_res:
            nms=list(all_res.keys())
            fig_c=go.Figure()
            fig_c.add_trace(go.Bar(name="Accuracy",x=nms,
                y=[all_res[n].get("accuracy",0) for n in nms],
                marker_color=["#059669" if n==best_name else "#93C5FD" for n in nms],
                marker_line_width=0,
                text=[f"{all_res[n].get('accuracy',0):.1%}" for n in nms],textposition="outside"))
            fig_c.add_trace(go.Bar(name="F1 Weighted",x=nms,
                y=[all_res[n].get("f1_weighted",0) for n in nms],
                marker_color=["#1D4ED8" if n==best_name else "#BFDBFE" for n in nms],
                marker_line_width=0,
                text=[f"{all_res[n].get('f1_weighted',0):.4f}" for n in nms],textposition="outside"))
            fig_c.update_layout(**CL(380,barmode="group",yaxis_range=[0.80,1.02],
                legend=dict(orientation="h",y=1.12,font=dict(size=10))))
            st.plotly_chart(fig_c,use_container_width=True)

    if "confusion_matrix" in best_r:
        st.markdown('<p class="shdr">Confusion Matrix</p>',unsafe_allow_html=True)
        cm=np.array(best_r["confusion_matrix"])
        cm_norm=cm.astype(float)/cm.sum(axis=1,keepdims=True)
        cl_labs=[c.replace("_"," ").title() for c in CLASSES]
        c1,c2=st.columns(2)
        for col,data,scale,title in [(c1,cm,"Blues","Raw Counts"),(c2,cm_norm,"Greens","Normalised (Recall)")]:
            with col:
                fig=px.imshow(data,x=cl_labs,y=cl_labs,color_continuous_scale=scale,
                              labels=dict(x="Predicted",y="Actual",color=""),
                              text_auto=".0f" if scale=="Blues" else ".2f",title=title)
                fig.update_layout(**CL(400))
                fig.update_xaxes(tickangle=-30,tickfont=dict(size=8,color="#64748B"))
                fig.update_yaxes(tickfont=dict(size=8,color="#64748B"))
                st.plotly_chart(fig,use_container_width=True)

    st.markdown('<p class="shdr">Per-Class Metrics Table</p>',unsafe_allow_html=True)
    per_prec=best_r.get("per_class_prec",{}); per_rec=best_r.get("per_class_rec",{})
    rows=[]
    for cls in CLASSES:
        sev=SL[SEVERITY[cls]]
        rows.append({"Class":cls.replace("_"," ").title(),"Severity":sev,
                     "Precision":round(per_prec.get(cls,per_cls.get(cls,0)),3),
                     "Recall":round(per_rec.get(cls,per_cls.get(cls,0)),3),
                     "F1":round(per_cls.get(cls,0),3),
                     "Cost/Escape":f"${DINFO[cls]['cost']}"})
    st.dataframe(pd.DataFrame(rows).style.set_properties(**{"font-size":"12.5px"}),
                 use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — DEFECT LIBRARY
# ══════════════════════════════════════════════════════════════════════════════
elif page=="📚  Defect Library":
    st.markdown("""<div class="banner">
      <div><h2>📚 Door Skin Defect Library</h2>
      <p>Visual reference · 8 defect types · Severity classifications · Remediation actions</p></div>
    </div>""",unsafe_allow_html=True)

    for i in range(0,len(CLASSES),2):
        cols=st.columns(2)
        for j,cls in enumerate(CLASSES[i:i+2]):
            with cols[j]:
                info=DINFO[cls]; col_=CC[cls]; sev=SL[SEVERITY[cls]]
                d=Path(os.path.join(BASE,f"data/images/val/{cls}"))
                files=list(d.glob("*.jpg")) if d.exists() else []
                if files:
                    bg=cv2.imread(str(random.choice(files)))
                    if bg is not None:
                        st.markdown(f"<div style='border:2.5px solid {col_};border-radius:10px;overflow:hidden;margin-bottom:10px'>",
                                    unsafe_allow_html=True)
                        st.image(cv2.cvtColor(bg,cv2.COLOR_BGR2RGB),use_container_width=True)
                        st.markdown("</div>",unsafe_allow_html=True)
                st.markdown(f"""<div class="card" style="border-left:5px solid {col_}">
                  <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;flex-wrap:wrap">
                    <span style="font-size:.9rem;font-weight:700;color:{col_}">
                      {cls.replace("_"," ").upper()}</span>
                    <span style="background:{SBG[sev]};color:{SFG[sev]};font-size:.68rem;
                      font-weight:700;padding:2px 10px;border-radius:12px">{sev}</span>
                    <span style="background:#EFF6FF;color:#1D4ED8;font-size:.68rem;
                      font-weight:700;padding:2px 10px;border-radius:12px">
                      ${info['cost']}/escape</span>
                  </div>
                  <p style="font-size:.82rem;color:#374151;margin:0 0 8px">{info['desc']}</p>
                  <p style="font-size:.78rem;margin:0">
                    <span style="color:#94A3B8;font-weight:600">Action: </span>
                    <span style="color:#374151">{info['action']}</span>
                  </p></div>""",unsafe_allow_html=True)
