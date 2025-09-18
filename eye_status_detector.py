import os, cv2, numpy as np, pandas as pd, tempfile, base64
import streamlit as st
from ultralytics import YOLO
from sort import Sort
from gtts import gTTS
from fpdf import FPDF

# Optional exports
try:
    import firebase_admin
    from firebase_admin import credentials, db
except:
    firebase_admin = None
try:
    from pymongo import MongoClient
except:
    MongoClient = None

# Multilingual labels
LANGS = {"English": "en", "தமிழ்": "ta", "हिन्दी": "hi"}
lang = st.sidebar.selectbox("Language / மொழி / भाषा", list(LANGS.keys()))
TRANS = {
    "Upload Video": {"ta": "வீடியோ பதிவேற்று", "hi": "वीडियो अपलोड करें"},
    "Process Video": {"ta": "வீடியோ செயலாக்கம்", "hi": "वीडियो संसाधित करें"},
    "Player Stats": {"ta": "பிளேயர் புள்ளியியல்", "hi": "खिलाड़ी आँकड़े"},
    "Event Log": {"ta": "நிகழ்வு பதிவு", "hi": "इवेंट लॉग"},
    "Download PDF": {"ta": "PDF பதிவிறக்கவும்", "hi": "PDF डाउनलोड करें"},
    "Download Highlight": {"ta": "ஹைலைட் பதிவிறக்கவும்", "hi": "हाइलाइट डाउनलोड करें"},
    "Export to": {"ta": "ஏற்றுமதி", "hi": "निर्यात करें"},
}
def _(label):
    code = LANGS[lang]
    return TRANS.get(label, {}).get(code, label)

st.set_page_config(layout="wide", page_title="Football Analysis Full App")
st.title("⚽ Football Match Analysis")

upload = st.file_uploader(_( "Upload Video" ), type=["mp4","avi"])
if not upload:
    st.stop()
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
tmp.write(upload.read())
video_path = tmp.name

# Model & Tracker
yolo = YOLO("yolov8x.pt")
tracker = Sort()

# Data stores
player_positions = {}
player_stats = {}
events = []  # frame, type, pid, team, timestamp
highlights = []

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W, H = 1280, 720
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_detected.mp4", fourcc, fps, (W, H))

# Goals
left_goal = (0,200,50,520)
right_goal = (1230,200,1280,520)
shot_thresh = 25
frame_num = 0
last_pid = None
last_frame = -30
last_pass = None
dribble = {}
team_pos = {"A":0,"B":0}

def role(frame, box):
    x1,y1,x2,y2 = box
    crop = frame[y1:y2, x1:x2]
    if crop.size==0: return "A"
    m=crop.mean(axis=(0,1)); b=m.mean()
    if b>180 and m[1]>m[0]>m[2]: return "R"
    elif b<90: return "A"
    else: return "B"

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.resize(frame, (W,H))
    frame_num +=1
    res = yolo(frame)
    dets, balls = [], []
    for r in res:
        for b in r.boxes:
            cls,conf = int(b.cls[0]), float(b.conf[0])
            x1,y1,x2,y2 = map(int,b.xyxy[0])
            if cls==0 and conf>0.6: dets.append([x1,y1,x2,y2,conf])
            elif cls==32 and conf>0.2: balls.append((x1,y1,x2,y2))
    tracks = tracker.update(np.array(dets)) if dets else []
    id2team = {}
    for tr in tracks:
        x1,y1,x2,y2,pid = map(int,tr)
        t = role(frame,(x1,y1,x2,y2))
        id2team[pid]=t
        if t!="R":
            cx,cy=(x1+x2)//2,(y1+y2)//2
            player_positions.setdefault(pid,[]).append((cx,cy))
    ball_center=None
    if balls:
        bx1,by1,bx2,by2=balls[0]
        ball_center=((bx1+bx2)//2,(by1+by2)//2)
        last_center=ball_center
    elif 'last_center' in locals(): ball_center=last_center
    if ball_center:
        best,md=None,1e9
        for tr in tracks:
            pid=int(tr[4])
            if id2team.get(pid)=="R": continue
            cx,cy=(tr[0]+tr[2])//2,(tr[1]+tr[3])//2
            d=np.linalg.norm(np.array(ball_center)-np.array((cx,cy)))
            if d<md: md=d; best=pid
        if best is not None:
            t=id2team[best]
            s=player_stats.setdefault(best,{"team":t,"poss":0,"passes":0,"shots":0,"goals":0,"dribbles":0})
            s["poss"]+=1; team_pos[t]+=1
            # pass
            if last_pid and last_pid!=best:
                fg=frame_num-last_frame
                if last_pass:
                    mv=np.linalg.norm(np.array(ball_center)-np.array(last_pass))
                    if fg>5 and mv>15:
                        player_stats[last_pid]["passes"]+=1
                        events.append({"frame":frame_num,"type":"pass","pid":last_pid,"team":player_stats[last_pid]["team"],"timestamp":frame_num/fps})
                        highlights.append((frame_num-10,frame_num+10))
            # goal/shot
            bx,by=ball_center
            lin=left_goal[0]<=bx<=left_goal[2] and left_goal[1]<=by<=left_goal[3]
            rin=right_goal[0]<=bx<=right_goal[2] and right_goal[1]<=by<=right_goal[3]
            if lin and t=="B":
                s["goals"]+=1; events.append({"frame":frame_num,"type":"goal","pid":best,"team":t,"timestamp":frame_num/fps})
                highlights.append((frame_num-20,frame_num+20))
            elif rin and t=="A":
                s["goals"]+=1; events.append({"frame":frame_num,"type":"goal","pid":best,"team":t,"timestamp":frame_num/fps})
                highlights.append((frame_num-20,frame_num+20))
            # shot
            if 'last_center' in locals():
                dv=np.linalg.norm(np.array(ball_center)-np.array(last_center))
                if dv>shot_thresh:
                    s["shots"]+=1
                    events.append({"frame":frame_num,"type":"shot","pid":best,"team":t,"timestamp":frame_num/fps})
                    highlights.append((frame_num-10,frame_num+10))
            # dribble
            if last_pid==best:
                dribble[best]=dribble.get(best,0)+1
                if dribble[best]>=15:
                    s["dribbles"]+=1
                    events.append({"frame":frame_num,"type":"dribble","pid":best,"team":t,"timestamp":frame_num/fps})
            else:
                dribble[best]=0
            last_pid=best; last_frame=frame_num; last_pass=ball_center
    out.write(frame)
cap.release(); out.release()

# Save CSVs
pd.DataFrame(events).to_csv("event_log.csv", index=False)
pd.DataFrame.from_dict(player_stats, orient="index").rename_axis("player_id").reset_index().to_csv("player_stats.csv", index=False)

# Extract highlight clips
if not os.path.exists("highlight_clips"): os.makedirs("highlight_clips")
cap2 = cv2.VideoCapture(video_path)
for idx,(a,b) in enumerate(highlights):
    cap2.set(cv2.CAP_PROP_POS_FRAMES, max(0,a))
    vw = cv2.VideoWriter(f"highlight_clips/high_{idx}.mp4", fourcc, fps, (W,H))
    for f in range(a,b+1):
        ret,fr = cap2.read()
        if not ret: break
        vw.write(fr)
    vw.release()
cap2.release()

# TTS commentary
audio_list=[]
for ev in events:
    txt=f"Event {ev['type']} by player {ev['pid']} team {ev['team']}"
    mp3=f"tts_{ev['frame']}.mp3"
    gTTS(txt, lang="en").save(mp3)
    audio_list.append(mp3)

# PDF summary
pdf=FPDF(); pdf.add_page(); pdf.set_font("Arial",size=12)
pdf.cell(0,10,txt="Match Summary", ln=True, align="C")
for pid, stats in player_stats.items():
    line=f"P{pid}(Team {stats['team']}) Poss {stats['poss']} Pass {stats['passes']} Shot {stats['shots']} Goal {stats['goals']} Dribble {stats['dribbles']}"
    pdf.cell(0,8,txt=line, ln=True)
pdf.output("summary_report.pdf")

# Export option
opt = st.sidebar.selectbox(_( "Export to" ), ["None","Firebase","MongoDB"])
if opt=="Firebase" and firebase_admin:
    cred=credentials.Certificate("path_to_credentials.json")
    firebase_admin.initialize_app(cred,{"databaseURL":"YOUR_DB_URL"})
    db.reference("match/stats").set(player_stats)
if opt=="MongoDB" and MongoClient:
    client = MongoClient("mongodb://localhost:27017")
    client.football.matches.insert_one({"stats":player_stats,"events":events})

# Streamlit UI
st.header(_( "Player Stats" ))
st.dataframe(pd.read_csv("player_stats.csv"))

st.header(_( "Event Log" ))
ev = pd.read_csv("event_log.csv")
st.dataframe(ev)

st.header("🎞️ Detected Video")
with open("output_detected.mp4",'rb') as f:
    st.video(f.read())

for idx,p in enumerate(highlights):
    if os.path.exists(f"highlight_clips/high_{idx}.mp4"):
        st.header(f"{_( 'Download Highlight' )} #{idx+1}")
        with open(f"highlight_clips/high_{idx}.mp4",'rb') as fh:
            st.download_button(f"Clip {idx+1}", data=fh.read(), file_name=f"highlight_{idx}.mp4", mime="video/mp4")

st.header(_( "Download PDF" ))
with open("summary_report.pdf",'rb') as pf:
    st.download_button("PDF", data=pf.read(), file_name="summary_report.pdf", mime="application/pdf")
