import os, sys, time, uuid, shutil
sys.path.insert(0, os.path.dirname(__file__))
from fastapi import FastAPI, Depends, Header, Query, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import db

app = FastAPI(title="RoadGuard AI"); pipe = None
class Cred(BaseModel): email: str; password: str; name: str = ""
class Src(BaseModel): source: str

@app.on_event("startup")
def boot():
    global pipe
    from vision import Pipeline
    db.init(); os.makedirs("uploads", exist_ok=True); pipe = Pipeline(); pipe.start()

def auth(authorization: str = Header(None), token: str = Query(None)):
    t = (authorization or "").replace("Bearer ", "") or token
    u = db.user_for(t) if t else None
    if not u: raise HTTPException(401, "Unauthorized")
    return u

@app.post("/api/signup")
def signup(c: Cred):
    if len(c.password) < 6 or "@" not in c.email or not c.name.strip(): raise HTTPException(400, "Enter name, valid email and 6+ char password")
    if not db.create_user(c.email.lower(), c.name.strip(), c.password): raise HTTPException(409, "Account already exists")
    t, n = db.login(c.email.lower(), c.password); return {"token": t, "name": n}

@app.post("/api/login")
def login(c: Cred):
    r = db.login(c.email.lower(), c.password)
    if not r: raise HTTPException(401, "Invalid email or password")
    return {"token": r[0], "name": r[1]}

@app.get("/api/stats")
def stats(u=Depends(auth)): return pipe.stats
@app.get("/api/events")
def events(u=Depends(auth)): return db.events()
@app.get("/api/history")
def history(u=Depends(auth)): return db.history()

@app.post("/api/source")
def source(s: Src, u=Depends(auth)): pipe.start(s.source); return {"ok": True}

@app.post("/api/upload")
def upload(file: UploadFile = File(...), u=Depends(auth)):
    p = f"uploads/{uuid.uuid4().hex}{os.path.splitext(file.filename)[1] or '.mp4'}"
    with open(p, "wb") as f: shutil.copyfileobj(file.file, f)
    pipe.start(p); return {"ok": True}

@app.get("/video_feed")
def feed(u=Depends(auth)):
    def gen():
        while True:
            if pipe.jpeg: yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + pipe.jpeg + b"\r\n"
            time.sleep(.04)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")

@app.get("/")
def index(): return FileResponse("app/static/index.html")
