import sqlite3, hashlib, secrets, time
import config as C

def conn():
    c = sqlite3.connect(C.DB_PATH, check_same_thread=False); c.row_factory = sqlite3.Row; return c

def init():
    with conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users(email TEXT PRIMARY KEY,name TEXT,salt TEXT,hash TEXT);
        CREATE TABLE IF NOT EXISTS sessions(token TEXT PRIMARY KEY,email TEXT,ts REAL);
        CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,ts REAL,zone TEXT,risk REAL,level TEXT,details TEXT);
        CREATE TABLE IF NOT EXISTS history(ts REAL,risk REAL,vehicles INT,speed REAL);""")

def _h(pw, salt): return hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), 200_000).hex()

def create_user(email, name, pw):
    salt = secrets.token_hex(16)
    try:
        with conn() as c: c.execute("INSERT INTO users VALUES(?,?,?,?)", (email, name, salt, _h(pw, salt)))
        return True
    except sqlite3.IntegrityError: return False

def login(email, pw):
    with conn() as c: u = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if not u or not secrets.compare_digest(u["hash"], _h(pw, u["salt"])): return None
    t = secrets.token_urlsafe(32)
    with conn() as c: c.execute("INSERT INTO sessions VALUES(?,?,?)", (t, email, time.time()))
    return t, u["name"]

def user_for(token):
    with conn() as c: return c.execute("SELECT u.name,u.email FROM sessions s JOIN users u ON u.email=s.email WHERE token=?", (token,)).fetchone()

def add_event(risk, level, details):
    with conn() as c: c.execute("INSERT INTO events(ts,zone,risk,level,details) VALUES(?,?,?,?,?)", (time.time(), C.ZONE_NAME, risk, level, details))

def add_history(risk, veh, speed):
    with conn() as c: c.execute("INSERT INTO history VALUES(?,?,?,?)", (time.time(), risk, veh, speed))

def events(n=20):
    with conn() as c: return [dict(r) for r in c.execute("SELECT * FROM events ORDER BY ts DESC LIMIT ?", (n,))]

def history(n=120):
    with conn() as c: return [dict(r) for r in c.execute("SELECT * FROM history ORDER BY ts DESC LIMIT ?", (n,))][::-1]
