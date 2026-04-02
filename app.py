# app.py
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3, cv2, mediapipe as mp, math, datetime

app = Flask(__name__)
app.secret_key = 'pbl_academic_key_2026'

# --- CONFIGURATION ---
config = {"CAMERA_INDEX": 0, "EAR_THRESHOLD": 0.25, "CONSECUTIVE_FRAMES": 1, "MIN_OPEN_FRAMES_AFTER_BLINK": 3}

def init_db():
    conn = sqlite3.connect('cognitive_load.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, timestamp TEXT, final_status TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS telemetry_data (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER, time_offset INTEGER, ear REAL, moe REAL, gsr REAL)')
    conn.commit(); conn.close()

init_db()

# --- GLOBALS ---
show_mesh = False
recording = False
current_session_id = None
tick_count = 0
system_metrics = {"ear":0.0, "mar":0.0, "moe":0.0, "blinks":0, "status":"IDLE", "gsr":0}

LEFT_EYE = [362, 385, 387, 263, 373, 380]; RIGHT_EYE = [33, 160, 158, 133, 153, 144]; INNER_LIPS = [78, 82, 13, 312, 308, 317, 14, 87]

def euclidean_distance(p1, p2): return math.dist([p1.x, p1.y], [p2.x, p2.y])

def calculate_ear(landmarks, eye_indices):
    p1,p2,p3,p4,p5,p6 = [landmarks[i] for i in eye_indices]
    return (euclidean_distance(p2, p6) + euclidean_distance(p3, p5)) / (2.0 * euclidean_distance(p1, p4))

def calculate_mar(landmarks, indices):
    v1 = euclidean_distance(landmarks[indices[1]], landmarks[indices[7]])
    v2 = euclidean_distance(landmarks[indices[2]], landmarks[indices[6]])
    v3 = euclidean_distance(landmarks[indices[3]], landmarks[indices[5]])
    h1 = euclidean_distance(landmarks[indices[0]], landmarks[indices[4]])
    return (v1 + v2 + v3) / (2.0 * h1)

def generate_frames():
    global system_metrics, show_mesh
    cap = cv2.VideoCapture(config["CAMERA_INDEX"])
    blink_c, f_c, op_f, pending = 0, 0, 0, False
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
        while cap.isOpened():
            if system_metrics['blinks'] == 0: blink_c = 0 
            success, frame = cap.read()
            if not success: break
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    ear = (calculate_ear(face_landmarks.landmark, LEFT_EYE) + calculate_ear(face_landmarks.landmark, RIGHT_EYE)) / 2.0
                    mar = calculate_mar(face_landmarks.landmark, INNER_LIPS)
                    moe = mar/ear if ear > 0.01 else mar/0.01
                    if ear < config["EAR_THRESHOLD"]: 
                        f_c += 1; op_f = 0
                        if f_c >= config["CONSECUTIVE_FRAMES"]: pending = True
                    else:
                        op_f += 1; f_c = 0
                        if pending and op_f >= config["MIN_OPEN_FRAMES_AFTER_BLINK"]: blink_c += 1; pending = False
                    system_metrics.update({"ear":round(ear,3), "mar":round(mar,3), "moe":round(moe,2), "blinks":blink_c, "status":"ACTIVE"})
                    if show_mesh:
                        h, w, _ = frame.shape
                        for conn in mp_face_mesh.FACEMESH_FACE_OVAL:
                            p1, p2 = face_landmarks.landmark[conn[0]], face_landmarks.landmark[conn[1]]
                            cv2.line(frame, (int(p1.x*w),int(p1.y*h)), (int(p2.x*w),int(p2.y*h)), (255,255,255), 1)
                        for idx in LEFT_EYE + RIGHT_EYE + INNER_LIPS:
                            pt = face_landmarks.landmark[idx]
                            cv2.circle(frame, (int(pt.x*w), int(pt.y*h)), 3, (255, 255, 255), -1); cv2.circle(frame, (int(pt.x*w), int(pt.y*h)), 4, (0, 0, 0), 1)
            ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- AUTH ROUTES (Fixed 404s) ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        u, p = request.form['username'], generate_password_hash(request.form['password'])
        conn = sqlite3.connect('cognitive_load.db'); c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?,?)", (u, p))
            conn.commit(); return redirect(url_for('login'))
        except:
            error = "User already exists!"
        finally:
            conn.close()
    return render_template('register.html', error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        u, p = request.form['username'], request.form['password']
        conn = sqlite3.connect('cognitive_load.db'); c = conn.cursor()
        c.execute("SELECT id, password FROM users WHERE username=?",(u,)); user = c.fetchone(); conn.close()
        if user and check_password_hash(user[1], p):
            session.update({'logged_in':True, 'user_id':user[0], 'username':u})
            return redirect(url_for('index'))
        else:
            error = "Invalid credentials."
    return render_template('login.html', error=error)

@app.route('/logout')
def logout(): stop_session(); session.clear(); return redirect(url_for('login'))

# --- SESSION & HARDWARE API ---
@app.route('/api/start_session', methods=['POST'])
def start_session():
    global recording, current_session_id, tick_count, system_metrics
    recording = True; tick_count = 0; system_metrics["blinks"] = 0
    conn = sqlite3.connect('cognitive_load.db'); c = conn.cursor()
    c.execute("INSERT INTO sessions (user_id, timestamp, final_status) VALUES (?, ?, 'RECORDING')", (session['user_id'], datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    current_session_id = c.lastrowid
    conn.commit(); conn.close()
    return jsonify({"status": "success"})

@app.route('/api/stop_session', methods=['POST'])
def stop_session():
    global recording, current_session_id
    if recording:
        conn = sqlite3.connect('cognitive_load.db'); c = conn.cursor()
        c.execute("UPDATE sessions SET final_status = 'COMPLETED' WHERE id = ?", (current_session_id,))
        conn.commit(); conn.close()
        recording = False; current_session_id = None
    return jsonify({"status": "success"})

@app.route('/api/log_telemetry', methods=['POST'])
def log_telemetry():
    if recording:
        global tick_count; tick_count += 1
        conn = sqlite3.connect('cognitive_load.db'); c = conn.cursor()
        c.execute("INSERT INTO telemetry_data (session_id, time_offset, ear, moe, gsr) VALUES (?, ?, ?, ?, ?)", (current_session_id, tick_count, system_metrics['ear'], system_metrics['moe'], system_metrics['gsr']))
        conn.commit(); conn.close()
    return jsonify({"status": "ok"})

# --- DATA RETRIEVAL ---
@app.route('/')
def index():
    if not session.get('logged_in'): return redirect(url_for('login'))
    return render_template('index.html', username=session['username'], user_id=session['user_id'])

@app.route('/api/history')
def get_history():
    conn = sqlite3.connect('cognitive_load.db'); conn.row_factory = sqlite3.Row; c = conn.cursor()
    c.execute("SELECT * FROM sessions WHERE user_id = ? ORDER BY id DESC", (session['user_id'],))
    res = [dict(row) for row in c.fetchall()]
    for s in res:
        if s['id'] != current_session_id and s['final_status'] == 'RECORDING': s['final_status'] = 'COMPLETED'
    conn.close(); return jsonify(res)

@app.route('/api/session_graph/<int:sid>')
def session_graph(sid):
    conn = sqlite3.connect('cognitive_load.db'); conn.row_factory = sqlite3.Row; c = conn.cursor()
    c.execute("SELECT * FROM telemetry_data WHERE session_id = ? ORDER BY time_offset ASC", (sid,)); data = [dict(row) for row in c.fetchall()]; conn.close(); return jsonify(data)

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/api/metrics')
def metrics(): return jsonify(system_metrics)
@app.route('/api/hardware', methods=['POST'])
def hardware_update(): system_metrics["gsr"] = request.get_json().get('gsr', 0); return jsonify({"status": "success"})
@app.route('/api/toggle_mesh', methods=['POST'])
def toggle_mesh():
    global show_mesh; show_mesh = request.get_json().get('show_mesh', False)
    return jsonify({"status": "success"})
@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    new_threshold = request.get_json().get('threshold')
    if new_threshold: config["EAR_THRESHOLD"] = round(new_threshold, 3); return jsonify({"status": "success", "new_threshold": config["EAR_THRESHOLD"]})
    return jsonify({"status": "error"}), 400

if __name__ == '__main__': app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)