"""
SQLite Database Layer for AI Physiotherapist.
Handles exercise session persistence, patient daily logs,
posture accuracy metrics, and physiotherapist daily reports.
"""

import sqlite3
import datetime
import json
from config import Config

def get_db_connection():
    conn = sqlite3.connect(Config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Table 1: Patient Profile
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patient_profile (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT DEFAULT 'Alex Mercer',
            patient_id TEXT DEFAULT 'PAT-2026-8841',
            physio_name TEXT DEFAULT 'Dr. Sarah Jenkins, PT',
            physio_email TEXT DEFAULT 'dr.jenkins@physiorehab.com',
            recovery_plan TEXT DEFAULT 'Post-Op Shoulder & Knee Rehabilitation Protocol',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Seed default profile if empty
    cursor.execute('SELECT COUNT(*) FROM patient_profile')
    if cursor.fetchone()[0] == 0:
        cursor.execute('''
            INSERT INTO patient_profile (patient_name, patient_id, physio_name, physio_email, recovery_plan)
            VALUES (?, ?, ?, ?, ?)
        ''', ('Alex Mercer', 'PAT-2026-8841', 'Dr. Sarah Jenkins, PT', 'dr.jenkins@physiorehab.com', 'Post-Op Shoulder & Knee Rehabilitation Protocol'))

    # Table 2: Exercise Sessions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS exercise_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exercise_id TEXT NOT NULL,
            exercise_name TEXT NOT NULL,
            reps_completed INTEGER DEFAULT 0,
            target_reps INTEGER DEFAULT 10,
            sets_completed INTEGER DEFAULT 1,
            avg_accuracy_score REAL DEFAULT 0.0,
            min_angle_achieved REAL DEFAULT 0.0,
            max_angle_achieved REAL DEFAULT 0.0,
            active_duration_seconds INTEGER DEFAULT 0,
            form_warnings TEXT DEFAULT '[]',
            session_date TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Table 3: Daily Summary Log
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            log_date TEXT UNIQUE NOT NULL,
            total_reps INTEGER DEFAULT 0,
            total_sessions INTEGER DEFAULT 0,
            overall_accuracy REAL DEFAULT 0.0,
            total_duration_minutes REAL DEFAULT 0.0,
            patient_pain_score INTEGER DEFAULT 0,
            notes TEXT DEFAULT ''
        )
    ''')

    # Table 4: Physiotherapist Sent Reports
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS physio_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_date TEXT NOT NULL,
            physio_email TEXT NOT NULL,
            patient_name TEXT NOT NULL,
            sessions_summary TEXT NOT NULL,
            therapist_notes TEXT DEFAULT '',
            status TEXT DEFAULT 'SENT',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()

def save_exercise_session(exercise_id, exercise_name, reps, target_reps, accuracy, duration_sec, min_angle, max_angle, warnings=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    today_str = datetime.date.today().isoformat()
    warnings_json = json.dumps(warnings if warnings else [])

    cursor.execute('''
        INSERT INTO exercise_sessions 
        (exercise_id, exercise_name, reps_completed, target_reps, avg_accuracy_score, 
         active_duration_seconds, min_angle_achieved, max_angle_achieved, form_warnings, session_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (exercise_id, exercise_name, reps, target_reps, accuracy, duration_sec, min_angle, max_angle, warnings_json, today_str))

    # Update or insert daily summary
    cursor.execute('SELECT * FROM daily_summaries WHERE log_date = ?', (today_str,))
    daily_row = cursor.fetchone()

    if daily_row:
        new_total_reps = daily_row['total_reps'] + reps
        new_total_sessions = daily_row['total_sessions'] + 1
        new_total_duration = daily_row['total_duration_minutes'] + (duration_sec / 60.0)
        # Weighted average for accuracy
        prev_weight = daily_row['total_reps']
        new_weight = prev_weight + reps
        new_accuracy = round(((daily_row['overall_accuracy'] * prev_weight) + (accuracy * reps)) / max(new_weight, 1), 1)

        cursor.execute('''
            UPDATE daily_summaries
            SET total_reps = ?, total_sessions = ?, overall_accuracy = ?, total_duration_minutes = ?
            WHERE log_date = ?
        ''', (new_total_reps, new_total_sessions, new_accuracy, new_total_duration, today_str))
    else:
        cursor.execute('''
            INSERT INTO daily_summaries (log_date, total_reps, total_sessions, overall_accuracy, total_duration_minutes)
            VALUES (?, ?, 1, ?, ?)
        ''', (today_str, reps, accuracy, round(duration_sec / 60.0, 2)))

    conn.commit()
    conn.close()

def get_today_summary():
    conn = get_db_connection()
    cursor = conn.cursor()
    today_str = datetime.date.today().isoformat()

    cursor.execute('SELECT * FROM daily_summaries WHERE log_date = ?', (today_str,))
    summary = cursor.fetchone()

    cursor.execute('SELECT * FROM exercise_sessions WHERE session_date = ? ORDER BY timestamp DESC', (today_str,))
    sessions = cursor.fetchall()

    conn.close()
    return {
        "summary": dict(summary) if summary else {
            "log_date": today_str, "total_reps": 0, "total_sessions": 0,
            "overall_accuracy": 0.0, "total_duration_minutes": 0.0
        },
        "sessions": [dict(s) for s in sessions]
    }

def get_recent_history(days=7):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT log_date, total_reps, total_sessions, overall_accuracy, total_duration_minutes
        FROM daily_summaries
        ORDER BY log_date DESC
        LIMIT ?
    ''', (days,))
    rows = cursor.fetchall()
    conn.close()

    result = [dict(r) for r in rows]
    # Reverse to return chronological order for charts
    result.reverse()
    return result

def save_physio_report(physio_email, patient_name, notes, sessions_summary):
    conn = get_db_connection()
    cursor = conn.cursor()
    today_str = datetime.date.today().isoformat()

    cursor.execute('''
        INSERT INTO physio_reports (report_date, physio_email, patient_name, sessions_summary, therapist_notes)
        VALUES (?, ?, ?, ?, ?)
    ''', (today_str, physio_email, patient_name, json.dumps(sessions_summary), notes))

    report_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return report_id

def get_patient_profile():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM patient_profile ORDER BY id DESC LIMIT 1')
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else {}

def update_patient_profile(patient_name, patient_id, physio_name, physio_email, recovery_plan):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE patient_profile
        SET patient_name = ?, patient_id = ?, physio_name = ?, physio_email = ?, recovery_plan = ?
        WHERE id = (SELECT MAX(id) FROM patient_profile)
    ''', (patient_name, patient_id, physio_name, physio_email, recovery_plan))
    conn.commit()
    conn.close()

# Auto initialize database on module load
init_db()
