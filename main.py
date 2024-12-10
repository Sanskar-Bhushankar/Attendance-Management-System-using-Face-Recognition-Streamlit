import streamlit as st
import face_recognition
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
import csv
import plotly.express as px

# Setup page config
st.set_page_config(
    page_title="Face Recognition Attendance System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main {
            padding: 0rem 2rem;
        }
        .stButton>button {
            width: 100%;
            padding: 0.5rem;
            margin-top: 1rem;
        }
        .css-18e3th9 {
            padding-top: 2rem;
        }
        .stDataFrame {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #f0f2f6;
        }
    </style>
""", unsafe_allow_html=True)

# Predefined subjects
SUBJECTS = [
    "Select Subject",
    "Mathematics",
    "AWS Cloud Computing",
    "Networking",
    "Database Management",
    "Python Programming",
    "Web Development",
    "Machine Learning",
    "Cybersecurity",
    "Data Structures"
]

# Initialize session state variables
if 'attendance_df' not in st.session_state:
    st.session_state.attendance_df = pd.DataFrame(
        columns=['Name', 'Subject', 'Date', 'Time']
    )
if 'subject' not in st.session_state:
    st.session_state.subject = ""
if 'camera_on' not in st.session_state:
    st.session_state.camera_on = False

def load_known_faces():
    """Load known faces from Training_images directory"""
    known_faces = []
    known_names = []
    training_dir = "Training_images"
    
    for filename in os.listdir(training_dir):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(training_dir, filename)
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)[0]
            known_faces.append(face_encoding)
            known_names.append(os.path.splitext(filename)[0])
    
    return known_faces, known_names

def mark_attendance(name, subject):
    """Mark attendance in CSV file"""
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    
    # Add to session state DataFrame
    new_row = pd.DataFrame({
        'Name': [name],
        'Subject': [subject],
        'Date': [date],
        'Time': [time]
    })
    st.session_state.attendance_df = pd.concat([st.session_state.attendance_df, new_row], ignore_index=True)
    
    # Save to CSV
    attendance_file = "attendance.csv"
    with open(attendance_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if os.path.getsize(attendance_file) == 0:
            writer.writerow(['Name', 'Subject', 'Date', 'Time'])
        writer.writerow([name, subject, date, time])

def generate_analytics():
    """Generate attendance analytics"""
    if not st.session_state.attendance_df.empty:
        # Attendance by subject
        subject_counts = st.session_state.attendance_df['Subject'].value_counts()
        fig_subject = px.bar(
            subject_counts,
            title="Attendance by Subject",
            labels={'value': 'Count', 'index': 'Subject'}
        )
        
        # Attendance by date
        date_counts = st.session_state.attendance_df['Date'].value_counts()
        fig_date = px.line(
            date_counts,
            title="Attendance Trend",
            labels={'value': 'Count', 'index': 'Date'}
        )
        
        return fig_subject, fig_date
    return None, None

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150", caption="Attendance System")
        st.title("Navigation")
        page = st.radio("Go to", ["Take Attendance", "Analytics", "Settings"])
    
    if page == "Take Attendance":
        st.title("👤 Face Recognition Attendance System")
        
        # Create two columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Attendance Section")
            
            # Subject dropdown
            subject = st.selectbox(
                "Select Subject",
                options=SUBJECTS,
                key="subject_dropdown"
            )
            
            # Take attendance button
            if st.button("📸 Take Attendance", key="take_attendance"):
                if subject and subject != "Select Subject":
                    st.session_state.subject = subject
                    st.session_state.camera_on = True
                else:
                    st.error("⚠️ Please select a subject first")
            
            # Camera section
            if st.session_state.camera_on:
                with st.spinner("Loading camera..."):
                    known_faces, known_names = load_known_faces()
                    camera = cv2.VideoCapture(0)
                    frame_placeholder = st.empty()
                    stop_button = st.button("⏹️ Stop Camera")
                    
                    while not stop_button and st.session_state.camera_on:
                        ret, frame = camera.read()
                        if not ret:
                            st.error("Failed to capture frame")
                            break
                            
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        face_locations = face_recognition.face_locations(rgb_frame)
                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                        
                        # Draw rectangles around faces
                        for (top, right, bottom, left) in face_locations:
                            cv2.rectangle(rgb_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        
                        for face_encoding in face_encodings:
                            matches = face_recognition.compare_faces(known_faces, face_encoding)
                            if True in matches:
                                match_index = matches.index(True)
                                name = known_names[match_index]
                                mark_attendance(name, st.session_state.subject)
                                st.success(f"✅ Attendance marked for {name} in {st.session_state.subject}")
                                st.session_state.camera_on = False
                                break
                        
                        frame_placeholder.image(rgb_frame, channels="RGB")
                    
                    if stop_button:
                        st.session_state.camera_on = False
                    
                    camera.release()
        
        with col2:
            st.markdown("### Recent Attendance")
            if not st.session_state.attendance_df.empty:
                st.dataframe(
                    st.session_state.attendance_df.tail(),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No attendance records yet")
    
    elif page == "Analytics":
        st.title("📊 Attendance Analytics")
        
        fig_subject, fig_date = generate_analytics()
        
        if fig_subject and fig_date:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_subject, use_container_width=True)
            with col2:
                st.plotly_chart(fig_date, use_container_width=True)
            
            # Additional statistics
            st.subheader("📈 Statistics")
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            with stats_col1:
                st.metric("Total Attendance", len(st.session_state.attendance_df))
            with stats_col2:
                st.metric("Unique Students", st.session_state.attendance_df['Name'].nunique())
            with stats_col3:
                st.metric("Subjects Covered", st.session_state.attendance_df['Subject'].nunique())
        else:
            st.info("No data available for analytics")
    
    elif page == "Settings":
        st.title("⚙️ Settings")
        st.subheader("Export Data")
        if st.button("📥 Download Attendance Records"):
            csv = st.session_state.attendance_df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                "attendance_records.csv",
                "text/csv",
                key='download-csv'
            )
        
        st.subheader("Database Management")
        if st.button("🗑️ Clear All Records"):
            if st.session_state.attendance_df.empty:
                st.error("No records to clear")
            else:
                st.session_state.attendance_df = pd.DataFrame(
                    columns=['Name', 'Subject', 'Date', 'Time']
                )
                st.success("All records cleared successfully")

if __name__ == "__main__":
    main()