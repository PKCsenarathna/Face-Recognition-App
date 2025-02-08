import tkinter as tk
from tkinter import filedialog, Label, Toplevel, Button, Entry, Listbox, Scrollbar
import cv2
import numpy as np
import face_recognition
from PIL import Image, ImageTk
import os
import mysql.connector
import pickle

# MySQL Database Connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="face_recognition_db"
)
cursor = db.cursor()

# Create table if it doesn't exist
cursor.execute("""
    CREATE TABLE IF NOT EXISTS known_faces (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        encoding BLOB NOT NULL
    )
""")

# Load known faces from database
known_faces = []
known_names = []

def load_known_faces():
    """Fetch all known faces from database"""
    global known_faces, known_names
    known_faces.clear()
    known_names.clear()

    cursor.execute("SELECT name, encoding FROM known_faces")
    for name, encoding_blob in cursor.fetchall():
        encoding = pickle.loads(encoding_blob)
        known_faces.append(encoding)
        known_names.append(name)

# Load faces initially
load_known_faces()

def identify_face(file_path):
    """Identifies a face from the uploaded image"""
    if not known_faces:
        result_label.config(text="No known faces available for recognition.")
        return

    image = cv2.imread(file_path)
    image = cv2.resize(image, (500, 500))
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_image)
    if not face_locations:
        result_label.config(text="No face detected in the uploaded image.")
        return

    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(face_distances) if face_distances.size else None

        name = "Unknown"
        if best_match_index is not None and matches[best_match_index]:
            name = known_names[best_match_index]

        result_label.config(text=f"Identified: {name}")

def upload_image():
    """Opens file dialog to select an image and previews before processing"""
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        initialdir=os.getcwd(),
        filetypes=()
    )
    
    if file_path:
        image = Image.open(file_path)
        image.thumbnail((300, 300))

        preview_window = Toplevel()
        preview_window.title("Image Preview")

        img = ImageTk.PhotoImage(image)
        preview_label = Label(preview_window, image=img)
        preview_label.image = img
        preview_label.pack(pady=10)

        def confirm_upload():
            preview_window.destroy()
            identify_face(file_path)

        confirm_button = Button(preview_window, text="Confirm Upload", command=confirm_upload)
        confirm_button.pack(pady=10)

def add_new_face():
    """Opens a file dialog to add a new face to the database"""
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        initialdir=os.getcwd(),
        filetypes=()
    )
    
    if file_path:
        new_face_window = Toplevel()
        new_face_window.title("Enter Name")

        Label(new_face_window, text="Enter Name:").pack(pady=5)
        name_entry = Entry(new_face_window)
        name_entry.pack(pady=5)

        def save_face():
            name = name_entry.get().strip()
            if not name:
                result_label.config(text="Name cannot be empty.")
                return

            image = face_recognition.load_image_file(file_path)
            encoding = face_recognition.face_encodings(image)

            if encoding:
                encoding_blob = pickle.dumps(encoding[0])

                cursor.execute("INSERT INTO known_faces (name, encoding) VALUES (%s, %s)", (name, encoding_blob))
                db.commit()

                load_known_faces()
                result_label.config(text=f"Face for {name} added successfully!")
                new_face_window.destroy()
            else:
                result_label.config(text="No face detected in the image.")

        Button(new_face_window, text="Save Face", command=save_face).pack(pady=10)

def delete_face():
    """Deletes a selected face from the database"""
    delete_window = Toplevel()
    delete_window.title("Delete a Face")

    Label(delete_window, text="Select a face to delete:").pack(pady=5)

    listbox = Listbox(delete_window)
    scrollbar = Scrollbar(delete_window, orient="vertical")
    listbox.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=listbox.yview)

    for name in known_names:
        listbox.insert(tk.END, name)

    listbox.pack(pady=5, side=tk.LEFT, fill=tk.BOTH)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def confirm_delete():
        selected_index = listbox.curselection()
        if selected_index:
            name_to_delete = listbox.get(selected_index[0])

            cursor.execute("DELETE FROM known_faces WHERE name = %s", (name_to_delete,))
            db.commit()

            load_known_faces()
            result_label.config(text=f"Deleted: {name_to_delete}")
            delete_window.destroy()

    Button(delete_window, text="Delete", command=confirm_delete).pack(pady=5)

def live_recognition():
    """Opens webcam and recognizes faces in real-time"""
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            best_match_index = np.argmin(face_distances) if face_distances.size else None

            if best_match_index is not None and matches[best_match_index]:
                name = known_names[best_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Live Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Create GUI
root = tk.Tk()
root.title("Face Recognition App")
root.geometry("400x500")

Label(root, text="Face Recognition App", font=("Arial", 16)).pack(pady=10)

upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=5)

add_face_button = tk.Button(root, text="Add New Face", command=add_new_face)
add_face_button.pack(pady=5)

delete_face_button = tk.Button(root, text="Delete Face", command=delete_face)
delete_face_button.pack(pady=5)

live_button = tk.Button(root, text="Live Recognition", command=live_recognition)
live_button.pack(pady=5)

result_label = Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()

# Close MySQL connection
cursor.close()
db.close()
