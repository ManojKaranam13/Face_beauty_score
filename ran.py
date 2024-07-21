import cv2
import mediapipe as mp
import numpy as np
import requests
from io import BytesIO
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox

mp_face_mesh = mp.solutions.face_mesh

# Predefined URLs for the images
image_url1 = "https://"  # Replace with actual URL
image_url2 = "https://"  # Replace with actual URL

def get_facial_landmarks(image):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0]

def calculate_distances(landmarks, image_shape):
    h, w, _ = image_shape
    points = [(int(landmark.x * w), int(landmark.y * h)) for landmark in landmarks.landmark]

    distances = {}
    distances['eye_to_eye'] = np.linalg.norm(np.array(points[33]) - np.array(points[263]))
    distances['nose_to_chin'] = np.linalg.norm(np.array(points[1]) - np.array(points[152]))
    distances['mouth_width'] = np.linalg.norm(np.array(points[61]) - np.array(points[291]))
    distances['eye_to_mouth'] = np.linalg.norm(np.array(points[13]) - np.array(points[14]))
    distances['nose_width'] = np.linalg.norm(np.array(points[197]) - np.array(points[5]))

    return distances

def calculate_beauty_score(distances):
    normalized_distances = {k: v / max(distances.values()) for k, v in distances.items()}

    score = (
            0.3 * normalized_distances['eye_to_eye'] +
            0.2 * normalized_distances['nose_to_chin'] +
            0.2 * normalized_distances['mouth_width'] +
            0.2 * normalized_distances['eye_to_mouth'] +
            0.1 * normalized_distances['nose_width']
    )

    return score * 100  # Scale to a range from 0 to 100

def beauty_score(image):
    landmarks = get_facial_landmarks(image)

    if landmarks is None:
        return None, "No face detected"

    distances = calculate_distances(landmarks, image.shape)
    score = calculate_beauty_score(distances)

    return score, "Beauty score calculated successfully"

def download_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def display_result():
    image1 = download_image(image_url1)
    image2 = download_image(image_url2)

    score1, message1 = beauty_score(image1)
    score2, message2 = beauty_score(image2)

    if score1 is None or score2 is None:
        messagebox.showerror("Error", "One of the images does not contain a detectable face.")
        return

    selected_image = selected_image_var.get()

    correct = False
    if selected_image == 'image1' and score1 > score2:
        correct = True
    elif selected_image == 'image2' and score2 > score1:
        correct = True

    message = "You guessed correct!" if correct else "You guessed wrong."
    messagebox.showinfo("Result", message)

# Create the main window
root = tk.Tk()
root.title("Beauty Score Comparison! Guess who is more Attractive according to AI")

# Download images
image1_pil = Image.open(BytesIO(requests.get(image_url1).content))
image2_pil = Image.open(BytesIO(requests.get(image_url2).content))

# Resize images to fit in the window
image1_pil = image1_pil.resize((250, 250), Image.Resampling.LANCZOS)
image2_pil = image2_pil.resize((250, 250), Image.Resampling.LANCZOS)

# Convert images to PhotoImage format
image1_tk = ImageTk.PhotoImage(image1_pil)
image2_tk = ImageTk.PhotoImage(image2_pil)

# Create and place the Image 1
image1_label = tk.Label(root, image=image1_tk)
image1_label.pack(side="left", padx=10)

# Create and place the Image 2
image2_label = tk.Label(root, image=image2_tk)
image2_label.pack(side="right", padx=10)

# Create and place the radio buttons for image selection
selected_image_var = tk.StringVar()
tk.Radiobutton(root, text="Image 1", variable=selected_image_var, value="image1").pack()
tk.Radiobutton(root, text="Image 2", variable=selected_image_var, value="image2").pack()

# Create and place the Compare button
compare_button = tk.Button(root, text="Compare", command=display_result)
compare_button.pack()

# Run the application
root.mainloop()
