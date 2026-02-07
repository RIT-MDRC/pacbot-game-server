import tkinter as tk
from tkinter import ttk
import cv2

def get_available_cameras(max_checks=10):
    available_cameras = []
    for i in range(max_checks):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

def get_camera_selection():
    root = tk.Tk()
    root.title("Camera Selection")
    
    # Get available cameras
    cameras = get_available_cameras()
    if not cameras:
        print("No cameras found!")
        root.destroy()
        return []

    # Variables to store selection
    cam1_var = tk.StringVar(value="None")
    cam2_var = tk.StringVar(value="None")
    
    camera_options = ["None"] + [str(c) for c in cameras]

    # UI Layout
    ttk.Label(root, text="Select Camera 1:").grid(row=0, column=0, padx=10, pady=10)
    combo1 = ttk.Combobox(root, textvariable=cam1_var, values=camera_options)
    combo1.grid(row=0, column=1, padx=10, pady=10)
    if cameras:
        combo1.current(1) # Default to first camera (after "None")

    ttk.Label(root, text="Select Camera 2:").grid(row=1, column=0, padx=10, pady=10)
    combo2 = ttk.Combobox(root, textvariable=cam2_var, values=camera_options)
    combo2.grid(row=1, column=1, padx=10, pady=10)
    if len(cameras) > 1:
         combo2.current(2) # Default to second camera (after "None")

    selected_cameras = []

    def on_submit():
        c1 = cam1_var.get()
        c2 = cam2_var.get()
        
        if c1 != "None":
            selected_cameras.append(int(c1))
        if c2 != "None":
            selected_cameras.append(int(c2))
            
        root.quit()
        root.destroy()

    ttk.Button(root, text="Start", command=on_submit).grid(row=2, column=0, columnspan=2, pady=20)

    root.mainloop()
    return selected_cameras

if __name__ == "__main__":
    print(f"Selected: {get_camera_selection()}")
