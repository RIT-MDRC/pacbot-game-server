import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import time


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
    # Row 0: Camera 1 selection
    ttk.Label(root, text="Select Camera 1:").grid(row=0, column=0, padx=10, pady=10)
    combo1 = ttk.Combobox(root, textvariable=cam1_var, values=camera_options)
    combo1.grid(row=0, column=1, padx=10, pady=10)

    # Row 1: Camera 2 selection
    ttk.Label(root, text="Select Camera 2:").grid(row=1, column=0, padx=10, pady=10)
    combo2 = ttk.Combobox(root, textvariable=cam2_var, values=camera_options)
    combo2.grid(row=1, column=1, padx=10, pady=10)

    # Row 2: Previews
    preview1_label = ttk.Label(root)
    preview1_label.grid(row=2, column=0, padx=10, pady=10)

    preview2_label = ttk.Label(root)
    preview2_label.grid(row=2, column=1, padx=10, pady=10)

    caps = {"cam1": None, "cam2": None}
    current_indices = {"cam1": -1, "cam2": -1}

    def update_previews():
        for key, var, label, cap_key in [
            ("cam1", cam1_var, preview1_label, "cam1"),
            ("cam2", cam2_var, preview2_label, "cam2"),
        ]:
            val = var.get()
            if val == "None":
                if caps[cap_key] is not None:
                    caps[cap_key].release()
                    caps[cap_key] = None
                    current_indices[cap_key] = -1
                label.config(image="")
                label.image = None
                continue

            idx = int(val)

            # Re-open if index changed
            if current_indices[cap_key] != idx:
                if caps[cap_key] is not None:
                    caps[cap_key].release()

                caps[cap_key] = cv2.VideoCapture(idx)
                if caps[cap_key].isOpened():
                    current_indices[cap_key] = idx
                else:
                    caps[cap_key] = None
                    current_indices[cap_key] = -1
                    label.config(image="")
                    label.image = None
                    continue

            current_cap = caps[cap_key]
            if current_cap and current_cap.isOpened():
                ret, frame = current_cap.read()
                if ret:
                    # Resize for preview
                    frame = cv2.resize(frame, (320, 240))
                    # Convert BGR to RGB
                    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(cv2image)
                    imgtk = ImageTk.PhotoImage(image=img)
                    label.imgtk = imgtk  # Keep a reference
                    label.configure(image=imgtk)

        root.after(30, update_previews)

    if cameras:
        combo1.current(1)  # Default to first camera (after "None")
    if len(cameras) > 1:
        combo2.current(2)  # Default to second camera (after "None")

    selected_cameras = []

    def on_submit():
        c1 = cam1_var.get()
        c2 = cam2_var.get()

        if c1 != "None":
            selected_cameras.append(int(c1))
        if c2 != "None":
            selected_cameras.append(int(c2))

        # Release all caps before closing
        for cap in caps.values():
            if cap:
                cap.release()

        time.sleep(0.2)

        root.quit()
        root.destroy()

    # Move Start button to row 3
    ttk.Button(root, text="Start", command=on_submit).grid(
        row=3, column=0, columnspan=2, pady=20
    )

    # Start update loop
    update_previews()

    root.mainloop()
    return selected_cameras


if __name__ == "__main__":
    print(f"Selected: {get_camera_selection()}")
