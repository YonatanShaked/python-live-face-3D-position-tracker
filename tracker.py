#!/usr/bin/env python
import os
import cv2 as cv
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk


class OBJ:
    def __init__(self, filename, swap_yz=False):
        self.vertices = []
        self.normals = []
        self.tex_coords = []
        self.faces = []
        print(f'loading obj model from: {filename}')
        for line in open(filename, "r"):
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swap_yz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swap_yz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.tex_coords.append(map(float, values[1:3]))
            elif values[0] == 'f':
                face = []
                tex_coords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        tex_coords.append(int(w[1]))
                    else:
                        tex_coords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, tex_coords))


class WebcamApp:
    def __init__(self, window, window_title):
        # cv init
        self.vid = cv.VideoCapture(0)
        if not self.vid.isOpened():
            print('Could not open camera')
            exit(1)

        self.width = int(self.vid.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.frame_rate = int(self.vid.get(cv.CAP_PROP_FPS))
        print(f'Camera Parameters: Width = {self.width}, Height = {self.height}, Frame rate = {self.frame_rate}')

        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        self.face_cascade = cv.CascadeClassifier(os.path.join(data_dir, 'haarcascade_frontalface_default.xml'))
        self.facemark = cv.face.createFacemarkLBF()
        self.facemark.loadModel(os.path.join(data_dir, 'lbfmodel.yaml'))
        self.obj = OBJ(os.path.join(data_dir, 'glasses.obj'), swap_yz=True)

        self.camera_matrix = np.eye(3, 3, dtype=np.float64)
        self.camera_matrix[0, 0] = self.height
        self.camera_matrix[1, 1] = self.height
        self.camera_matrix[0, 2] = self.width / 4
        self.camera_matrix[1, 2] = self.height / 4

        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        self.object_points = np.array([
            [8.27412, 1.33849, 10.63490],     # left eye corner 45
            [-8.27412, 1.33849, 10.63490],    # right eye corner 36
            [0, -4.47894, 17.73010],          # nose tip 30
            [-4.61960, -10.14360, 12.27940],  # right mouth corner 48
            [4.61960, -10.14360, 12.27940]    # left mouth corner 54
        ])

        self.object_points_ids = [45, 36, 30, 48, 54]

        self.prev_rvec = np.zeros((3, 1), dtype=np.float64)
        self.prev_tvec = np.zeros((3, 1), dtype=np.float64)
        self.lpf_enabled = False

        self.draw_debug = False

        # tk init
        self.window = window
        self.window.title(window_title)
        self.window.geometry(f'{self.width}x{self.height}')
        self.canvas = tk.Canvas(window)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.photo: ImageTk = None
        self.window.bind('<Key>', self._on_key_press)

    def start(self):
        self._update()
        self.window.mainloop()

    def quit(self):
        self.vid.release()
        self.window.destroy()

    def _processes_cv_frame(self, frame):
        frame_gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))

        if self.draw_debug:
            for (x, y, w, h) in faces:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 155, 155), 2)

        landmarks = []
        try:
            success, landmarks = self.facemark.fit(frame_gray, faces)
        except cv.error:
            success = False

        if success:
            for i in range(len(landmarks)):
                if self.draw_debug:
                    cv.face.drawFacemarks(frame, landmarks[i], (255, 255, 0))

                points2d = np.array([[landmarks[i][0][op_id][0], landmarks[i][0][op_id][1]] for op_id in self.object_points_ids])
                success, rvec, tvec = cv.solvePnP(self.object_points, points2d, self.camera_matrix, self.dist_coeffs, useExtrinsicGuess=True, rvec=np.array([[0.0], [0.0], [0.0]]), tvec=np.array([[0.0], [0.0], [0.0]]))
                if success:
                    if self.lpf_enabled:
                        rvec = 0.6 * self.prev_rvec + 0.4 * rvec
                        tvec = 0.6 * self.prev_tvec + 0.4 * tvec

                    if self.draw_debug:
                        cv.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 20)
                    self._render_obj(frame, self.obj, rvec, tvec)

                    self.prev_rvec = rvec
                    self.prev_tvec = tvec

    def _render_obj(self, frame, obj, rvec, tvec):
        vertices = obj.vertices
        scale_matrix = np.eye(3) * 0.2

        for face in obj.faces:
            face_vertices = face[0]
            points = np.array([vertices[vertex - 1] for vertex in face_vertices])
            points = np.dot(points, scale_matrix)
            points = np.array([[p[0], p[1], p[2]] for p in points])
            points = np.expand_dims(points, axis=1)
            imgpts, _ = cv.projectPoints(points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
            imgpts = np.int32(imgpts)
            cv.fillConvexPoly(frame, imgpts, (10, 10, 10))

    def _update(self):
        # cv frame handle
        ret, frame = self.vid.read()
        if ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            self._processes_cv_frame(frame)
            frame = cv.resize(frame, (self.window.winfo_width(), self.window.winfo_height()))
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # update tk
        self.window.after(1, self._update)

    def _on_key_press(self, event):
        if event.char == 'q':
            self.quit()
        elif event.char == 'd':
            self.draw_debug = not self.draw_debug
        elif event.char == 'l':
            self.lpf_enabled = not self.lpf_enabled
        elif event.char == '1':
            self.obj = OBJ(os.path.join('data', 'glasses.obj'), swap_yz=True)
        elif event.char == '2':
            self.obj = OBJ(os.path.join('data', 'hat.obj'), swap_yz=True)
        elif event.char == '3':
            self.obj = OBJ(os.path.join('data', 'head.obj'), swap_yz=True)


def main():
    root = tk.Tk()
    app = WebcamApp(root, 'webcam')
    app.start()


if __name__ == '__main__':
    main()
