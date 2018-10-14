from tkinter import *
import cv2
import sqlite3
import os
import time
import PIL.Image, PIL.ImageTk


class Application:

    def __init__(self, window, video_source=0):
        self.window = window

        self.video_source = video_source
        self.video = cv2.VideoCapture(self.video_source)
        self.video_height = 300
        self.video_width = 300
        # get scaling factor
        self.scaling_factor = self.video_height / float(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.video_width / float(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)) < self.scaling_factor:
            self.scaling_factor = self.video_width / float(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        # self.video_height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # self.video_width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        if not self.video.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self.canvas_bot = Canvas(window, width=self.video_width, height=self.video_height, background="white")
        self.canvas_bot.grid(row=0, column=0)

        self.frame_recommendation = Frame(window,
                                          width=self.video_width,
                                          height=self.video_height,
                                          background="white")
        self.frame_recommendation.grid(row=0, column=1)
        self.text_recommendation = Label(self.frame_recommendation, text="Rcommendation system")
        self.text_recommendation.pack(fill=BOTH, padx=20, pady=80)

        self.canvas_video = Canvas(window, width=self.video_width, height=self.video_height)
        self.canvas_video.grid(row=0, column=2)

        self.frame_chat = Frame(window,
                                width=self.video_width,
                                height=self.video_height,
                                background="white")
        self.frame_chat.grid(row=1, column=0)
        self.label_chat = Label(self.frame_chat, text="Conversation")
        self.label_chat.pack(fill=BOTH, padx=80, pady=20)

        self.btn_start = Button(window, text="Start", command=self.start_cam)
        self.btn_start.grid(row=2, column=0, sticky=W)

        self.btn_stop = Button(window, text="Stop", command=self.stop_cam)
        self.btn_stop.grid(row=2, column=0, sticky=E)

        # detector
        self.conn = sqlite3.connect('database.db')
        self.cursor = self.conn.cursor()
        self.fname = "recognizer/trainingData.yml"
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(self.fname)

        self.image = None
        self.run = False
        self.delay = 15
        # self.update()
        self.cam_stream()
        self.window.mainloop()

    def start_cam(self):
        self.video.open(self.video_source)

    def stop_cam(self):
        self.video.release()

    def cam_stream(self):
        if self.video.isOpened():
            ret, frame = self.video.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    ids, conf = self.recognizer.predict(gray[y:y + h, x:x + w])
                    self.cursor.execute("select name from users where id = (?);", (ids,))
                    result = self.cursor.fetchall()
                    name = result[0][0]
                    if conf < 50:
                        cv2.putText(frame, name, (x + 2, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 255, 0), 2)
                    else:
                        cv2.putText(frame, 'No Match', (x + 2, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                img = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img.thumbnail((self.video_height, self.video_width))

                self.image = PIL.ImageTk.PhotoImage(image=img)

                self.canvas_video.create_image(0, 0,
                                               image=self.image,
                                               anchor=NW)
        self.window.after(self.delay, self.cam_stream)


    # def update(self):
    #     # Get a frame from the video source
    #     ret, frame = self.video.get_frame()
    #
    #     if ret:
    #         self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
    #         self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
    #
    #     # self.window.after(self.delay, self.update)


# class VideoCapture:
#
#     def __init__(self, video_source=0):
#         # Open the video source
#         self.vid = cv2.VideoCapture(video_source)
#
#         if not self.vid.isOpened():
#             raise ValueError("Unable to open video source", video_source)
#
#         # Get video source width and height
#
#         self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
#         self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
#
#     def get_frame(self):
#         if self.vid.isOpened():
#             ret, frame = self.vid.read()
#             if ret:
#                 # Return a boolean success flag and the current frame converted to BGR
#                 return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#             else:
#                 return (ret, None)
#         else:
#             return None
#
#     # Release the video source when the object is destroyed
#     def __del__(self):
#         if self.vid.isOpened():
#             self.vid.release()


Application(Tk())
