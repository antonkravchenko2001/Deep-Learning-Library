import cv2
import tkinter as tk
from PIL import ImageGrab
from Model import *

"""erase digit"""


def clear_widget():
    global cv
    cv.delete("all")


def activate_event(event):
    global lastx, lasty
    cv.bind("<B1-Motion>", draw_lines)
    lastx, lasty = event.x, event.y


"""draw digit"""


def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=8, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE, splinesteps=12)
    lastx, lasty = x, y


"""recognise digit"""


def recognize_digit():
    global image_number
    filename = f'image_{image_number}.png'
    widget = cv
    x = root.winfo_rootx() + widget.winfo_x()
    y = root.winfo_rooty() + widget.winfo_y()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()
    ImageGrab.grab().crop((x, y, x1, y1)).save(filename)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    """capture contours"""
    cotours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for ctr in cotours:
        """pick only sufficiently large contours"""
        if cv2.contourArea(ctr) > 1500 and cv2.contourArea(ctr) < 5000:
            rect = cv2.boundingRect(ctr)
            x, y, w, h = rect
            img = th[y:y + h, x:x + w]
            """create a rectangle around region of interest"""
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 1)
            """resize region of interest"""
            img = cv2.resize(img, (28, 28))
            """scale region of interest"""
            img = img/255
            """reshape region of interest"""
            img = img.reshape(1, 1, 28*28)
            """predict digit"""
            prediction = Model.predict(img)
            prediction = prediction[0]
            data = str(prediction)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontscale = 0.5
            color = (255, 0, 0)
            thickness = 1
            """output prediction on the screen"""
            cv2.putText(image, data, (x, y-5), font, fontscale, color, thickness)
            cv2.imshow('image', image)
            cv2.waitKey(0)


root = tk.Tk()
root.resizable(0, 0)
root.title("Hand written digit recognizer")
lastx, lasty = None, None
image_number = 0
"""create Canvas"""
cv = tk.Canvas(root, width=640, height=480, bg="white")
cv.grid(row=0, column=0, pady=2, sticky=tk.W, columnspan=2)
cv.bind("<Button-1>", activate_event)
"""button to recognize digit"""
button_save = tk.Button(text="recognize digit", command=recognize_digit)
button_save.grid(row=2, column=0, pady=1, padx=1)
button_clear = tk.Button(text="clear widget", command=clear_widget)
button_clear.grid(row=2, column=1, pady=1, padx=1)
root.mainloop()