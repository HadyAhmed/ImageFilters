from tkinter import *
from tkinter import filedialog

import cv2
import numpy as np

# Dealing with tkinter For Python GUI
root = Tk()
root.geometry("500x320")
root.title("Image Manipulation")
cons = 10


# Function Used in Select Button to open file explorer and open image
def goto():
    global path
    global image
    path = filedialog.askopenfilename()

    # Save the incoming Image into variable image
    image = cv2.imread(path)

    cv2.namedWindow('Original Image Selected', cv2.WINDOW_NORMAL)
    cv2.imshow('Original Image Selected', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test():
    print(path)


def sharp():
    kernel_sharp = np.array([[-1, -1, -1],
                             [-1, 9, -1],
                             [-1, -1, -1]])

    sharpened = cv2.filter2D(image, -1, kernel_sharp)
    cv2.namedWindow('Image sharp', cv2.WINDOW_NORMAL)
    cv2.imshow("Image sharp", sharpened)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def blur():
    blur = cv2.GaussianBlur(image, (7, 7), 0)

    cv2.namedWindow('Gaussian Blur', cv2.WINDOW_NORMAL)
    cv2.imshow('Gaussian Blur', blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gray():
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('Grayscale', cv2.WINDOW_NORMAL)
    cv2.imshow('Grayscale', gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Laplacian():
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    cv2.namedWindow('laplacian', cv2.WINDOW_NORMAL)
    cv2.imshow('laplacian', laplacian)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def threshold():
    grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    retval, threshold = cv2.threshold(grayscaled, 127, 255, cv2.THRESH_BINARY)
    retval2, threshold2 = cv2.threshold(
        grayscaled, 127, 255, cv2.THRESH_BINARY_INV)

    cv2.namedWindow('threshold', cv2.WINDOW_NORMAL)
    cv2.imshow('threshold', threshold)
    cv2.namedWindow('threshold2', cv2.WINDOW_NORMAL)
    cv2.imshow('threshold2', threshold2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def red():
    # split function  splits image into each color index
    B, G, R = cv2.split(image)

    # create matrix of zeros
    # with dimensions of the image h * w
    zeros = np.zeros(image.shape[:2], dtype="uint8")

    cv2.namedWindow('Red', cv2.WINDOW_NORMAL)
    cv2.imshow("Red", cv2.merge([zeros, zeros, R]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def green():
    # split function  splits image into each color index
    B, G, R = cv2.split(image)

    # create matrix of zeros
    # with dimensions of the image h * w
    zeros = np.zeros(image.shape[:2], dtype="uint8")

    cv2.namedWindow('Green', cv2.WINDOW_NORMAL)
    cv2.imshow("Green", cv2.merge([zeros, G, zeros]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def blue():
    # split function  splits image into each color index
    B, G, R = cv2.split(image)

    # create matrix of zeros
    # with dimensions of the image h * w
    zeros = np.zeros(image.shape[:2], dtype="uint8")

    cv2.namedWindow('Blue', cv2.WINDOW_NORMAL)
    cv2.imshow("Blue", cv2.merge([B, zeros, zeros]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def edge():
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    new = cv2.Canny(gray_image, 70, 170)

    cv2.namedWindow('edge', cv2.WINDOW_NORMAL)
    cv2.imshow("edge", new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def median():
    median = cv2.medianBlur(image, 5)

    cv2.namedWindow('median', cv2.WINDOW_NORMAL)
    cv2.imshow('median', median)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def equl():
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray_image)

    res = np.hstack((gray_image, equ))

    cv2.namedWindow('equl', cv2.WINDOW_NORMAL)
    cv2.imshow("equl", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def equlColor():
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    res = np.hstack((image, img_output))

    cv2.namedWindow('Color input image', cv2.WINDOW_NORMAL)
    cv2.imshow('Color input image', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plus():
    global cons

    cons = cons + 10

    if (cons > 255):
        cons = 255


def mins():
    global cons

    cons = cons - 10

    if (cons < 0):
        cons = 0


def sum():
    global path
    global image2

    path = filedialog.askopenfilename()

    image2 = cv2.imread(path)

    lol = cv2.add(image, image2)

    cv2.namedWindow('Color input image', cv2.WINDOW_NORMAL)
    cv2.imshow('Color input image', lol)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def saturation2():
    global cons

    lol = cv2.add(image, cons)

    cv2.namedWindow('Color input image', cv2.WINDOW_NORMAL)
    cv2.imshow('Color input image', lol)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Forming The UI Layout
btn1 = Button(root, text="Select Image", command=goto)
btn1.config(width=15)
btn1.grid(column=0, row=1, padx=10, pady=5)

filterLabel = Label(root, text="Filters :")
filterLabel.grid(column=1, row=1, padx=10, pady=5)

btn2 = Button(root, text="sharp", command=sharp)
btn2.grid(column=1, row=3, padx=10, pady=5)
btn2.config(width=15)

btn3 = Button(root, text="blur", command=blur)
btn3.grid(column=1, row=4, padx=10, pady=5)
btn3.config(width=15)

btn4 = Button(root, text="gray", command=gray)
btn4.grid(column=1, row=5, padx=10, pady=5)
btn4.config(width=15)

btn5 = Button(root, text="Laplacian", command=Laplacian)
btn5.grid(column=1, row=6, padx=10, pady=5)
btn5.config(width=15)

btn6 = Button(root, text="threshold", command=threshold)
btn6.grid(column=1, row=7, padx=10, pady=5)
btn6.config(width=15)

btn7 = Button(root, text="green", command=green)
btn7.grid(column=1, row=8, padx=10, pady=2)
btn7.config(width=15)

btn8 = Button(root, text="blue", command=blue)
btn8.grid(column=1, row=9, padx=10, pady=2)
btn8.config(width=15)

btn9 = Button(root, text="red", command=red)
btn9.grid(column=2, row=3, padx=10, pady=2)
btn9.config(width=15)

btn10 = Button(root, text="edge detection", command=edge)
btn10.grid(column=2, row=4, padx=10, pady=5)
btn10.config(width=15)

btn11 = Button(root, text="median", command=median)
btn11.grid(column=2, row=5, padx=10, pady=5)
btn11.config(width=15)

btn12 = Button(root, text="equl", command=equl)
btn12.grid(column=2, row=6, padx=10, pady=5)
btn12.config(width=15)

btn13 = Button(root, text="equl Color", command=equlColor)
btn13.grid(column=2, row=7, padx=10, pady=5)
btn13.config(width=15)

btn17 = Button(root, text="sum", command=sum)
btn17.grid(column=2, row=8, padx=10, pady=5)
btn17.config(width=15)

btn18 = Button(root, text="saturation", command=saturation2)
btn18.grid(column=2, row=9, padx=10, pady=5)
btn18.config(width=15)

root.mainloop()
