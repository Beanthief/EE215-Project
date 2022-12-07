from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import cv2

class OpenCVGUI:
    def __init__(self):
        self.mainwindow = Tk()
        self.mainwindow.title("OpenCV GUI")
        self.mainwindow.geometry("105x170")
        self.mainwindow.resizable(0, 0)

        # create a grid of buttons
        self.buttonBrowse = Button(self.mainwindow, text="Browse Files", command=self.get_image)
        self.buttonBrowse.grid(row=0, column=0, padx=10, pady=4)
        self.buttonDisplayImage = Button(self.mainwindow, text="Display Image", command=self.display_image)
        self.buttonDisplayImage.grid(row=1, column=0, padx=10, pady=4)
        self.buttonFindFaces = Button(self.mainwindow, text="Find Faces", command=self.find_faces)
        self.buttonFindFaces.grid(row=2, column=0, padx=10, pady=4)
        self.buttonFindPeople = Button(self.mainwindow, text="Find People", command=self.find_people)
        self.buttonFindPeople.grid(row=3, column=0, padx=10, pady=4)
        self.buttonCamera = Button(self.mainwindow, text="Camera", command=self.find_video_faces)
        self.buttonCamera.grid(row=4, column=0, padx=10, pady=4)

        self.mainwindow.mainloop()

    def get_image(self):
        self.path = filedialog.askopenfilename()
        self.image = cv2.imread(self.path)
    
    def validate_image(self):
        if self.image is None:
            messagebox.showerror("Error", "No image loaded")
            return False
        return True

    def display_image(self):
        # Check if image is loaded
        if self.validate_image() == False:
            return

        # Create a copy image to display within screen bounds
        height = self.mainwindow.winfo_screenheight() - 100
        ratio = height / self.image.shape[0]
        width = int(self.image.shape[1] * ratio)
        dim = (width, height)
        displayImage = cv2.resize(self.image, dim, interpolation = cv2.INTER_AREA)

        # Display image and close when any key is pressed
        cv2.imshow("Image", displayImage)
        if cv2.waitKey(0) & 0xFF == ord('s'):
            cv2.imwrite("result.png", self.image)
            cv2.destroyAllWindows()

    def find_faces(self):
        # Check if image is loaded
        if self.validate_image() == False:
            return

        # Import the haar cascade
        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        # Run the cascade on the image
        faces = faceCascade.detectMultiScale(self.image, minNeighbors=10)

        # Draw rectangles around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(self.image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Show the image
        self.display_image()

    def find_people(self):
        # Check if image is loaded
        if self.validate_image() == False:
            return

        # Create a grayscale copy of the image for HOG
        graysample = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Set up the detector with default parameters and trained model
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Detect people in the image
        (rects, _) = hog.detectMultiScale(graysample, winStride=(8, 8), padding=(18, 18), scale=1.07)

        # Draw a rectangle around the people
        for (x, y, w, h) in rects:
            # Remove any rectangles that are too small
            if w > 100 and h > 200:
                cv2.rectangle(self.image, (x + 40, y + 35), (x + w - 40, y + h - 35), (0, 0, 255), 2)

        # Show the image
        self.display_image()
    
    def find_video_faces(self):
        # Get the input from camera and run the cascade on the image
        video_capture = cv2.VideoCapture(0)

        # Import the haar cascade
        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        
        while True:
            # Capture frame-by-frame
            _, frame = video_capture.read()

            # Run the cascade on the image
            faces = faceCascade.detectMultiScale(frame, minNeighbors=10)

            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow('Video', frame)

            # Close when q is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()

OpenCVGUI()