import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import imutils
import easyocr
import csv

class LicensePlateRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("License Plate Recognition App")
        self.root.geometry("800x600")

        # Initialize car details
        self.cars = self.load_car_details()

        self.create_widgets()

    def create_widgets(self):
        # Create and set up the main frame
        self.main_frame = tk.Frame(self.root, bg="#F5F5F5")
        self.main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Create a label to display the processed image
        self.processed_image_label = tk.Label(self.main_frame, bg="white")
        self.processed_image_label.pack(pady=10)

        # Create labels for displaying license plate and car details
        self.license_plate_label = tk.Label(self.main_frame, text="License Plate: ", font=("Helvetica", 14), bg="#F5F5F5")
        self.license_plate_label.pack()

        self.car_details_label = tk.Label(self.main_frame, text="Car Details: ", font=("Helvetica", 14), bg="#F5F5F5")
        self.car_details_label.pack()

        # Create buttons for image capture and inputting start/end points
        self.capture_button = tk.Button(self.main_frame, text="Capture Image", command=self.capture_image,
                                        font=("Helvetica", 12), bg="#4CAF50", fg="white", padx=10, pady=5)
        self.capture_button.pack(side=tk.LEFT, padx=5)

        self.file_button = tk.Button(self.main_frame, text="Select Image File", command=self.select_image_file,
                                     font=("Helvetica", 12), bg="#FFC107", fg="black", padx=10, pady=5)
        self.file_button.pack(side=tk.LEFT, padx=5)

        # Exit button
        self.exit_button = tk.Button(self.main_frame, text="Exit", command=self.root.quit,
                                     font=("Helvetica", 12), bg="#E74C3C", fg="white", padx=10, pady=5)
        self.exit_button.pack(side=tk.LEFT, padx=5)

    def load_car_details(self):
        # Load car details from CSV file
        cars = []
        try:
            with open("Cars.csv", mode='r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    cars.append(row)
        except FileNotFoundError:
            messagebox.showerror("Error", "Car details CSV file not found!")
        except Exception as e:
            messagebox.showerror("Error", str(e))
        return cars

    def select_image_file(self):
        file_path = filedialog.askopenfilename(title="Select Image File",
                                                filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            img = cv2.imread(file_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Process the image and perform license plate recognition
            license_plate_text, car_details = self.process_image(gray)

            # Display the processed image
            processed_img = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(processed_img))
            self.processed_image_label.configure(image=photo)
            self.processed_image_label.image = photo

            # Display license plate information (if detected)
            self.license_plate_label.configure(text=f"License Plate: {license_plate_text or 'Not Detected'}")

            # Display car details (if detected)
            self.car_details_label.configure(text=f"Car Details: {car_details or 'Not Found'}")

    def capture_image(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open camera!")
            return
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image.")
            cap.release()
            return
        img = frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cap.release()

        # Process the image and perform license plate recognition
        license_plate_text, car_details = self.process_image(gray)

        # Display the processed image
        processed_img = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(processed_img))
        self.processed_image_label.configure(image=photo)
        self.processed_image_label.image = photo

        # Display license plate information (if detected)
        self.license_plate_label.configure(text=f"License Plate: {license_plate_text or 'Not Detected'}")

        # Display car details (if detected)
        self.car_details_label.configure(text=f"Car Details: {car_details or 'Not Found'}")

    def process_image(self, img):
        bfilter = cv2.bilateralFilter(img, 11, 17, 17)
        edged = cv2.Canny(bfilter, 30, 200)

        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        location = None

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break

        if location is None:
            return None, None  # No license plate found

        mask = np.zeros(img.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)

        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = img[x1:x2 + 1, y1:y2 + 1]

        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)

        try:
            text = result[0][-2]
        except IndexError:
            print("Sorry, Car number not detected!!")
            return None, None

        # Search for the car details in the cars list
        car_details = self.find_car_details(text)

        return text, car_details

    def find_car_details(self, license_plate):
        for car in self.cars:
            if car["Car_Number"] == license_plate:
                return f"Make: {car['Make']}, Model: {car['Model']}, Age: {car['Age_Of_Car']} years, Servicing: {car['Months_without_Servicing']} months"
        return "Car details not found."

if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateRecognitionApp(root)
    root.mainloop()
