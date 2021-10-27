from tkinter import Button, Label, filedialog, messagebox
from PIL import Image
from PIL.ImageTk import PhotoImage
import cv2
from .Segmentation import Segmentation
from .Contours import GetContourImage

part = {
    "background": 0,
    "face": 1,
    "left_brow": 2,
    "right_brow": 3,
    "left_eye": 4,
    "right_eye": 5,
    "left_ear": 7,
    "right_ear": 8,
    "nose": 10,
    "upper_lip": 12,
    "lower_lip": 13,
    "neck": 14,
    "necklace": 15,
    "cloth": 16,
    "hair": 17,
}


class UI:
    def __init__(self, hwnd):
        self.Hwnd = hwnd
        self.ImageLabels = []
        self.Images = []
        self.OutputImage = Label(self.Hwnd)
        self.ImageCount = 0
        self.OpenImageBtn()
        self.SaveImageBtn()
        self.PrevImageBtn()
        self.NextImageBtn()
        


    def OpenImageBtn(self):
        Button(self.Hwnd, text="Open Image", 
        command=self.OpenImageFile).grid(row=0, column=0, columnspan=3, sticky="ew")
        
    def PrevImageBtn(self):
        Button(self.Hwnd, text="<<", command=self.PrevImage).grid(row=1, column=0, sticky="ns")
    def NextImageBtn(self):
        Button(self.Hwnd, text=">>", command=self.NextImage).grid(row=1, column=2, sticky="ns")
    def SaveImageBtn(self):
        Button(self.Hwnd, text="Save Image", command=self.SaveImage).grid(row=2, column=0, columnspan=3, sticky="nsew")
    def OpenImageFile(self):
        # Get filename
        filename = filedialog.askopenfilename(title="Open Image")

        # Check if filename not null
        if filename:
            # Check image list (Image format for save image) and image labels(PhotoImage format for output)
            # if not null then clear lists
            if self.Images and self.ImageLabels:
                self.Images.clear()
                self.ImageLabels.clear()

            # Read Image
            image = cv2.imread(filename)

            # Resize Image
            image = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
            
            # Image segmentation and extract contour from the given annotation
            # Segmentation return mask and annotation
            faceMaskColor = [[200, 0, 0]]
            facemask, faceanno = Segmentation(image, [part["face"]], faceMaskColor)
            facecontour = GetContourImage(faceanno, (255, 255, 255, 255))

            mouthMaskColor = [[0, 200, 0], [0, 200, 0]]
            mouthmask, mouthanno = Segmentation(image, [part["upper_lip"], part["lower_lip"]], mouthMaskColor)
            mouthcontour = GetContourImage(mouthanno, (255, 255, 255, 255))

            facemouthMaskColor = [[200, 0, 0], [0, 200, 0], [0, 200, 0]]
            facemouthMask, facemouthAnno = Segmentation(image, [part["face"], part["upper_lip"], part["lower_lip"]], facemouthMaskColor)
            
            # Create Images from Arrays
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            facemouthMask = Image.fromarray(facemouthMask)
            facemask = Image.fromarray(facemask)
            faceanno = Image.fromarray(faceanno)
            facecontour = Image.fromarray(facecontour).convert("RGB")
            mouthmask = Image.fromarray(mouthmask)
            mouthanno = Image.fromarray(mouthanno)
            mouthcontour = Image.fromarray(mouthcontour).convert("RGB")

            # Add Images to Image List
            self.Images.append(image)
            self.Images.append(facemouthMask)
            self.Images.append(facemask)
            self.Images.append(faceanno)
            self.Images.append(facecontour)
            self.Images.append(mouthmask)
            self.Images.append(mouthanno)
            self.Images.append(mouthcontour)

            # Create PhotoImages from Images
            image = PhotoImage(image)
            facemouthMask = PhotoImage(facemouthMask)
            facemask = PhotoImage(facemask)
            faceanno = PhotoImage(faceanno)
            facecontour = PhotoImage(facecontour)
            mouthmask = PhotoImage(mouthmask)
            mouthanno = PhotoImage(mouthanno)
            mouthcontour = PhotoImage(mouthcontour)

            # Add PhotoImages to Image Label List
            self.ImageLabels.append(image)
            self.ImageLabels.append(facemouthMask)
            self.ImageLabels.append(facemask)
            self.ImageLabels.append(faceanno)
            self.ImageLabels.append(facecontour)
            self.ImageLabels.append(mouthmask)
            self.ImageLabels.append(mouthanno)
            self.ImageLabels.append(mouthcontour)

            # Output Image with Created Image Labels
            self.OutputImage = Label(image=self.ImageLabels[0])
            self.OutputImage.grid(row=1, column=1)

    def NextImage(self):
        if self.ImageLabels and self.ImageCount + 1 < len(self.ImageLabels):       
            self.ImageCount = self.ImageCount + 1
            self.OutputImage.grid_forget()
            self.OutputImage = Label(image=self.ImageLabels[self.ImageCount])
            self.OutputImage.grid(row=1, column=1)

    def PrevImage(self):
        if self.ImageLabels and self.ImageCount - 1 > -1:       
            self.ImageCount = self.ImageCount - 1
            self.OutputImage.grid_forget()
            self.OutputImage = Label(image=self.ImageLabels[self.ImageCount])
            self.OutputImage.grid(row=1, column=1)
    
    def SaveImage(self):
        if self.ImageLabels:
            filedirectory = filedialog.asksaveasfile(mode="w", defaultextension=".jpg")
            if not filedirectory:
                return
            self.Images[self.ImageCount].save(filedirectory)
            

