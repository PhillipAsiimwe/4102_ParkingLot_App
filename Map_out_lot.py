import cv2
import numpy as np

class Draw_ParkingSpaces:
    resetkey = ord('r') #press r to reset image if a mistake was made
    quitkey = ord('q') #press q to quit from mapping
    red = (0,0,255)
    white = (255,255,255)
    
    def __init__(self,image,out,coords):
        self.coord = coords #output coordinates to an array
        self.out = out # output name of file to save coordinates
        self.vid = cv2.VideoCapture(image) #initalise video capture for the program
        self.clicks = 0 #keeps track of the clicks
        self.id = 0 # keeps track of the number of recorded parking spots
        self.coordinates = [] #empty array to populate the parking lots
        self.windowname = "Draw Parking spaces" # window name
        cv2.namedWindow(self.windowname,cv2.WINDOW_GUI_EXPANDED)
        cv2.setMouseCallback(self.windowname,self.mouseEvent)

    def generate(self):
        _,image = self.vid.read() #load the first frame of the image
        copy = image.copy() #makes a copy
        self.image = copy #load image into class memory
        while True:
            cv2.imshow(self.windowname,copy) #display image
            key = cv2.waitKey(0)
            if key == Draw_ParkingSpaces.resetkey:
                self.image = self.image.copy() #resets image
            elif key == Draw_ParkingSpaces.quitkey:
                break
        cv2.destroyWindow(self.windowname) #destroys image when done
        saveCoordinates = np.array(self.coord) #collect the coordinates
        np.save(self.out,saveCoordinates) #saves coordinates to file

    def mouseEvent(self,event,x,y,flags,params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.coordinates.append((x,y)) #adds the click to the current array
            self.clicks +=1 # keeps track of the clicks

            if self.clicks >=4: #on 4 clicks the box has been created
                cv2.line(self.image,self.coordinates[2],self.coordinates[3],Draw_ParkingSpaces.red,1)#draw line of the box
                cv2.line(self.image,self.coordinates[3],self.coordinates[0],Draw_ParkingSpaces.red,1) #draw line of the box
                self.clicks = 0 #resets clicks
                coordin = np.array(self.coordinates) #adds complete coordinate to array
                self.coord.append(coordin) #add array of coordinates of one box to the array keeping track of all the boxes.
                Draw_ParkingSpaces.drawcontour(self,self.image,coordin,str(self.id +1 ),Draw_ParkingSpaces.white,Draw_ParkingSpaces.red) #draws contour on image to be visible for visual response

                for i in range(0,4):#loops to reset the array
                    self.coordinates.pop()
                self.id+=1 #increment id number
            elif self.clicks > 1: # if after second click a line can be drawn
                cv2.line(self.image,self.coordinates[-2],self.coordinates[-1],Draw_ParkingSpaces.red,1) #first line to be shown to user
        cv2.imshow(self.windowname,self.image) #shows the current image

    def drawcontour(self,image,coordinates,label,font_col,border,line = 1 , font = cv2.FONT_HERSHEY_SIMPLEX,fontscale = 0.5): #function to draw image and show id
        cv2.drawContours(image,[coordinates],contourIdx=-1,color=border,thickness=2,lineType=cv2.LINE_8)
        moms = cv2.moments(coordinates)
        center = (int(moms["m10"]/moms["m00"])-3,
                  int(moms["m01"]/moms["m00"])+3) #gets the center of the box using moments
        cv2.putText(image,label,center,font,fontscale,font_col,line,cv2.LINE_AA) #puts image tex in the middle of the box 
