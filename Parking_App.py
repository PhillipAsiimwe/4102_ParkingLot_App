from detecto import core,utils
from Map_out_lot import Draw_ParkingSpaces
import cv2
import os.path
from torchvision import transforms
import numpy as np
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-v","--Video",required=True,help="Name of the video to be detected add format, ie.mp4")
args = vars(ap.parse_args())
red = (0,0,255)
blue = (255,0,0)
white = (255,255,255)
parkingspace_Coords = []
videofile = args["Video"]
wheredot = videofile.find('.')#finds the extenstion
removemp4 = videofile[:-(wheredot)]#removes the extenstion for file saving
outputfile = removemp4+'output.avi'
if not (os.path.isfile(videofile)):
    print("[INFO] Could not find file Exiting program")
    exit()
filenme = 'ParkingLotData_'+removemp4
if os.path.isfile(filenme+'.npy'):
    print('[INFO] File exists *** importing parkinglot coordinates')
    parkingspace_Coords = np.load(filenme+'.npy')
else:
    print('[INFO] File does not exist please outline parking spaces')
    gen = Draw_ParkingSpaces(videofile,filenme,parkingspace_Coords)
    gen.generate()

print('[INFO] Loading Model')
model = core.Model.load('carmodel.pth', ['car'])
print('[INFO] Model loaded')

subtractor = cv2.createBackgroundSubtractorMOG2(9000,700)
def compare(rect1,rect2):#Compare function for rectangles to see if they overlap
    RECL1x = rect1[0]
    RECL1y = rect1[1]
    RECR1x = rect1[2]
    RECR1y = rect1[3]
    RECL2x = rect2[0][0]
    RECL2y = rect2[0][1]
    RECR2x = rect2[1][0]
    RECR2y = rect2[1][1]
    if (RECL1x > RECR2x or RECL2x > RECR1x):
        return False
    if (RECL1y > RECR2y or RECL2y > RECR1y):
        return False
    return True
def comparesp(rect1,rect2):#Compare function for rectangles to see if they overlap
    RECL1x = rect1[0]
    RECL1y = rect1[1]
    RECR1x = rect1[2]
    RECR1y = rect1[3]
    RECL2x = rect2[0][0]
    RECL2y = rect2[0][1]
    RECR2x = rect2[2][0]
    RECR2y = rect2[2][1]
    if (RECL1x > RECR2x or RECL2x > RECR1x):
        return False
    if (RECL1y > RECR2y or RECL2y > RECR1y):
        return False
    return True
def procesVideo(model, input_file, output_file, fps=30):

    video = cv2.VideoCapture(input_file)

    # Video frame dimensions
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Scale down frames when passing into model for faster speeds
    scaled_size = 800
    scale_down_factor = min(frame_height, frame_width) / scaled_size

    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_width, frame_height))

    # Transform to apply on individual frames of the video
    transform_frame = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(scaled_size),
        transforms.ToTensor(),
        utils.normalize_transform(),
    ])

    while True:
        carMovin = []#new array to keep track of all the moving cars
        ret, frame = video.read() #Read frame
        if not ret:#if frame not read exits loop
            break
        frameArea = frame_height*frame_width #frame area to be used on mask to track objects big enough
        mask = subtractor.apply(frame) #background subtractor method
        newmask = cv2.medianBlur(mask,3) #remove salt and paper noise
        (contour, heir) = cv2.findContours(newmask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#Find the contours of moving objects
        for c in contour: #go through list of contours and find the ones big enough
            (x, y, w, h) = cv2.boundingRect(c)
            area = ((x + w) - x + 1) * ((y + h) - y + 1)
            if (area > (frameArea * 0.001)):
                carMovin.append([(x, y), (x + w, y + h)]) #put into the array objects big enough
        transformed_frame = transform_frame(frame) #makes the frame smaller to be processed by detecto function
        labels,boxs,scores = model.predict(transformed_frame) #once predictions have been made positive cars are saved as tensorflow arrays
        boxs *=scale_down_factor #applys the scale down factor so the mapping is correct

        for box in boxs:#get every box from the predictions
            colourcar = False #keep track of car that is moving
            box =np.array(box.tolist(),dtype=int) #convert from tensorflow array to np array
            for cars in carMovin:# go through every moving car found
                if (compare(box,cars)): #first compare function used to find rectangles that overlap
                    colourcar = True #if true the box should be coloured
            if (colourcar):
                cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),red,2)#colour box red

        for spots in parkingspace_Coords:#go through all the parking spots saved
            colourspot = False #keep track of parking spot that has a car in it
            for box in boxs: #Find all the cars that are parked
                box = np.array(box.tolist(), dtype=int) #convert from tensorflow array to np array
                if comparesp(box, spots): # second compare function for parking spots and car objects
                    colourspot = True
            if (colourspot):
                cv2.rectangle(frame, (spots[0][0],spots[0][1]), (spots[2][0], spots[2][1]), blue, 2)#parking lot with car
            else:
                cv2.rectangle(frame, (spots[0][0],spots[0][1]), (spots[2][0], spots[2][1]), white, 2)#parking lot without car
        out.write(frame)#writes frame to file
        # If the 'q' key is pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    # When finished, release the video capture and writer objects
    video.release()
    out.release()
    # Close all the frames
    cv2.destroyAllWindows()
print('[INFO] Processing Video')
procesVideo(model,videofile,outputfile)
print('[INFO] Video processed')
