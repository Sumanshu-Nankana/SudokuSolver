import cv2
import numpy as np
import operator
from imutils import contours
import imutils
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from solve_sudoku_grid3 import sudoku_solver, print_grid

# Load the model
new_model = tf.keras.models.load_model('../model/Digit_Recognizer.h5')

#To order predicted digit nested list 
def display_predList(predList):
    predicted_digits = []
    for i in range(len(predList)):
        for j in range(len(predList)):
            predicted_digits.append(predList[j][i])
    
    return predicted_digits

#Parameters for Warping the image
margin = 10
case = 28 + 2*margin
perspective_size = 9*case

flag = 0
ans = 0

frame = cv2.imread('../Images/sudoku.jpg')
p_frame = frame.copy()

#Process the frame to find contour
gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.imwrite('../Images/1_bgr_2_gray.jpg', gray)
gray=cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imwrite('../Images/2_gaussian_blur.jpg', gray)
thresh=cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)
cv2.imwrite('../Images/3_adaptive_threshold.jpg', thresh)

#Get all the contours in the frame
contours_, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour = None
maxArea = 0

#Find the largest contour(Sudoku Grid)
for c in contours_:
    area = cv2.contourArea(c)

    if area > 25000:
        peri = cv2.arcLength(c, True)
        polygon = cv2.approxPolyDP(c, 0.01*peri, True)

        if area>maxArea and len(polygon)==4:
            contour = polygon
            maxArea = area

#Draw the contour and extract Sudoku Grid
if contour is not None:
    cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)
    points = np.vstack(contour).squeeze()
    points = sorted(points, key=operator.itemgetter(1))

    if points[0][0]<points[1][0]:
        if points[3][0]<points[2][0]:
            pts1 = np.float32([points[0], points[1], points[3], points[2]])
        else:
            pts1 = np.float32([points[0], points[1], points[2], points[3]])
    else:
        if points[3][0]<points[2][0]:
            pts1 = np.float32([points[1], points[0], points[3], points[2]])
        else:
            pts1 = np.float32([points[1], points[0], points[2], points[3]])

    pts2 = np.float32([[0, 0], [perspective_size, 0], [0, perspective_size], [perspective_size, perspective_size]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    perspective_window =cv2.warpPerspective(p_frame, matrix, (perspective_size, perspective_size))
    cv2.imwrite('../Images/4_warp_perspective.jpg', perspective_window)
    result = perspective_window.copy()

    #Process the extracted Sudoku Grid
    p_window = cv2.cvtColor(perspective_window, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('../Images/5_bgr_2_gray.jpg', p_window)

    p_window = cv2.GaussianBlur(p_window, (5, 5), 0)
    cv2.imwrite('../Images/6_gaussian_blur.jpg', p_window)

    p_window = cv2.adaptiveThreshold(p_window, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)
    cv2.imwrite('../Images/7_adaptive_thresh.jpg', p_window)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    p_window = cv2.morphologyEx(p_window, cv2.MORPH_CLOSE, vertical_kernel)
    lines = cv2.HoughLinesP(p_window, 1, np.pi/180, 120, minLineLength=40, maxLineGap=10)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(perspective_window, (x1, y1), (x2, y2), (0, 255, 0), 2)

    #Invert the grid for digit recognition
    invert = 255 - p_window
    cv2.imwrite('../Images/8_invert.jpg', invert)

    invert_window = invert.copy()

    invert_window = invert_window /255

    i = 0

    #Check if the answer has been already predicted or not
    #If not predict the answer
    #Else only get the cell regions
    if flag != 1:
        predicted_digits = []
        pixels_sum = []

    #To get individual cells
    for y in range(9):
        predicted_line = []
        for x in range(9):
                    y2min = y*case+margin
                    y2max = (y+1)*case-margin
                    x2min = x*case+margin
                    x2max = (x+1)*case-margin

                    #Obtained Cell
                    image = invert_window[y2min:y2max, x2min:x2max]

                    #Process the cell to feed it into model
                    img = cv2.resize(image,(28,28))
                    img = img.reshape((1,28,28,1))

                    #Get sum of all the pixels in the cell
                    #If sum value is large it means the cell is blank
                    pixel_sum = np.sum(img)
                    pixels_sum.append(pixel_sum)

                    #Predict the digit in the cell
                    pred = new_model.predict(img)
                    predicted_digit = pred.argmax()

                    #For blank cells set predicted digit to 0
                    if pixel_sum > 750.0:
                        predicted_digit = 0

                    predicted_line.append(predicted_digit)                        

                    #If we already have predicted result, display it on window
                    if flag == 1:
                        ans = 1
                        x_pos = int((x2min + x2max)/ 2)+10
                        y_pos = int((y2min + y2max)/ 2)-5
                        image = cv2.putText(result, str(pred_digits[i]), (y_pos, x_pos), cv2.FONT_HERSHEY_SIMPLEX,  
                        1, (255, 0, 0), 2, cv2.LINE_AA)

                    i = i + 1

        #Get predicted digit list
        if flag != 1:
            predicted_digits.append(predicted_line)
    
    print('-----Predicted Digits------')
    print_grid(predicted_digits)
    print('---------------------------')
    #Get solved Sudoku
    flag, grid = sudoku_solver(predicted_digits)
    if flag:
        print('--------Solution----------')
        print_grid(grid)
    else:
        print('Unable to Solve Sudoku, Something went wrong')
    
cv2.destroyAllWindows()