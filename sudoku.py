
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def get_predictions(digits,model):
    
    predictions = []
    length = len(digits)
    for i in range(length):
        if cv.countNonZero(digits[i]) == 0:
            predictions.append(0)
        else:
                    
                    pred = model.predict(digits[i].reshape(1,28,28,1))
                    predictions.append(pred.argmax())
                                 
    return predictions

def get_board(predictions):
    board = np.array(predictions)
    board = board.reshape(9,9)
    
    return board

    
def contourns(image):
    area = 0
    contours, hierarchy = cv.findContours(image,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    for con in contours:
        area = cv.contourArea(con)
    return area


def limpieza(digits,mask):
    digits_clean = []
    
    for i in range(len(digits)):
        aux = digits[i]
        aux = cv.bitwise_and(aux,mask)
        
        if contourns(aux) < 30:
            aux = np.zeros_like(aux)
        
        digits_clean.append(aux)
    return digits_clean


def getDigits(img):
    im = []
    iniY = 0
    iniX = 0
    
    for j in range(28,280,28):
        for i in range(28,280,28):
            im.append(img[iniY:j, iniX:i])
            if i < 253:
                iniX = i
        iniX = 0
        if j < 253:
            iniY = j
        else:
            iniY = 0
    return im

#%%

img = cv.imread("sudoku_real_2.jpeg",0)
img = cv.imread("sudoku.jpg",0)

#img = cv.imread("sudoku1.jpg",0)
img_binaria = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,57,5)

# %% Marco las esquinas dependiendo de la imagen se tienen que elegir a los puntos


srcCalib = np.array([[163, 207],
                    [807, 180],
                    [108, 806],
                    [875, 782]], dtype=float)
"""

srcCalib = np.array([[20, 318],
                    [837, 295],
                    [38, 1121],
                    [846, 1121]], dtype=float)

"""

dstCalib = np.meshgrid(np.arange(2) * 252.0 , np.arange(2) * 252.0)
dstCalib = np.reshape(dstCalib, (2,-1)).T


# ploteo imagenes binarizadas con sus corners

fig = plt.figure("comparaperspectivetransform", figsize=(12, 5))
fig.clf()
ax1 = plt.subplot(121)
ax1.imshow(img, cmap = "gray")
ax1.plot(*srcCalib.T, 'ob')

ax2 = plt.subplot(122)
ax2.plot(*dstCalib.T, 'ob')
ax2.axis('equal')
ax2.set_xlabel("X [mm]")
ax2.set_ylabel("Y [mm]")

plt.tight_layout()

# %% calcular la homografia entre ellos

H, mask = cv.findHomography(srcCalib, dstCalib)

print(H)
srcCalib = srcCalib.reshape((-1, 1, 2))
dstProj = cv.perspectiveTransform(srcCalib, H)
dstProj = dstProj.reshape((-1, 2))

ax2.plot(*dstProj.T, 'xr')

# %% ahora con la imagen

dsize = (252, 252)
warped = cv.warpPerspective(img, H, dsize)

ax2.imshow(warped, cmap = "gray")


img_ = cv.adaptiveThreshold(warped,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,57,5)


digits = getDigits(img_)
plt.figure()
for i in range(0,81):
    plt.subplot(9,9,i+1)
    plt.imshow(digits[i], cmap = "gray")
   
#mask = np.zeros((30,30),dtype='uint8')
#aa = cv.floodFill(digits[1],mask,(0,27),(0,0,0))



mask = np.zeros((28,28), dtype='uint8')
mask1 = np.zeros((28,28), dtype='uint8')
mask2 = np.zeros((28,28), dtype='uint8')
mask3 = np.zeros((28,28), dtype='uint8')

mask4 = cv.rectangle(mask, (0,0),(28,5), 255, -1 )
mask5 = cv.rectangle(mask1, (0,0),(5,28), 255, -1 )
mask6 = cv.rectangle(mask2, (0,23),(28,28), 255, -1 )
mask7 = cv.rectangle(mask3, (23,0),(28,28), 255, -1 )

mask8 = cv.bitwise_or(mask4,mask5)
mask8 = cv.bitwise_or(mask8,mask6)
mask8 = cv.bitwise_or(mask8,mask7)
mask8 = cv.bitwise_not(mask8)

digits = limpieza(digits,mask8)

plt.figure()
for i in range(0,81):
    plt.subplot(9,9,i+1)
    plt.imshow(digits[i], cmap = "gray")

    


predictions = get_predictions(digits,model)
board = get_board(predictions)   
solve(board)
printBoard(board)
