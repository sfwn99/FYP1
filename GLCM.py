import matplotlib.pyplot as plt
from skimage.feature import greycomatrix,greycoprops
from skimage import io
import cv2
import numpy as np
from scipy import stats
import matplotlib.patches as patches
import timeit
import xlsxwriter




def Average(lst):
    return sum(lst)/len(lst)
def convert(list):
    # Converting integer list to string list
    s = [str(i) for i in list]

    # Join list items using join()
    res = float("".join(s))

    return (res)

import pathlib


grade = ["A","B","C","D"]
for letter in range(4):

    filesnumber = 0
    for path in pathlib.Path("segmented/{}TOP".format(grade[letter])).iterdir():
        if path.is_file():
            filesnumber += 1


    #Saving to excel
    workbook = xlsxwriter.Workbook('Grade{}.xlsx'.format(grade[letter]))
    worksheet = workbook.add_worksheet()
    row = 1


    for imgnum in range(filesnumber):

        imgnum +=1
        image = io.imread('segmented/{}TOP/segmented_invert_{}-{}T.jpg'.format(grade[letter],grade[letter],imgnum))

        nsize = (71,115)
        #(y,x)
        y = 25
        x = 70
        region = image[y+2:y+nsize[0],x+1:x+nsize[1]]


        plt.figure()
        ax = plt.subplot(1,2,1)
        ax.imshow(image,cmap='gray')
        rect = patches.Rectangle((x,y),nsize[1],nsize[0],linewidth=0.5,edgecolor='r',facecolor="none")
        ax.add_patch(rect)
        ax= plt.subplot(1,2,2)
        ax.imshow(region,cmap='gray')
        plt.show()


        corr = []
        energy = []
        contrast = []
        ASM = []
        diss = []
        homogene=[]

        glcm = greycomatrix(region, distances=[1,3],angles=[0, np.pi/4, np.pi/2, 3*np.pi/4])
        corr.append(Average(Average(greycoprops(glcm,'correlation')[:,:])))
        energy.append(Average(Average(greycoprops(glcm,'energy')[:,:])))
        contrast.append(Average(Average(greycoprops(glcm,'contrast')[:,:])))
        ASM.append(Average(Average(greycoprops(glcm,'ASM')[:,:])))
        diss.append(Average(Average(greycoprops(glcm,'dissimilarity')[:,:])))
        homogene.append(Average(Average(greycoprops(glcm,'homogeneity')[:,:])))




        print('Correlation:',corr)
        print('Energy:',energy)
        print('Contrast:',contrast)
        print('ASM:',ASM)
        print('Dissimilarity:',diss)
        print('Homogeneity:',homogene)





        worksheet.write('A1', 'Image')
        worksheet.write('B1', 'Correlation')
        worksheet.write('C1', 'Energy')
        worksheet.write('D1', 'Contrast')
        worksheet.write('E1', 'ASM')
        worksheet.write('F1', 'Dissimilarity')
        worksheet.write('G1', 'Homogeneity')

        content =[imgnum,convert(corr),convert(energy),convert(contrast),convert(ASM),convert(diss),convert(homogene)]
        column = 0
        for item in content:
            # write operation perform

            worksheet.write(row, column,item)

            # incrementing the value of row by one
            # with each iterations.
            column += 1

        row += 1




    workbook.close()

