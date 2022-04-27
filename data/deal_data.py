import os
from shutil import copy2

def cut_datasets(saveTrainPATH, saveTestPATH, saveCVPATH, category):
    for j in range(len(category)):
        PATH = "./data/archive/"+  category[j]
        #子文件夹
        for childPATH in os.listdir(PATH):
            #子文件夹路径
            NewChildPATH = PATH + '/'+ str(childPATH)
            print(NewChildPATH)
            trainfiles = os.listdir(NewChildPATH)
            num_image = len(trainfiles)
            
            # print( NewChildPATH + "   \t num_image: " + str(num_image) )
            
            index_list = list(range(num_image))
            # print(index_list)
            # random.shuffle(index_list)
            num = 0
            
            #保存trian的路径-----------------------------
            trainDir = saveTrainPATH + '/' + category[j]
            #先判断是否存在这个文件夹
            if not os.path.exists(trainDir):                   
                os.mkdir(str(trainDir))
                
            
            childTrainDir = trainDir + '/' + str(childPATH)
            #判断子子文件夹是否存在,若不存在则创建(老套娃了~~)
            if not os.path.exists(childTrainDir):
                os.mkdir(str(childTrainDir))
                
            
            #保存test的路径---------------------------------
            testDir = saveTestPATH + '/' + category[j]   
            if not os.path.exists(testDir):
                os.mkdir(str(testDir))
                
                
            childTestDir = testDir + '/' + str(childPATH)
            #判断子子文件夹是否存在,若不存在则创建
            if not os.path.exists(childTestDir):
                os.mkdir(str(childTestDir))

            #保存cv的路径---------------------------------
            cvDIR = saveCVPATH + '/' + '/' + category[j]   
            if not os.path.exists(cvDIR):
                os.mkdir(str(cvDIR))
                
                
            childCVDir = cvDIR + '/' + str(childPATH)
            #判断子子文件夹是否存在,若不存在则创建
            if not os.path.exists(childCVDir):
                os.mkdir(str(childCVDir))
                
            for i in index_list:
                fileName = os.path.join(NewChildPATH, trainfiles[i])
                if num < num_image * 0.6:
                    copy2(fileName, childTrainDir)  #复制过去,不改变原来目录的图片
                elif num < num_image * 0.9:
                    copy2(fileName, childTestDir)
                else:
                    copy2(fileName,childCVDir)
                num += 1
               
            
            print(trainDir,'\n',testDir )
if __name__ == '__main__':
    category = ["raw-img"]
    saveTrainPATH = "./data/cif10/train"
    saveTestPATH ="./data/cif10/test"
    saveCVPATH = "./data/cif10/cv"
    cut_datasets(saveTrainPATH, saveTestPATH, saveCVPATH, category)