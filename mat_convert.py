import scipy.io as sio
import os
### convert .mat to python file ###
load=sio.loadmat("/Users/len/Desktop/HRI/YouCook/Annotations/Object_Tracks/0005_whisk_1.mat")

print (load.keys())
label=load['Lab1'][0][0][0][0]
print (label)

### object list ###
''' 
object_dict={
"apple": 0, "blender": 1, "bowl": 2, "bread": 3, "brocolli": 4, "brush": 5, "butter": 6, "carrot": 7,
"chicken": 8, "chocolate": 9, "corn": 10, "creamcheese": 11, "croutons": 12, "cucumber": 13,
"cup": 14, "doughnut": 15, "egg": 16, "fish": 17, "flour": 18, "fork": 19, "hen": 20, "jelly": 21, "knife": 22, "lemon": 23,
"lettuce": 24, "meat": 25, "milk": 26, "mustard": 27, "oil": 28, "onion": 29, "pan": 30, "peanutbutter": 31,
"pepper": 32, "pitcher": 33, "plate": 34, "pot": 35, "salmon": 36, "salt": 37, "spatula": 38, "spoon": 39,
"spreader": 40, "steak": 41, "sugar": 42, "tomato": 43, "tongs": 44, "turkey": 45, "whisk": 46, "yogurt": 47
}
'''


video_name = load['LabFile'][0][:4] # extract the video name.
num_frame = load['FinalLocations'].shape[1]-1
path = "/Users/len/Desktop/annotation"
cnt  = 4343
for i in range(num_frame):
    f=open(path + "/" "{:0>6d}".format(cnt) + ".txt","a") 
    cnt += 1
    coor = load["FinalLocations"][:4,i]
    if not any(coor):
        continue
    for j in range(4):
        coor = load["FinalLocations"][j,i]
        coor = int(coor)
        f.write(str(coor)+" ")
    f.write(str(label) + "\n")
    f.close()


