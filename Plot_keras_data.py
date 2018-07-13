from matplotlib import pyplot as pl
import pylab as pla
import numpy as np
import EDA_myAPI_ML as eda
import math

#
# load the data
#JPG to numpy array to be use in Tensorflow/Keras
dir_oasis = "D:\ESISAR\Okayama_University\Python\Image_Dataset\oasis\\"
dir_gaped = r"D:\ESISAR\Okayama_University\Python\Image_Dataset\GAPED_2\GAPED\GAPED\\"


csv_oasis = "D:\ESISAR\Okayama_University\Python\Image_Dataset\oasis\OASIS.csv"
csv_gaped = [dir_gaped+"A.csv",dir_gaped+"H.csv",dir_gaped+"N.csv",
                    dir_gaped+"P.csv",dir_gaped+"Sn.csv",dir_gaped+"Sp.csv"]

# load oasis 
Name_oasis,Valence_mean_oasis,Valence_SD_oasis,Valence_N_oasis,Arousal_mean_oasis,_oasisArousal_SD_oasis,Arousal_N_oasis = [],[],[],[],[],[],[]
Name_oasis,Valence_mean_oasis,Valence_SD_oasis,Valence_N_oasis,Arousal_mean_oasis,_oasisArousal_SD_oasis,Arousal_N_oasis = eda.OpenCsvFile(csv_oasis)

# load gaped 
Name_gaped,Valence_mean_gaped,Arousal_mean_gaped =[],[],[]
for i in range(len(csv_gaped)):
    Name_gapedtmp,Valence_mean_gapedtmp,Arousal_mean_gapedtmp = [],[],[]
    Name_gapedtmp,Valence_mean_gapedtmp,Arousal_mean_gapedtmp = eda.OpenCsvFile_Gaped(csv_gaped[i])
    Name_gaped.extend(Name_gapedtmp)
    Valence_mean_gaped.extend(Valence_mean_gapedtmp)
    Arousal_mean_gaped.extend(Arousal_mean_gapedtmp)
    

tm_gaped = pla.arange(1., len(Valence_mean_gaped)+1.)
tm_oasis = pla.arange(1., len(Valence_mean_oasis)+1.)

fig = pl.figure()
pl.subplot(3, 1, 1)
pl.hist(Valence_mean_gaped, bins = 7)
pl.title("Raw Valence")

eda.normalizerange(Valence_mean_gaped,1,7)
pl.subplot(3, 1, 2)
pl.hist(Valence_mean_gaped, bins = 7)
pl.title("Normalize Valence")

eda.rescaling(Valence_mean_gaped,1,7)
pl.subplot(3, 1, 3)
pl.hist(Valence_mean_gaped, bins = 7)
pl.title("Rescale Valence")
pl.show()
fig = pl.figure()
#pl.subplot(2, 1, 1)
eda.normalizerange(Valence_mean_gaped,1,7)
eda.rescaling(Valence_mean_gaped,1,7)
eda.normalizerange(Arousal_mean_gaped,1,7)
eda.rescaling(Arousal_mean_gaped,1,7)
pl.scatter(Valence_mean_gaped, Arousal_mean_gaped, marker = '+')
axes = pl.gca()
axes.set_xlim([-3.5,3.5])
axes.set_ylim([-3.5,3.5])
pl.title("Gaped scatter plot")
#pl.plot(Valence_mean_gaped,Arousal_mean_gaped)
pl.show()


fig = pl.figure()
#pl.subplot(2, 1, 1)
eda.rescaling(Valence_mean_oasis,1,7)
eda.rescaling(Arousal_mean_oasis,1,7)
pl.scatter(Valence_mean_oasis, Arousal_mean_oasis, marker = '+')
pl.title("Oasis scatter plot")
axes = pl.gca()
axes.set_xlim([-3.5,3.5])
axes.set_ylim([-3.5,3.5])
#pl.plot(Valence_mean_gaped,Arousal_mean_gaped)
pl.show()
#
#cmap = pl.cm.RdYlGn
#norm = pl.Normalize(1,4)
#
#fig,ax = pl.subplots()
#sc_oasis = pl.scatter(Valence_mean_oasis, Arousal_mean_oasis, marker = '+', label ='OASIS')
#annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
#                    bbox=dict(boxstyle="round", fc="w"),
#                    arrowprops=dict(arrowstyle="->"))
#annot.set_visible(False)
#pl.title("Oasis scatter plot")
#def update_annot(ind):
#
#    pos = sc_oasis.get_offsets()[ind["ind"][0]]
#    annot.xy = pos
#    text = "{}".format( 
#                           " ".join([Name_oasis[n] for n in ind["ind"]]))
#    annot.set_text(text)
##    annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
#    annot.get_bbox_patch().set_alpha(0.4)
#
#
#def hover(event):
#    vis = annot.get_visible()
#    if event.inaxes == ax:
#        cont, ind = sc_oasis.contains(event)
#        if cont:
#            update_annot(ind)
#            annot.set_visible(True)
#            fig.canvas.draw_idle()
#        else:
#            if vis:
#                annot.set_visible(False)
#                fig.canvas.draw_idle()
#
#fig.canvas.mpl_connect("motion_notify_event", hover)
##pl.show()
#
#fig2,ax2 = pl.subplots()
#sc_gaped = pl.scatter(Valence_mean_gaped, Arousal_mean_gaped, marker = '+', label ='gaped')
#annot2 = ax2.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
#                    bbox=dict(boxstyle="round", fc="w"),
#                    arrowprops=dict(arrowstyle="->"))
#annot2.set_visible(False)
#pl.title("Gaped scatter plot")
#def update_annot22(ind):
#
#    pos = sc_gaped.get_offsets()[ind["ind"][0]]
#    annot2.xy = pos
#    text = "{}".format( 
#                           " ".join([Name_gaped[n] for n in ind["ind"]]))
#    annot2.set_text(text)
##    annot2.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
#    annot2.get_bbox_patch().set_alpha(0.4)
#
#
#def hover2(event):
#    vis = annot2.get_visible()
#    if event.inaxes == ax2:
#        cont, ind = sc_gaped.contains(event)
#        if cont:
#            update_annot22(ind)
#            annot2.set_visible(True)
#            fig2.canvas.draw_idle()
#        else:
#            if vis:
#                annot2.set_visible(False)
#                fig2.canvas.draw_idle()
#
#fig2.canvas.mpl_connect("motion_notify_event", hover2)
#pl.show()



###############################################################################
#Q_Gaped = [[] for i in range(16)]
#for i in range(len(Name_gaped)):
#    Lv = Valence_mean_gaped[i]
#    La = Arousal_mean_gaped[i]
#    r = math.sqrt(Lv**2 + La**2)
#    Theta = math.atan2(Lv,La) + math.pi
#    for j in range(8):
#        if ((j*math.pi)/4 <= Theta <= ((j+1)*math.pi)/4):
#            if (0<=r<=1.5):
#                Q_Gaped[j].append(Name_gaped[i])
#            else:
#                Q_Gaped[j+8].append(Name_gaped[i])
#            
#H_Gaped = [len(q) for q in Q_Gaped]
#
## Make a fake dataset:
#bars = ('0 - pi/4', 'pi/4 - pi/2', 'pi/2 - 3pi/4','3pi/4 - pi','pi - 5pi/4', '5pi/4 - 3pi/2', '3pi/2 - 7pi/4', '7pi/4 - 2pi',
#        '0 - pi/4', 'pi/4 - pi/2', 'pi/2 - 3pi/4','3pi/4 - pi','pi - 5pi/4', '5pi/4 - 3pi/2', '3pi/2 - 7pi/4', '7pi/4 - 2pi')
#
#fig = pl.figure()
#y_pos = np.arange(len(bars))
## Create bars
#pl.bar(y_pos, H_Gaped)
#
#pl.title('Gaped')
# # Create names on the x-axis
#pl.xticks(y_pos, bars)
# # Show graphic
#pl.show()
#
#
#
#Q_Oasis = [[] for i in range(16)]
#for i in range(len(Name_oasis)):
#    Lv = Valence_mean_oasis[i]
#    La = Arousal_mean_oasis[i]
#    r = math.sqrt(Lv**2 + La**2)
#    Theta = math.atan2(Lv,La) + math.pi
#    for j in range(8):
#        if ((j*math.pi)/4 <= Theta <= ((j+1)*math.pi)/4):
#            if (0<=r<=1.5):
#                Q_Oasis[j].append(Name_oasis[i])
#            else:
#                Q_Oasis[j+8].append(Name_oasis[i])
#            
#H_Oasis = [len(q) for q in Q_Oasis]
#
#
#bars = ('0 - pi/4', 'pi/4 - pi/2', 'pi/2 - 3pi/4','3pi/4 - pi','pi - 5pi/4', '5pi/4 - 3pi/2', '3pi/2 - 7pi/4', '7pi/4 - 2pi',
#        '0 - pi/4', 'pi/4 - pi/2', 'pi/2 - 3pi/4','3pi/4 - pi','pi - 5pi/4', '5pi/4 - 3pi/2', '3pi/2 - 7pi/4', '7pi/4 - 2pi')
#
#fig = pl.figure()
#y_pos = np.arange(len(bars))
## Create bars
#pl.bar(y_pos, H_Oasis)
# # Create names on the x-axis
#pl.xticks(y_pos, bars)
#pl.title('Oasis')
# # Show graphic
#pl.show()

    