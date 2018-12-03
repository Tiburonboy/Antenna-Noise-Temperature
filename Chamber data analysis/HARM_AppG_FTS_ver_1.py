"""
Python program to process HARM FTS antenna data for FTSR App G
by Tony Cirineo
Date: Nov 25 , 2018 code copied from HARM_FTS_ver_3.py and modified
Name: HARM AppG TM ver 1
Synopsis: This program will process HARM GPS data.

Description: 

This script will process HARM antenna pattern data.  Data 
is collected in the chamber by 


 need to fix theta and phi <<<<<-------

This script will read the excel file and process the data.  

The FTS antenna has two ports (SMA connectors).  Internally to the antenna 
assembly there are two antenna elements and a combiner/splitter network 
that allows two receivers to be connected to the FTS antenna.  The 
ports are referred to as front and rear and are not labeled 
since they are interchangeable.  

The FTS antenna patterns are measured at three frequencies 424, 425 and 
426 MHz, which are the frequencies for the low end of the band, center 
frequency and the high end of the band.  

Full antenna patterns (4*pi steradian, with a 2 degree step in theta and phi) 
are collected from the front antenna port at each frequency.  A roll antenna 
pattern (2 degree step in phi, theta = 90 degrees) is measured from the rear 
antenna port at each frequency.  This verifies that both antenna ports are 
functional, while reducing data collection time.   

The antenna shall have the following gains:
(a) Roll: The antenna shall have a gain of not less than -23 dBi, over 95% 
of the roll plane (theta = 90 degrees, phi varied).
(b) Nose:  The antenna shall have a gain of not less than -13 dBi over 95% 
of the angles measured at the nose (theta = 0 degrees, phi varied).
(c) Tail:  The antenna shall have a gain of not less than -15 dBi over 95% 
of the angles measured at the tail (theta = 180 degrees, phi varied).
(d) Sphere:  The antenna shall have a gain of not less than -17 dBi
over 95% of the radiation sphere.

The 95% rule for nose, tail and roll planes is coded to not consider the
lowest gains for 5% of the angles when determining the minimum gain in
these planes. 

The 95% of the radiation sphere rule is coded to not consider the lowest
gains which are associated with 5% of the surface of the radiation sphere.



FTS antenna spec...

The angle phi is taken to be zero degrees at the top center of the missile (umbilical location) and measured clockwise as viewed from the tail of the missile.  The angle theta is zero at the missile nose.  See IRIG 253-93, Figure 3-4, for a diagram of the aerodynamic vehicle coordinate system.  

Full antenna patterns (4*pi steradian, with a 2 degree step in theta and phi) 
are collected from the front antenna port at each frequency.  

A roll antenna 
pattern (2 degree step in phi, theta = 90 degrees) is measured from the rear 
antenna port at each frequency.     


Requires: Python version 3 or higher
Author(s): Tony Cirineo
Revision History
11/25/2018: code copied from HARM_FTS_ver_3.py and modified.
"""
# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
# %%
# hard code the path to working directory
os.chdir(r'C:\Users\Jim\Documents\Python SciPy\Antenna Chamber\HARM\App G data\FTS\patterns')
  
xl = pd.ExcelFile('HarmFTS-SN25.xlsx')
dfs = {sheet_name: xl.parse(sheet_name) for sheet_name in xl.sheet_names}

# read data frame, H & V polarization, Front port, save max of Horz or Vert
Mf_f424 = np.maximum(dfs['Horz 424'].values,dfs['Vert 424'].values)
Mf_f425 = np.maximum(dfs['Horz 425'].values,dfs['Vert 425'].values)
Mf_f426 = np.maximum(dfs['Horz 426'].values,dfs['Vert 426'].values)

# read data frame, H & V polarization ROLL CUT, Rear port, save max of Horz or Vert
Mr_f424 = np.maximum(dfs['Roll 424'].values[:,0],dfs['Roll 424'].values[:,1])
Mr_f425 = np.maximum(dfs['Roll 425'].values[:,0],dfs['Roll 425'].values[:,1])
Mr_f426 = np.maximum(dfs['Roll 426'].values[:,0],dfs['Roll 426'].values[:,1])


# %%

# change folder for storing plots
os.chdir(r'C:\Users\Jim\Documents\Python SciPy\Antenna Chamber\HARM\App G data\FTS') 

'''
plot full radiation pattern
'''
# fix phi and theta, are they correct?
phi = np.linspace(0,360,181,endpoint=True)
theta = np.linspace(0,180,91,endpoint=True)
X, Y = np.meshgrid(theta,phi)

fig, ax = plt.subplots()
ax.set_title('TM radiation pattern, 230 MHz')
ax.set_xlabel('phi?')
ax.set_ylabel('theta?')

cs = ax.contourf(X, Y, Mf_f425)
cbar = fig.colorbar(cs)
cbar.ax.set_ylabel('amplitude, dBi')

plt.savefig('testplot_FTS_full.png', dpi=300)
plt.show()


'''
CDF for TM antenna

'''
# %%
"""
Calculate areas of spherical squares
--->>> object shape info: hard coded <<<---
"""
def spherical_squares():
    Area=[]
    R=1
    lonDelta = 2
    for Az in range(0,181,2):
        if Az == 0:
            lat1Rad = np.radians(90-Az)
            lat2Rad = np.radians(90-Az-1)
        elif Az == 180:
            lat1Rad = np.radians(90-Az+1)
            lat2Rad = np.radians(90-Az)
        else:
            lat1Rad = np.radians(90-Az-1)
            lat2Rad = np.radians(90-Az+1)
        AreaRing = 2*np.pi*R**2*np.fabs(np.sin(lat1Rad)-np.sin(lat2Rad))
        Area.append(AreaRing / (360/lonDelta))
    Z = [Area]*181
    Z = np.array(Z)     
    return Z
    # end function

# %%
"""
Calculate the minimum antenna gain over 95% of the radiation sphere. 
1) At AZ = 0 and 180 (nose and tail) take the mean of these values
2) Slice out of the nose and tail data
3) Flatten arrays
4) Append nose and tail averages and area
5) Use Anthony Lin's area code to calculate gain cut off
A = tempory gain matrix
B = temport spherical square surface area matrix
args: data_object
returns: minimum gain over 95% of radiation sphere by area
"""

# calculate areas of spherical squares
sqr_areas = spherical_squares()

nose_avg_gain = np.mean(Mf_f425[:,0])    #average gain at nose
tail_avg_gain = np.mean(Mf_f425[:,90])   #average gain at tail

A = Mf_f425[:,1:91]   #slice out nose and tail data
A = A[:,0:89]       #slice out tail data
gains = A.flatten()

B = sqr_areas[:,1:91]   #slice out nose area column
B = B[:,0:89]   #slice out tail area column
B = np.array(B)
areas = B.flatten()

#append nose & tail gains and area
gains = np.append(gains,nose_avg_gain)
gains = np.append(gains,tail_avg_gain)
areas = np.append(areas,sqr_areas[0,0]*180.0)    #append area of north pole cap
areas = np.append(areas,sqr_areas[0,0]*180.0)    #append area of south pole cap

R=1 #Unit Radius for Sphere, Normalize Later
AreaEquation = 4*np.pi*R**2
FifteenPercent = AreaEquation/20

# search for lowest gain and remove area, until 15% is gone
AreaRemoved = 0
while True:
    MinIndex = np.argmin(gains)  #get index of smallest gain
    AreaRemoved = AreaRemoved+areas[MinIndex]
    if AreaRemoved >= FifteenPercent:
        GainCutoff = gains[MinIndex]
        break
    else:
        gains[MinIndex] = 1000   #set to high value when removed

print(GainCutoff)

# %%
'''
polar plot for roll cut
Need to add pitch cut <<<<<<----------------

roll cut = S2{45,:}
pitch cut = S2[:,0]+S2[:,90]

A roll antenna 
pattern (2 degree step in phi, theta = 90 degrees) 
spec limit: 85% CDF 30 to 150 +/-60
The antenna shall have a gain of not less than –3.85 dBi over 
85% of the measurement angle theta varied from 30 to 150 
degrees and phi varied over +/- 60 degrees. 
'''

# plot roll and pitch cuts
roll = Mf_f425[:,45]

pitch = np.flip(np.concatenate((
    Mf_f425[0,45:90],
    np.flip(Mf_f425[90,45:91],axis=0),    
    np.flip(Mf_f425[90,0:45],axis=0),    
    Mf_f425[0,0:45])),axis=0)

yaw = np.flip(np.concatenate((
    Mf_f425[45,45:90],
    np.flip(Mf_f425[135,45:91],axis=0),    
    np.flip(Mf_f425[135,0:45],axis=0),    
    Mf_f425[45,0:45])),axis=0)

ax = plt.subplot(111, polar=True)
ax.grid(True)
minGrid = -25	 #set plot grid
maxGrid = 15
gridSpacing = 5
ax.set_yticks(np.arange(minGrid,maxGrid,gridSpacing))
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.autoscale(enable=False)

# Polar plot for data
# set x axis
plot_x = phi * np.pi/180

# add data to the plot
ax.plot(plot_x,roll+10, color='r', linewidth = 2, label = 'roll')
ax.plot(plot_x,pitch+10, color='b', linewidth = 2, label = 'pitch')
ax.plot(plot_x,yaw+10, color='k', linewidth = 2, label = 'yaw')

# Setting the graph title & legend
#ax.set_title("HARM TM roll & pitch cut")      
ax.legend(bbox_to_anchor =(-.5,1),loc=2)

# save the figure to a temporary file, default seems to be *.png
plt.savefig('testplot_FTS_roll_pitch_yaw_cut.png', dpi=300)
plt.show()


# %%
'''
combine the two images

'''
images = [Image.open(n) for n in ['testplot_FTS_full.png','testplot_FTS_roll_pitch_yaw_cut.png']]
widths, heights = zip(*(i.size for i in images))

total_width = sum(widths)
max_height = max(heights)

new_im = Image.new('RGB', (total_width, max_height))

x_offset = 0
for im in images:
  new_im.paste(im, (x_offset,0))
  x_offset += im.size[0]

new_im.save('testplot_FTS_combo.png')

# %%

  #%%  
'''
Read S11 data

'''
# hard code the path to working directory
os.chdir(r'C:\Users\Jim\Documents\Python SciPy\Antenna Chamber\HARM\App G data\FTS\S11') 

freq = np.loadtxt('HARM_FTS_SN_0011.CSV',delimiter=',',skiprows=5,usecols=(1))/1e6
S11_dB = np.loadtxt('HARM_FTS_SN_0011.CSV',delimiter=',',skiprows=5,usecols=(2))
S12_dB = np.loadtxt('HARM_FTS_SN_0011.CSV',delimiter=',',skiprows=5,usecols=(4))
S21_dB = np.loadtxt('HARM_FTS_SN_0011.CSV',delimiter=',',skiprows=5,usecols=(6))
S22_dB = np.loadtxt('HARM_FTS_SN_0011.CSV',delimiter=',',skiprows=5,usecols=(8))

os.chdir(r'C:\Users\Jim\Documents\Python SciPy\Antenna Chamber\HARM\App G data\FTS') 

#limit = np.ones_like(freq,dtype=float)
#limit[int(np.argwhere(freq == 230.5)):int(np.argwhere(freq == 231.5))] = -9.49
plt.plot(freq,S11_dB,label='S11')
plt.plot(freq,S12_dB,label='S12')
plt.plot(freq,S21_dB,label='S21')
plt.plot(freq,S22_dB,label='S22')

#plt.plot(freq,limit,'r-',label='limit')
plt.ylabel('Return loss, dB')
plt.ylim(-25,-10)
plt.xlabel('Frequency, MHz')
plt.xlim(422,428)
plt.title('Return loss, test plot')
plt.grid()
plt.legend()
plt.savefig('testplot_FTS_S11.png', dpi=300)
plt.show()


# %%


