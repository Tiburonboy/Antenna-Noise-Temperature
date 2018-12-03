"""
Python program to process HARM TM antenna data for FTSR App G
by Tony Cirineo
Date: Nov 25 , 2018 code copied from HARM_FTS_ver_3.py and modified
Name: HARM AppG TM ver 1
Synopsis: This program will process HARM GPS data.

Description: 

This script will process HARM antenna pattern data.  Data 
is collected in the chamber by 


 need to fix theta and phi <<<<<-------

This script will read the excel file and process the data.  


TM antenna spec
NAWC-CH3156
NAVAL AIR WARFARE CENTER
WEAPONS DIVISION
DEPARTMENT OF THE NAVY
PERFORMANCE SPECIFICATION
FOR
TELEMETRY ANTENNA SYSTEM
3.1.2.1	Center frequency
The antenna shall operate at a center frequency of 231.0 MHz or 241.2 MHz as specified in the contract.
3.1.2.2	Bandwidth
The minimum operating bandwidth measured at the port of each antenna element shall be +/- 0.45MHz centered at the frequency specified in paragraph 3.1.2.1 at a maximum voltage standing wave ratio (VSWR) of 2.0 to 1.

3.1.2.3	Impedance
The antenna element shall have a nominal characteristic impedance of 50 ohms measured at the antenna port.  All gain and VSWR requirements shall be measured with respect to 50 ohms at the antenna port or antenna element.
3.1.2.4	Antenna patterns
The antenna system patterns shall be measured with respect to linear polarization.  The antenna design shall minimize the number and angular width of deep nulls in the pattern.  The antenna shall have the following gains:
(a) Roll: The antenna system shall have a gain of not less than -20 dBi in the roll plane (theta = 90 degrees, phi varied) for both the horizontal and the vertical polarized measurement at the specified measurement angle.
(b) Nose:  The antenna shall have a gain of not less than -10 dBi at the nose (theta = 0 degrees, phi varied).  The gain at the nose is the maximum value of the horizontal or vertical polarized measurement at the specified measurement angle. 
(c) Tail:  The antenna system shall have a gain of not less than -15 dBi at the tail (theta = 180 degrees, phi varied).  The gain at the tail is the maximum value of the horizontal or vertical polarized measurement at the specified measurement angle.
(d) Sphere:  The antenna system shall have a gain over the full radiation sphere of not less than -15 dBi over 90% of the radiation sphere. The gain for the sphere is the maximum value of the horizontal or vertical polarized measurement at the specified measurement angle.
(e) The predominate polarization shall be orientated along the Z axis of figure 6, with the average orthogonal component not more than 15 dB lower than the predominate component.

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
import matplotlib.pyplot as plt
from PIL import Image
# %%
# hard code the path to working directory
os.chdir(r'C:\Users\Jim\Documents\Python SciPy\Antenna Chamber\HARM\App G data\TM\patterns\Dec 9 2009 A010 011 012') 

# read CSV data files, HARM TM 010 011 012 H NOSE A1  231MHz
H_A1 = np.loadtxt('HARM TM 010 011 012 H NOSE A1.TXT')
H_A2 = np.loadtxt('HARM TM 010 011 012 H NOSE A2.TXT')
H_A3 = np.loadtxt('HARM TM 010 011 012 H NOSE A3.TXT')

H_P1 = np.loadtxt('HARM TM 010 011 012 H NOSE P1.TXT')
H_P2 = np.loadtxt('HARM TM 010 011 012 H NOSE P2.TXT')
H_P3 = np.loadtxt('HARM TM 010 011 012 H NOSE P3.TXT')

V_A1 = np.loadtxt('HARM TM 010 011 012 V NOSE A1.TXT')
V_A2 = np.loadtxt('HARM TM 010 011 012 V NOSE A2.TXT')
V_A3 = np.loadtxt('HARM TM 010 011 012 V NOSE A3.TXT')

V_P1 = np.loadtxt('HARM TM 010 011 012 V NOSE P1.TXT')
V_P2 = np.loadtxt('HARM TM 010 011 012 V NOSE P2.TXT')
V_P3 = np.loadtxt('HARM TM 010 011 012 V NOSE P3.TXT')


# pick max value from vert or horiz
M1 = np.maximum(H_A1,V_A1)
M2 = np.maximum(H_A2,V_A2)
M3 = np.maximum(H_A3,V_A3)
# %%

# change folder for storing plots
os.chdir(r'C:\Users\Jim\Documents\Python SciPy\Antenna Chamber\HARM\App G data\TM') 

'''
plot full radiation pattern
'''
# fix phi and theta, are they correct?
phi = np.linspace(0,358,180,endpoint=True)
theta = np.linspace(0,180,91,endpoint=True)
X, Y = np.meshgrid(phi, theta)

fig, ax = plt.subplots()
ax.set_title('TM radiation pattern, 230 MHz')
ax.set_xlabel('phi?')
ax.set_ylabel('theta?')

cs = ax.contourf(X, Y, M2)
cbar = fig.colorbar(cs)
cbar.ax.set_ylabel('amplitude, dBi')

plt.savefig('testplot_TM_full.png', dpi=300)
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
    Z = [Area]*180    # not 181
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

nose_avg_gain = np.mean(M2[:,0])    #average gain at nose
tail_avg_gain = np.mean(M2[:,90])   #average gain at tail

A = M2[:,1:91]   #slice out nose and tail data
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
FifteenPercent = AreaEquation*0.15

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
roll = M2[45,:]
pitch = np.flip(np.concatenate((
    M2[45:90,0],
    np.flip(M2[45:90,90],axis=0),    
    np.flip(M2[0:45,90],axis=0),    
    M2[0:45,0])),axis=0)

'''
pitch = np.concatenate((
    S2[0:45,0],
    np.flip(S2[0:45,90],axis=0), 
    np.flip(S2[45:90,90],axis=0),         
    S2[45:90,0]))
'''

yaw = np.flip(np.concatenate((
    M2[45:90,45],
    np.flip(M2[45:90,135],axis=0),    
    np.flip(M2[0:45,135],axis=0),    
    M2[0:45,45])),axis=0)

##pitch = np.concatenate((S2[0:90,0],np.flip(S2[0:90,90],axis=0)))

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
plt.savefig('testplot_TM_roll_pitch_yaw_cut.png', dpi=300)
plt.show()

# %%
'''
combine the two images

'''
images = [Image.open(n) for n in ['testplot_TM_full.png','testplot_TM_roll_pitch_yaw_cut.png']]
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
os.chdir(r'C:\Users\Jim\Documents\Python SciPy\Antenna Chamber\HARM\App G data\TM\S11\231_MHz\A-018') 

freq = np.loadtxt('HARM SN A-018 (231 MHz) TUNED  CUREDspt.CSV',delimiter=',',skiprows=5,usecols=(1))/1e6
S11_dB = np.loadtxt('HARM SN A-018 (231 MHz) TUNED  CUREDspt.CSV',delimiter=',',skiprows=5,usecols=(2))

os.chdir(r'C:\Users\Jim\Documents\Python SciPy\Antenna Chamber\HARM\App G data\TM') 


limit = np.ones_like(freq,dtype=float)
limit[int(np.argwhere(freq == 230.5)):int(np.argwhere(freq == 231.5))] = -9.49
plt.plot(freq,S11_dB,label='S11')
plt.plot(freq,limit,'r-',label='limit')
plt.ylabel('Return loss, dB')
plt.ylim(-20,0)
plt.xlabel('Frequency, MHz')
plt.xlim(230,232)
plt.title('Return loss, test plot')
plt.grid()
plt.legend()
plt.savefig('testplot_TM_S11.png', dpi=300)
plt.show()


# %%


