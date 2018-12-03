"""
Python program to process HARM GPS antenna data for FTSR App G
by Tony Cirineo
Date: Nov 25 , 2018 code copied from HARM_FTS_ver_3.py and modified
Name: HARM AppG GPS ver 1
Synopsis: This program will process HARM GPS data.

Description: 

This script will process HARM antenna pattern data.  Data 
is collected in the chamber by 


 need to fix theta and phi <<<<<-------

This script will read the excel file and process the data.  


Each excel file has data for antenna assembly organized by sheets, one 
file per antenna serial number.



NAWC-CH3155 Performance Specification for GPS antenna

3.1.2.1	Center frequency
The antenna shall operate at a center frequency of 1575.42 MHz .
3.1.2.2	Bandwidth
The minimum operating bandwidth measured at the port of the antenna shall
be +/- 5MHz centered at the frequency specified in paragraph 3.1.2.1 
at a maximum voltage standing wave ratio (VSWR) of 2.0 to 1.

3.1.2.3	Impedance
The antenna shall have a nominal characteristic impedance 
of 50 ohms measured at each antenna port.  All gain and VSWR 
requirements shall be measured with respect to 50 ohms at the antenna port.
3.1.2.4	Antenna patterns
The antenna patterns shall be measured with respect to RHCP 
or vertical and horizontal linear polarization using the proper 
method to combine the results.  The antenna shall have a gain of not 
less than –3.85 dBi over 85% of the measurement angle theta varied 
from 30 to 150 degrees and phi varied over +/- 60 degrees. 

The angle phi is taken to be zero degrees at the top center of the 
missile (umbilical location) and measured clockwise as viewed from 
the tail of the missile.  The angle theta is zero at the missile nose.  
 

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
os.chdir(r'C:\Users\Jim\Documents\Python SciPy\Antenna Chamber\HARM\App G data\GPS\patterns\Feb 4 2010 003') 

# read CSV data files
H_A1 = np.loadtxt('HARM GPS 003 H NOSE A1.TXT')
H_A2 = np.loadtxt('HARM GPS 003 H NOSE A2.TXT')
H_A3 = np.loadtxt('HARM GPS 003 H NOSE A3.TXT')

H_P1 = np.loadtxt('HARM GPS 003 H NOSE P1.TXT')
H_P2 = np.loadtxt('HARM GPS 003 H NOSE P2.TXT')
H_P3 = np.loadtxt('HARM GPS 003 H NOSE P3.TXT')

V_A1 = np.loadtxt('HARM GPS 003 V NOSE A1.TXT')
V_A2 = np.loadtxt('HARM GPS 003 V NOSE A2.TXT')
V_A3 = np.loadtxt('HARM GPS 003 V NOSE A3.TXT')

V_P1 = np.loadtxt('HARM GPS 003 V NOSE P1.TXT')
V_P2 = np.loadtxt('HARM GPS 003 V NOSE P2.TXT')
V_P3 = np.loadtxt('HARM GPS 003 V NOSE P3.TXT')

# calculate vector sum of V and H
CV_1 = 10**(V_A1/10)*np.cos(V_P1*np.pi/180) + 1j*10**(V_A1/10)*np.sin(V_P1*np.pi/180)
CH_1 = 10**(H_A1/10)*np.cos(H_P1*np.pi/180) + 1j*10**(H_A1/10)*np.sin(H_P1*np.pi/180)
C1 = CV_1 + CH_1 
S1 = 10*np.log10(np.abs(C1))+35

CV_2 = 10**(V_A2/10)*np.cos(V_P2*np.pi/180) + 1j*10**(V_A2/10)*np.sin(V_P2*np.pi/180)
CH_2 = 10**(H_A2/10)*np.cos(H_P2*np.pi/180) + 1j*10**(H_A2/10)*np.sin(H_P2*np.pi/180)
C2 = CV_2 + CH_2
S2 = 10*np.log10(np.abs(C2))+35

CV_3 = 10**(V_A3/10)*np.cos(V_P3*np.pi/180) + 1j*10**(V_A3/10)*np.sin(V_P3*np.pi/180)
CH_3 = 10**(H_A3/10)*np.cos(H_P3*np.pi/180) + 1j*10**(H_A3/10)*np.sin(H_P3*np.pi/180)
C3 = CV_3 + CH_3
S3 = 10*np.log10(np.abs(C3))+35

# %%
# change folder for storing plots
os.chdir(r'C:\Users\Jim\Documents\Python SciPy\Antenna Chamber\HARM\App G data\GPS') 

'''
plot full radiation pattern
'''
# fix phi and theta, are they correct?
phi = np.linspace(0,358,180,endpoint=True)
theta = np.linspace(0,180,91,endpoint=True)
X, Y = np.meshgrid(phi, theta)

fig, ax = plt.subplots()
ax.set_title('GPS radiation pattern, 1.575 GHz')
ax.set_xlabel('phi?')
ax.set_ylabel('theta?')

cs = ax.contourf(X, Y, S2)
cbar = fig.colorbar(cs)
cbar.ax.set_ylabel('amplitude, dBic')

plt.savefig('testplot_GPS_full.png', dpi=300)
plt.show()


# %%
'''
CDF for GPS antenna

S1


'''
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
'''
calculate 85% minimum gain over required angles
'''
# calculate areas of spherical squares
sqr_areas = spherical_squares()


data_obj = np.copy(S2[15:75,:]) # slice out desired theta section, 30 to 150 degrees
data_obj1 = np.copy(data_obj[:,0:30]) # slice out phi 0 to 60 degrees
data_obj2 = np.copy(data_obj[:,150:180]) # slice out phi 300 to 360 degrees
gains = data_obj1.flatten() + data_obj2.flatten()

del data_obj, data_obj1, data_obj2

data_obj = np.copy(sqr_areas[:,15:75]) # slice out desired theta section, 30 to 150 degrees
data_obj1 = np.copy(data_obj[0:30,:]) # slice out phi 0 to 60 degrees
data_obj2 = np.copy(data_obj[150:180,:]) # slice out phi 300 to 360 degrees
areas = data_obj1.flatten() + data_obj2.flatten()


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
calculate 85% minimum gain over upper hemisphere
theta = 0 to 180
phi = 0 to 90 and 270 to 360
'''
# calculate areas of spherical squares
sqr_areas = spherical_squares()


data_obj = np.copy(S2) # slice out desired theta section, 0 to 180 degrees
data_obj1 = np.copy(data_obj[:,0:45]) # slice out phi 0 to 90 degrees
data_obj2 = np.copy(data_obj[:,135:180]) # slice out phi 300 to 360 degrees
gains = data_obj1.flatten() + data_obj2.flatten()

del data_obj, data_obj1, data_obj2

data_obj = np.copy(sqr_areas) # slice out desired theta section, 30 to 150 degrees
data_obj1 = np.copy(data_obj[0:45,:]) # slice out phi 0 to 60 degrees
data_obj2 = np.copy(data_obj[135:180,:]) # slice out phi 300 to 360 degrees
areas = data_obj1.flatten() + data_obj2.flatten()


R=1 #Unit Radius for Sphere, Normalize Later
AreaEquation = 4*np.pi*R**2
FifteenPercent = AreaEquation*0.128

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
# To add one subplot
# The 111 specifies 1 row, 1 column on subplot #1
#ax = fig.add_subplot(111, polar=True)
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

plot_LimitLine = np.empty(180)
plot_LimitLine.fill(-100)  # set to -100 dBic for angles not defined
plot_LimitLine[0:30].fill(-3.85)
plot_LimitLine[150:180].fill(-3.85)

# add data to the plot
ax.plot(plot_x,S1[45,:], color='r', linewidth = 2, label = '1.565 GHz')
ax.plot(plot_x,S2[45,:], color='b', linewidth = 2, label = '1.575 GHz')
ax.plot(plot_x,S3[45,:], color='k', linewidth = 2, label = '1.585 GHz')
ax.plot(plot_x,plot_LimitLine, 'g--', linewidth = 2, label = 'Limit')

# Setting the graph title & legend
#ax.set_title("HARM GPS roll Cut")      
ax.legend(bbox_to_anchor =(-.5,1),loc=2)

# save the figure to a temporary file, default seems to be *.png
plt.savefig('testplot_GPS_roll.png', dpi=300)
plt.show()


# %%
# plot roll and pitch cuts
roll = S2[45,:]
pitch = np.flip(np.concatenate((
    S2[45:90,0],
    np.flip(S2[45:90,90],axis=0),    
    np.flip(S2[0:45,90],axis=0),    
    S2[0:45,0])),axis=0)

'''
pitch = np.concatenate((
    S2[0:45,0],
    np.flip(S2[0:45,90],axis=0), 
    np.flip(S2[45:90,90],axis=0),         
    S2[45:90,0]))
'''

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
ax.plot(plot_x,roll, color='r', linewidth = 2, label = 'roll')
ax.plot(plot_x,pitch, color='b', linewidth = 2, label = 'pitch')

# Setting the graph title & legend
# ax.set_title("HARM GPS roll & pitch cut at 1.575GHz")      
ax.legend(bbox_to_anchor =(1,1),loc=2)

# save the figure to a temporary file, default seems to be *.png
plt.savefig('testplot_GPS_roll_pitch_cut.png', dpi=300)
plt.show()

# %%
'''
plot contour side by side with pitch and roll
can't get to work
try using python to combine png files.
'''
# plot contour to the right side
# fix phi and theta, are they correct?
phi = np.linspace(0,358,180,endpoint=True)
theta = np.linspace(0,180,91,endpoint=True)
X, Y = np.meshgrid(phi, theta)

#fig = plt.figure()
#ax = plt.subplot(211)

#fig, ax = plt.subplot(121)
fig, ax = plt.subplots(nrows=1,ncols=3,sharex=False,sharey=False,squeeze=True)
#fig = plt.figure()
#ax = fig.subplots(nrows=1,ncols=2)
ax[0].set_title('GPS radiation pattern, 1.575 GHz')
ax[0].set_xlabel('phi?')
ax[0].set_ylabel('theta?')

cs = ax[0].contourf(X, Y, S2)
cbar = fig.colorbar(cs,cax=ax[1])
cbar.ax.set_ylabel('amplitude, dBic')
#'''
# plot roll and pitch cuts, to the left side
roll = S2[45,:]
pitch = np.flip(np.concatenate((
    S2[45:90,0],
    np.flip(S2[45:90,90],axis=0),    
    np.flip(S2[0:45,90],axis=0),    
    S2[0:45,0])),axis=0)

#fig2,ax2 = plt.subplot(1, 2, 2)
#plt.figure(2)
ax = plt.subplot(133,polar=True)
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
ax.plot(plot_x,roll, color='r', linewidth = 2, label = 'roll')
ax.plot(plot_x,pitch, color='b', linewidth = 2, label = 'pitch')

# Setting the graph title & legend
# ax.set_title("HARM GPS roll & pitch cut at 1.575GHz")      
ax.legend(bbox_to_anchor =(1,1),loc=2)
#'''
plt.show()
# %%
'''
combine the two images
'testplot_GPS_roll_pitch_cut.png'
'testplot_GPS_full.png'
'''
images = [Image.open(n) for n in ['testplot_GPS_full.png','testplot_GPS_roll_pitch_cut.png']]
widths, heights = zip(*(i.size for i in images))

total_width = sum(widths)
max_height = max(heights)

new_im = Image.new('RGB', (total_width, max_height))

x_offset = 0
for im in images:
  new_im.paste(im, (x_offset,0))
  x_offset += im.size[0]

new_im.save('testplot_GPS_combo.png')

# %%

'''
Find azmuth angles for which gain is greater than X.

what angle is gain greater than X? 

check for non zero 1st value
check for gaps
check for empty list

still needs work <<<<<-------------------
'''

# set gain values to False if less than limit
mask = np.less_equal(np.ones_like(S2)*-3,S2)

#np.where(mask[71,0:100] == True)
Q = S2*mask

plt.imshow(Q)
plt.show()

def missing_elements(L):
    start, end = L[0], L[-1]
    return sorted(set(range(start, end + 1)).difference(L))

P = np.zeros(len(theta),dtype=float)
# at phi = 0, find threshold angles
for i in range(len(theta)):
    L = np.where(mask[i,0:90] == True)[0].tolist()

    if len(L) == 0:
        P[i] = 0
        continue
    if L[0] != 0:
        P[i] = 0        
        continue
    if len(missing_elements(L)) > 0:
        P[i] = missing_elements(L)[0]
    else:
        P[i] = L[-1]

P1 = np.zeros(len(theta),dtype=float)
# at phi = 0, find threshold angles
for i in range(len(theta)):
    L = np.where(np.flip(mask[i,90:180],axis=0) == True)[0].tolist()    

    if len(L) == 0:
        P1[i] = 0
        continue
    if L[0] != 0:
        P1[i] = 0        
        continue
    if len(missing_elements(L)) > 0:
        P1[i] = missing_elements(L)[0]
    else:
        P1[i] = L[-1]

P = np.concatenate((P,P1))
plt.plot(90-2*P)
plt.xlim(0,50)
plt.ylim(0,40)
plt.show()

# %%
    
'''
Read S11 data
3.1.2.2	Bandwidth
The minimum operating bandwidth measured at the port of the antenna shall
be +/- 5MHz centered at the frequency specified in paragraph 3.1.2.1 
at a maximum voltage standing wave ratio (VSWR) of 2.0 to 1.


'''
# hard code the path to working directory
os.chdir(r'C:\Users\Jim\Documents\Python SciPy\Antenna Chamber\HARM\App G data\GPS\S11\data') 

freq = np.loadtxt('SN 009 GPS.CSV',delimiter=',',skiprows=5,usecols=(1))/1e9
S11_dB = np.loadtxt('SN 009 GPS.CSV',delimiter=',',skiprows=5,usecols=(2))

os.chdir(r'C:\Users\Jim\Documents\Python SciPy\Antenna Chamber\HARM\App G data\GPS\S11') 

limit = np.ones_like(freq,dtype=float)
limit[int(np.argwhere(freq == 1.57)):int(np.argwhere(freq == 1.58))] = -9.54
plt.plot(freq,S11_dB,label='S11')
plt.plot(freq,limit,'r-',label='limit')
plt.ylabel('Return loss, dB')
plt.ylim(-20,0)
plt.xlabel('Frequency, GHz')
plt.xlim(1.56,1.59)
plt.title('Return loss, test plot')
plt.grid()
plt.legend()
plt.savefig('testplot_GPS_S11.png')
plt.show()

print('average refl coef = {:.1f} dB'.format(S11_dB[int(np.argwhere(freq == 1.57)):int(np.argwhere(freq == 1.58))].mean()))
print('min refl coef = {:.1f} dB'.format(S11_dB[int(np.argwhere(freq == 1.57)):int(np.argwhere(freq == 1.58))].min()))
print('max refl coef = {:.1f} dB'.format(S11_dB[int(np.argwhere(freq == 1.57)):int(np.argwhere(freq == 1.58))].max()))

