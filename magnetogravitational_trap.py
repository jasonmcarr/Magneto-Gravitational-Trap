import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

atmos = pd.read_csv('/Users/Katie/Desktop/2018-2019/PHSX444/Lab3/trajectory3.csv')
atmos = atmos.loc[:, "x":"ep"]
atmos = atmos.values

lv = pd.read_csv('/Users/Katie/Desktop/2018-2019/PHSX444/Lab3/120mTorr_Low_Trajectory.csv')
lv = lv.loc[:, "x":"ep"]
lv = lv.values

hv = pd.read_csv('/Users/Katie/Desktop/2018-2019/PHSX444/Lab3/1E-7Torr_High_Trajectory.csv')
hv = hv.loc[:, "x":"ep"]
hv = hv.values

cal = 1.667 #multiply to convert pixels to microns
atm_ax = atmos[:, 0]*cal #axial motion
atm_v = atmos[:, 1]*cal #vertical motion
atm_error = atmos[:, 7]*cal #error

lv_ax = lv[:, 0]*cal
lv_v = lv[:, 1]*cal
lv_error = lv[:, 7]*cal

hv_ax = hv[:, 0]*cal
hv_v = hv[:, 1]*cal
hv_error = hv[:, 7]*cal

atm_ax_avg = atm_ax - np.mean(atm_ax)
atm_v_avg = atm_v - np.mean(atm_v)
lv_ax_avg = lv_ax - np.mean(lv_ax)
lv_v_avg = lv_v - np.mean(lv_v)
hv_ax_avg = hv_ax - np.mean(hv_ax)
hv_v_avg = hv_v - np.mean(hv_v)

fig0, ax0 = plt.subplots(3, 1, figsize = (8, 18))
ax0[0].plot(atm_ax, atm_v)
ax0[0].set_title("Atmospheric pressure")
ax0[0].set_xlabel("z ($\mathrm{\mu m}$)")
ax0[0].set_ylabel("y ($\mathrm{\mu m}$)")

ax0[1].plot(lv_ax, lv_v)
ax0[1].set_title("Rough vacuum")
ax0[1].set_xlabel("z ($\mathrm{\mu m}$)")
ax0[1].set_ylabel("y ($\mathrm{\mu m}$)")

ax0[2].plot(hv_ax, hv_v)
ax0[2].set_title("High vacuum")
ax0[2].set_xlabel("z ($\mathrm{\mu m}$)")
ax0[2].set_ylabel("y ($\mathrm{\mu m}$)")
plt.show()

#histograms
fig1, ax1 = plt.subplots(1, 2, figsize = (16, 6))
fig1.suptitle("Histograms for atmospheric pressure", fontsize=16, fontweight='bold')
ax1[0].hist(atm_v_avg**2, bins = 30)
ax1[0].set_title("Vertical motion")
ax1[0].set_yscale("log")
ax1[0].set_xlabel("$|y-y_0|^2$ $\mathrm{\mu m}^2$")
ax1[0].set_ylabel("Occurences")

ax1[1].hist(atm_ax_avg**2, bins = 30)
ax1[1].set_title("Axial motion")
ax1[1].set_yscale("log")
ax1[1].set_xlabel("$|z-z_0|^2$ $\mathrm{\mu m}^2$")
ax1[1].set_ylabel("Occurences")

fig2, ax2 = plt.subplots(1, 2, figsize = (16, 6))
fig2.suptitle("Histograms for rough vacuum", fontsize=16, fontweight='bold')
ax2[0].hist(lv_v_avg**2, bins = 30)
ax2[0].set_title("Vertical motion")
ax2[0].set_yscale("log")
ax2[0].set_xlabel("$|y-y_0|^2$ $\mathrm{\mu m}^2$")
ax2[0].set_ylabel("Occurences")

ax2[1].hist(lv_ax_avg**2, bins = 30)
ax2[1].set_title("Axial motion")
ax2[1].set_yscale("log")
ax2[1].set_xlabel("$|z-z_0|^2$ $\mathrm{\mu m}^2$")
ax2[1].set_ylabel("Occurences")

fig3, ax3 = plt.subplots(1, 2, figsize = (16, 6))
fig3.suptitle("Histograms for high vacuum", fontsize=16, fontweight='bold')
ax3[0].hist(hv_v_avg**2, bins = 30)
ax3[0].set_title("Vertical motion")
ax3[0].set_yscale("log")
ax3[0].set_xlabel("$|y-y_0|^2$ $\mathrm{\mu m}^2$")
ax3[0].set_ylabel("Occurences")

ax3[1].hist(hv_ax_avg**2, bins = 30)
ax3[1].set_title("Axial motion")
ax3[1].set_yscale("log")
ax3[1].set_xlabel("$|z-z_0|^2$ $\mathrm{\mu m}^2$")
ax3[1].set_ylabel("Occurences")
plt.show()

##joint probability distributions
fig4, ax4 = plt.subplots(1, 2, figsize = (16, 6))
fig4.suptitle("Joint probability distribution for atmospheric pressure", fontsize=16, fontweight='bold')
ax4[0].plot(atm_v[0:4996], atm_v[4:5000], '.')
ax4[0].set_title("Vertical motion")
ax4[0].set_xlabel("y(t) (microns)")
ax4[0].set_ylabel("y(t+0.008 s) (microns)")

ax4[1].plot(atm_ax[0:4985], atm_ax[15:5000], '.')
ax4[1].set_title("Axial motion")
ax4[1].set_xlabel("z(t) (microns)")
ax4[1].set_ylabel("z (t+0.03 s) (microns)")

fig5, ax5 = plt.subplots(1, 2, figsize = (16, 6))
fig5.suptitle("Joint probability distribution for rough vacuum", fontsize=16, fontweight='bold')
ax5[0].plot(lv_v[0:4999], lv_v[1:5000], '.')
ax5[0].set_title("Vertical motion")
ax5[0].set_xlabel("y(t) (microns)")
ax5[0].set_ylabel("y(t+0.002) (microns)")

ax5[1].plot(lv_ax[0:4996], lv_ax[4:5000], '.')
ax5[1].set_title("Axial motion")
ax5[1].set_xlabel("z(t) (microns)")
ax5[1].set_ylabel("z(t+0.008 s) (microns)")

fig6, ax6 = plt.subplots(1, 2, figsize = (16, 6))
fig6.suptitle("Joint probability distribution for high vacuum", fontsize=16, fontweight='bold')
ax6[0].plot(hv_v[0:4999], hv_v[1:5000], '.')
ax6[0].set_title("Vertical motion")
ax6[0].set_xlabel("y(t) (microns)")
ax6[0].set_ylabel("y(t+0.002 s) (microns)")

ax6[1].plot(hv_ax[0:4996], hv_ax[4:5000], '.')
ax6[1].set_title("Axial motion")
ax6[1].set_xlabel("z(t) (microns)")
ax6[1].set_ylabel("z(t+0.008 s) (microns")
plt.show()

##ffts
atm_ax_avg_ha = atm_ax_avg*np.hanning(len(atm_ax_avg))
fatm_ax = np.fft.rfft(atm_ax_avg)
nuatm_ax = np.fft.rfftfreq(len(atm_ax_avg), 1./500.)
#
#atm_v_avg_ha = atm_v_avg*np.hanning(len(atm_v_avg))
fatm_v = np.fft.rfft(atm_v_avg)
nuatm_v = np.fft.rfftfreq(len(atm_v_avg), 1./500.)

fig7, ax7 = plt.subplots(1, 2, figsize = (16, 6))
fig7.suptitle("FFTs for Atmospheric Pressure", fontsize = 16, fontweight = 'bold')
ax7[0].plot(nuatm_ax, np.abs(fatm_ax)**2)
ax7[0].set_yscale('log')
ax7[0].set_title("Axial Motion")
ax7[0].set_xlabel("Frequency (Hz)")
ax7[0].set_ylabel("Power Spectrum")
ax7[0].set_ylim(1e2)
ax7[0].set_xlim(0, 120)

ax7[1].plot(nuatm_v, np.abs(fatm_v)**2)
ax7[1].set_yscale('log')
ax7[1].set_title("Vertical Motion")
ax7[1].set_xlabel("Frequency (Hz)")
ax7[1].set_ylabel("Power Spectrum")
ax7[1].set_ylim(1e2)
ax7[1].set_xlim(0, 120)
plt.show()

lv_ax_avg_ha = lv_ax_avg*np.hanning(len(lv_ax_avg))
flv_ax = np.fft.rfft(lv_ax_avg)
nulv_ax = np.fft.rfftfreq(len(lv_ax_avg), 1./500.)
#
#lv_v_avg_ha = lv_v_avg*np.hanning(len(lv_v_avg))
flv_v = np.fft.rfft(lv_v_avg)
nulv_v = np.fft.rfftfreq(len(lv_v_avg), 1./500.)

fig8, ax8 = plt.subplots(1, 2, figsize = (16, 6))
fig8.suptitle("FFTs for Rough Vacuum", fontsize = 16, fontweight = 'bold')
ax8[0].plot(nulv_ax, np.abs(flv_ax)**2)
ax8[0].set_yscale('log')
ax8[0].set_title("Axial Motion")
ax8[0].set_xlabel("Frequency (Hz)")
ax8[0].set_ylabel("Power Spectrum")
ax8[0].set_ylim(1e5)
ax8[0].set_xlim(0, 120)

ax8[1].plot(nulv_v, np.abs(flv_v)**2)
ax8[1].set_yscale('log')
ax8[1].set_title("Vertical Motion")
ax8[1].set_xlabel("Frequency (Hz)")
ax8[1].set_ylabel("Power Spectrum")
ax8[1].set_ylim(1e3)
ax8[1].set_xlim(0, 120)
plt.show()

hv_ax_avg_ha = hv_ax_avg*np.hanning(len(hv_ax_avg))
fhv_ax = np.fft.rfft(hv_ax_avg)
nuhv_ax = np.fft.rfftfreq(len(hv_ax_avg), 1./500.)
#
#hv_v_avg_ha = hv_v_avg*np.hanning(len(hv_v_avg))
fhv_v = np.fft.rfft(hv_v_avg)
nuhv_v = np.fft.rfftfreq(len(hv_v_avg), 1./500.)

fig9, ax9 = plt.subplots(1, 2, figsize = (16, 6))
fig9.suptitle("FFTs for High Vacuum", fontsize = 16, fontweight = 'bold')
ax9[0].plot(nuhv_ax, np.abs(fhv_ax)**2)
ax9[0].set_yscale('log')
ax9[0].set_title("Axial Motion")
ax9[0].set_xlabel("Frequency (Hz)")
ax9[0].set_ylabel("Power Spectrum")
ax9[0].set_ylim(1e6)
ax9[0].set_xlim(0, 120)

ax9[1].plot(nuhv_v, np.abs(fhv_v)**2)
ax9[1].set_yscale('log')
ax9[1].set_title("Vertical Motion")
ax9[1].set_xlabel("Frequency (Hz)")
ax9[1].set_ylabel("Power Spectrum")
ax9[1].set_ylim(1e6)
ax9[1].set_xlim(0, 120)
plt.show()

#measure particle masses    
def function_to_fit(x, a, b):
    return(a*np.exp(-b*x))
    
w_v = 90 #Hz
w_ax = 10 #Hz
kB = 1.38e-23 #m^2 kg s^-2 K^-1
T = 298 #K
#atmospheric pressure:
atm_v_avg = atm_v_avg[np.where(np.abs(atm_v_avg)**2<6)]
y_atm_v = np.histogram((np.abs(atm_v_avg)*10**-6)**2, bins=30)
#atm_error = 2*atm_v_avg*atm_error[0:49999]
popt_atm_v, pcov_atm_v = curve_fit(function_to_fit, y_atm_v[1][5:30], y_atm_v[0][5:30], 
                                   sigma = np.sqrt(y_atm_v[0][5:30]), 
                                   p0 = np.array([1e3, 3.35e9]))
m_atm_v = (2*kB*T*popt_atm_v[1])/(w_v**2)

y_atm_ax = np.histogram((np.abs(atm_ax_avg)*10**-6)**2, bins=30)
popt_atm_ax, pcov_atm_ax = curve_fit(function_to_fit, y_atm_ax[1][0:30], y_atm_ax[0][0:30], 
                                   sigma = np.sqrt(y_atm_ax[0][0:30]),
                                   p0 = np.array([1e3, 3.35e9]))
m_atm_ax = (2*kB*T*popt_atm_ax[1])/(w_ax**2)

#low vacuum:
lv_v_avg = lv_v_avg[np.where(np.abs(lv_v_avg)**2<4.75)]
y_lv_v = np.histogram((np.abs(lv_v_avg)*10**-6)**2, bins=30)
popt_lv_v, pcov_lv_v = curve_fit(function_to_fit, y_lv_v[1][0:30], y_lv_v[0][0:30], 
                                   sigma = np.sqrt(y_lv_v[0][0:30]),
                                   p0 = np.array([1e3, 3.35e9]))
m_lv_v = (2*kB*T*popt_lv_v[1])/(w_v**2)

lv_ax_avg = lv_ax_avg[np.where(np.abs(lv_ax_avg)**2<25)]
y_lv_ax = np.histogram((np.abs(lv_ax_avg)*10**-6)**2, bins=30)
popt_lv_ax, pcov_lv_ax = curve_fit(function_to_fit, y_lv_ax[1][0:30], y_lv_ax[0][0:30], 
                                   sigma = np.sqrt(y_lv_ax[0][0:30]),
                                   p0 = np.array([1e3, 3.35e9]))
m_lv_ax = (2*kB*T*popt_lv_ax[1])/(w_ax**2)
#covariance of parameters could not be estimated?

#high vacuum:
hv_v_avg = hv_v_avg[np.where(np.abs(hv_v_avg)**2<230)]
y_hv_v = np.histogram((np.abs(hv_v_avg)*10**-6)**2, bins=30)
popt_hv_v, pcov_hv_v = curve_fit(function_to_fit, y_hv_v[1][0:30], y_hv_v[0][0:30], 
                                   sigma = np.sqrt(y_hv_v[0][0:30]),
                                   p0 = np.array([1e3, 3.35e9]))
m_hv_v = (2*kB*T*popt_hv_v[1])/(w_v**2)

hv_ax_avg = hv_ax_avg[np.where(np.abs(hv_ax_avg)**2<5000)]
y_hv_ax = np.histogram((np.abs(hv_ax_avg)*10**-6)**2, bins=30)
popt_hv_ax, pcov_hv_ax = curve_fit(function_to_fit, y_hv_ax[1][0:30], y_hv_ax[0][0:30], 
                                   sigma = np.sqrt(y_hv_ax[0][0:30]),
                                   p0 = np.array([1e3, 3.35e9]))
m_hv_ax = (2*kB*T*popt_hv_ax[1])/(w_ax**2)

m_v = np.array([m_atm_v, m_lv_v, m_hv_v])
m_ax = np.array([m_atm_ax, m_atm_ax, m_atm_ax])
m_avg = np.array([(m_v[0]+m_ax[0])/2, (m_v[1]+m_ax[1])/2, (m_v[2]+m_ax[2])/2])
print(m_avg)

sigma_atm_v = np.sqrt(np.diag(pcov_atm_v))
sigma_m_atm_v = sigma_atm_v[1]
sigma_atm_ax = np.sqrt(np.diag(pcov_atm_ax))
sigma_m_atm_ax = sigma_atm_ax[1]

sigma_lv_v = np.sqrt(np.diag(pcov_lv_v))
sigma_m_lv_v = sigma_lv_v[1]
sigma_lv_ax = np.sqrt(np.diag(pcov_lv_ax))
sigma_m_lv_ax = sigma_lv_ax[1]

sigma_hv_v = np.sqrt(np.diag(pcov_hv_v))
sigma_m_hv_v = sigma_hv_v[1]
sigma_hv_ax = np.sqrt(np.diag(pcov_hv_ax))
sigma_m_hv_ax = sigma_hv_ax[1]

sigma_m_atm = np.sqrt(sigma_m_atm_v**2 + sigma_m_atm_ax**2)
sigma_m_lv = np.sqrt(sigma_m_lv_v**2 + sigma_m_lv_ax**2)
sigma_m_hv = np.sqrt(sigma_m_hv_v**2 + sigma_m_hv_ax**2)

sigma_m = np.array([sigma_m_atm, sigma_m_lv, sigma_m_hv])
print(sigma_m)

plt.figure(figsize = (8, 6))
plt.plot(y_atm_v[1][0:30], y_atm_v[0][0:30], '.', label = "original data")
plt.plot(y_atm_v[1][5:30], function_to_fit(y_atm_v[1][5:30], *popt_atm_v), label = "best fit")
plt.yscale('log')
plt.ylabel("Occurences")
plt.xlabel("$|y-y_0|^2$ $\mathrm{\mu m}^2$")
plt.title("Boltzmann Distribution Fit: Vertical Motion, Atmospheric Pressure")
plt.legend()
plt.show()

#residuals!!!
r_atm_v = np.zeros(len(y_atm_v[0][5:30]))
for i in range(len(r_atm_v)):
    r_atm_v[i] = y_atm_v[0][i] - function_to_fit(y_atm_v[1][i], *popt_atm_v)
r_atm_ax = np.zeros(len(y_atm_ax[0]))
for i in range(len(r_atm_ax)):
    r_atm_ax[i] = y_atm_ax[0][i] - function_to_fit(y_atm_ax[1][i], *popt_atm_ax)
    
r_lv_v = np.zeros(len(y_lv_v[0]))
for i in range(len(r_lv_v)):
    r_lv_v[i] = y_lv_v[0][i] - function_to_fit(y_lv_v[1][i], *popt_lv_v)
r_lv_ax = np.zeros(len(y_lv_ax[0]))
for i in range(len(r_lv_ax)):
    r_lv_ax[i] = y_lv_ax[0][i] - function_to_fit(y_lv_ax[1][i], *popt_lv_ax)
    
r_hv_v = np.zeros(len(y_hv_v[0]))
for i in range(len(r_hv_v)):
    r_hv_v[i] = y_hv_v[0][i] - function_to_fit(y_hv_v[1][i], *popt_hv_v)
r_hv_ax = np.zeros(len(y_hv_ax[0]))
for i in range(len(r_hv_ax)):
    r_hv_ax[i] = y_hv_ax[0][i] - function_to_fit(y_hv_ax[1][i], *popt_hv_ax)

#print(np.shape(y_atm_ax[1]), len(r_atm_ax))
fig10, ax10 = plt.subplots(3, 2, figsize = (16, 24))
fig10.suptitle("Residuals", fontsize=16, fontweight='bold')
ax10[0, 0].errorbar(y_atm_v[1][5:30], r_atm_v, np.sqrt(y_atm_v[0][5:30]), fmt='.')
ax10[0, 0].set_xlabel("$|y-y_0|^2$ $\mathrm{\mu m}^2$")
ax10[0, 0].set_ylabel("Residuals")
ax10[0, 0].set_title("Vertical motion, atmospheric pressure")

ax10[0, 1].errorbar(y_atm_ax[1][0:30], r_atm_ax, np.sqrt(y_atm_ax[0]), fmt='.')
ax10[0, 1].set_xlabel("$|z-z_0|^2$ $\mathrm{\mu m}^2$")
ax10[0, 1].set_ylabel("Residuals")
ax10[0, 1].set_title("Axial motion, atmospheric pressure")

ax10[1, 0].errorbar(y_lv_v[1][0:30], r_lv_v, np.sqrt(y_lv_v[0]), fmt='.')
ax10[1, 0].set_xlabel("$|y-y_0|^2$ $\mathrm{\mu m}^2$")
ax10[1, 0].set_ylabel("Residuals")
ax10[1, 0].set_title("Vertical motion, rough vacuum")

ax10[1, 1].errorbar(y_lv_ax[1][0:30], r_lv_ax, np.sqrt(y_lv_ax[0]), fmt='.')
ax10[1, 1].set_xlabel("$|z-z_0|^2$ $\mathrm{\mu m}^2$")
ax10[1, 1].set_ylabel("Residuals")
ax10[1, 1].set_title("Axial motion, rough vacuum")

ax10[2, 0].errorbar(y_hv_v[1][0:30], r_hv_v, np.sqrt(y_hv_v[0]), fmt='.')
ax10[2, 0].set_xlabel("$|y-y_0|^2$ $\mathrm{\mu m}^2$")
ax10[2, 0].set_ylabel("Residuals")
ax10[2, 0].set_title("Vertical motion, high vacuum")

ax10[2, 1].errorbar(y_hv_ax[1][0:30], r_hv_ax, np.sqrt(y_hv_ax[0]), fmt='.')
ax10[2, 1].set_xlabel("$|z-z_0|^2$ $\mathrm{\mu m}^2$")
ax10[2, 1].set_ylabel("Residuals")
ax10[2, 1].set_title("Axial motion, high vacuum")

#chi-squared
chi2_atm_v = np.sum((r_atm_v/np.sqrt(y_atm_v[0][5:30]))**2)
chi2red_atm_v = chi2_atm_v/(len(y_atm_v[1])-2)
chi2_atm_ax = np.sum((r_atm_ax/np.sqrt(y_atm_ax[0]))**2)
chi2red_atm_ax = chi2_atm_ax/(len(y_atm_ax[1])-2)

chi2_lv_v = np.sum((r_lv_v/np.sqrt(y_lv_v[0]))**2)
chi2red_lv_v = chi2_lv_v/(len(y_lv_v[1])-2)
chi2_lv_ax = np.sum((r_lv_ax/np.sqrt(y_lv_ax[0]))**2)
chi2red_lv_ax = chi2_lv_ax/(len(y_lv_ax[1])-2)

chi2_hv_v = np.sum((r_hv_v/np.sqrt(y_hv_v[0]))**2)
chi2red_hv_v = chi2_hv_v/(len(y_hv_v[1])-2)
chi2_hv_ax = np.sum((r_hv_ax/np.sqrt(y_hv_ax[0]))**2)
chi2red_hv_ax = chi2_hv_ax/(len(y_hv_ax[1])-2)

chi2red_v = np.array([chi2red_atm_v, chi2red_lv_v, chi2red_hv_v])
chi2red_ax = np.array([chi2red_atm_ax, chi2red_lv_ax, chi2red_hv_ax])

print(chi2red_v, '\n', chi2red_ax)


#quality factor

def Q(fr, fft, nufft):
    #finds quality factor Q = fr/delta(f)
    #where fr is the resonant frequency
    #and delta(f) is the full width at half maximum of that peak
    #fft should be the actual power spectrum
    
    #first find FWHM of fft data
    difference = max(fft[(fr-10)*10**2:(fr+10)*10**2]) - min(fft[(fr-10)*10**2:(fr+10)*10**2])
    HM = difference/2
    
    nearest = (np.abs(fft[(fr-10)*10**2:(fr+10)*10**2]-HM)).argmin()
    HWHM = np.abs(nearest - min(fft[(fr-10)*10**2:(fr+10)*10**2]))
    FWHM = HWHM*2
    
    return(fr/FWHM)
    
pressure = np.array([760, 120e-3, 1e-7])
Q_v = np.zeros(3)
Q_v[0] = Q(w_v, fatm_v, nuatm_v)
Q_v[1] = Q(w_v, flv_v, nulv_v)
Q_v[2] = Q(w_v, fhv_v, nuhv_v)

Q_ax = np.zeros(3)
Q_ax[0] = Q(w_ax, fatm_ax, nuatm_ax)
Q_ax[1] = Q(w_ax, flv_ax, nulv_ax)
Q_ax[2] = Q(w_ax, fhv_ax, nuhv_ax)

Q_tot = np.zeros(3)
for i in range(len(Q_tot)):
    Q_tot[i] = (Q_v[i] + Q_ax[i])/2

plt.figure(figsize = (8, 6))
plt.plot(pressure, Q_tot, 'o')
plt.xlabel("Pressure (Torr)")
plt.ylabel("Quality factor (Q)")
plt.title("Quality factor vs. pressure")
plt.xscale("log")
plt.show()

plt.figure(figsize = (8, 6))
plt.plot(pressure, Q_v, 'o')
plt.xlabel("Pressure (Torr)")
plt.ylabel("Quality factor (Q)")
plt.title("Vertical quality factor vs. pressure")
plt.xscale("log")
plt.show()

plt.figure(figsize = (8, 6))
plt.plot(pressure, Q_ax, 'o')
plt.xlabel("Pressure (Torr)")
plt.ylabel("Quality factor (Q)")
plt.title("Axial quality factor vs. pressure")
plt.xscale("log")
plt.show()