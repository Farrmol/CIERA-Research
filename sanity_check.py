import numpy as np
from orbitize import driver
import matplotlib.pyplot as plt

r = 50 # [mas]
e = 0.3
sma = 5/3 # [au]
inc = np.radians(30) # [radians]
Omega = np.radians(60) # [radians]
omega = np.radians(120) # [radians]
plx = 30 # [mas]
albedo= 0.5
tau=0
tau_ref_epoch = 0 # [mjd]
total_mass = 4.632800608828007 # [Msol]
num_secondary_bodies = 1
filename = "simulated_ra_dec_data.csv"

my_driver = driver.Driver(
    filename,
    "MCMC",
    num_secondary_bodies,
    total_mass,
    plx,
    mass_err=0,
    plx_err=0,
    system_kwargs={"tau_ref_epoch":tau_ref_epoch},
)

params_arr = np.array([
    sma, e, inc, omega, Omega, tau, plx, total_mass
])
params_arr = params_arr.reshape((8,1))

ra, dec, _, _ = my_driver.system.compute_all_orbits(params_arr,  my_driver.system.data_table['epoch'])

fig, ax = plt.subplots(3,1)
ax[0].plot(ra[:,1,0], dec[:,1,0])
ax[0].scatter(my_driver.system.data_table['quant1'], my_driver.system.data_table['quant2'])
ax[0].set_xlabel('$\Delta$ra [mas]')
ax[0].set_ylabel('$\Delta$dec [mas]')

ax[1].plot(my_driver.system.data_table['epoch'], dec[:,1,0])
ax[1].scatter(my_driver.system.data_table['epoch'], my_driver.system.data_table['quant2'])
ax[1].set_xlabel('time [mjd]')
ax[1].set_ylabel('$\Delta$dec [mas]')

ax[2].plot(my_driver.system.data_table['epoch'], ra[:,1,0])
ax[2].scatter(my_driver.system.data_table['epoch'], my_driver.system.data_table['quant1'])
ax[2].set_xlabel('time [mjd]')
ax[2].set_ylabel('$\Delta$ra [mas]')

plt.tight_layout()
plt.savefig('sanity_check.png', dpi=250)