#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  PyesianBlocks.py
#  
#  Copyright 2023 Alexis Andrés <alexis.andres1210@gmail.com>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  
"""
================================================================================================
This is a simple Python module to obtain the bayesian blocks representation for binned data.
It is written specifically to work with any kind of light curve data.

Notes
-----
See Scargle  et al. (2013): https://ui.adsabs.harvard.edu/abs/2013ApJ...764..167S/abstract
for more details on the Bayesian Blocks algorithm
================================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


class BayesianBlocks():
    """
    ==============================================================================================
    A class to obtain the bayesian blocks representation for binned data
    
    Attributes
    ----------
    #-# time: array
        The array of the independent variable
        
    #-# flux: array
        The array of the dependent variable
        
    Methods
    -------
    #-# get_prior():
        Get the prior of the dataset
        
    #-# get_fitness():
        Computes the likelihood of the dataset
        
    #-# get_changepoints():
        Obtains the change points of the bayesian blocks
        
    #-# plot_blocks():
        Computes flux of bayesian blocks and plots results
        
    ==============================================================================================
    """
    
    def __init__(self, time, flux):
        """
        Initialise the bayesian blocks object
        
        Parameters
        ----------
        #-# time: array
            The array of the independent variable
        
        #-# flux: array
            The array of the dependent variable
        """
        
        #- Reading and setting data
        self.time = time
        self.flux = flux
        
        #- Get the power of 10 of the minimum 
        n = float(('{:.5e}'.format(np.min(self.flux))).split('e')[1])
        
        #- Synthetic (renormalized) flux
        self.n_flux = self.flux*10**(-n)
            
        
    def get_prior(self, fp_rate=0.05, ncp_prior=None):
        """
        Gets the prior of the dataset
        
        Parameters
        ----------
        #-# fp_rate: float
            The false positive rate (default 0.05)
            
        #-# ncp_prior: float
            The number of change points prior (default None)
        """
        
        #- Getting length of data
        self.num_points = len(self.time)
        
        #- Setting false positive rate
        self.fp_rate = fp_rate
        
        #- Setting the prior for the number of change points
        if ncp_prior is not None:
            self.ncp_prior = ncp_prior
            
        else:
            self.ncp_prior = 4 - np.log( self.fp_rate / ( 0.0136 * self.num_points ** (0.478) ) )
            
        print('A change point prior of {} was obtained using a fp_rate of {}'.format(self.ncp_prior, self.fp_rate))
        
        
    def get_fitness(self, t_start=None, t_stop=None):
        """        
        Computes the likelihood of the dataset
        
        Parameters
        ----------
        #-# t_start: int or float
            The start time of the analysis (default None)
            
        #-# t_stop: int or float
            The stop time of the analysis (default None)
        """
        
        #- Time array separation
        time_sep        = np.diff(self.time)
        time_sep_median = np.median(time_sep)
        
        #- Setting the start time
        if t_start is not None:
            t_start = [t_start]
            
        else:
            #- Default start time
            t_start = [self.time[0] - 0.5*time_sep_median]
            
        #- Setting the stop time
        if t_stop is not None:
            t_stop = [t_stop]
            
        else:
            #- Default start time
            t_stop = [self.time[-1] + 0.5*time_sep_median]
        
        
        #- Getting length of the blocks
        block_length = np.concatenate((t_start, 0.5*(self.time[1:] + self.time[:-1]), t_stop))
        block_length = t_stop - block_length
        
        #- Empty arrays to store likelihood
        self.best = np.zeros(self.num_points)
        self.last = np.zeros(self.num_points, dtype=np.int64)
        
        #- Loop to find the change points
        for i in range(self.num_points):
            
            #- Width of the i-th block
            width = block_length[:i+1] - block_length[i+1]
            width[width == 0.] = np.inf 
            
            #- All counts in this bin
            nn_cum_vec = np.cumsum(self.n_flux[i::-1])[::-1]
            
            #- Fitness function for binned data
            fit_vec      = nn_cum_vec*(np.log(nn_cum_vec) - np.log(width))
            fit_vec      = fit_vec - self.ncp_prior
            fit_vec[1:] += self.best[:i]
            
            #- Maximum of fitness at the i-th change point
            i_max        = np.argmax(fit_vec)
            self.best[i] = fit_vec[i_max]
            self.last[i] = i_max
            

    def get_changepoints(self):
        """
        Obtains the change points of the bayesian blocks
        """
        
        #- Find changepoints by iteratively peeling off the last block
        index = self.last[-1]
        self.change_points = []
        
        while index > 0:
            self.change_points.append(index)
            index = self.last[index - 1]
            
        #- Reversing the change_points
        self.change_points = self.change_points[::-1]
        
        #- Length of change points
        self.num_change_points = len(self.change_points)
        self.num_blocks        = self.num_change_points + 1
        
        print('Change points obtained')
        print('{} blocks were found'.format(self.num_blocks))
            
    
    def plot_blocks(self, c='red', lw='1.5', ls='-', label='Bayesian blocks representation'):
        """
        Computes flux of bayesian blocks and plots results
        
        Parameters
        ----------
        #-# c: str
            Color for plotting the blocks
            
        #-# lw: int or float
            Linewidth for the blocks
            
        #-# ls: str
            Linestyle for the blocks
            
        #-# label: str
            Label for the blocks plot
        """
        
        #- Time values of the change points
        cp_times = self.time[self.change_points]
        cpt_use  = np.concatenate(([np.min(self.time)], cp_times, [np.max(self.time)]))
        
        #- Block's number
        num_blocks    = len(cpt_use) - 1
        self.flux_vec = np.zeros(num_blocks)
        
        #- Initial flux value
        flux_old = 0
        
        #- Loop to compute the mean flux in each block
        for id_block in range(num_blocks):
            
            #- Initial and final time of the i-th bolck
            tt_1 = cpt_use[id_block]
            tt_2 = cpt_use[id_block + 1]
            
            #- Index of initial time of block
            ii_start = np.where(tt_1 <= self.time)[0]
            ii_start = ii_start[0]
            
            #- Index of ending time of block
            ii_end   = np.where(self.time[1:] > tt_2)[0]
        
            if not ii_end.size:
                ii_end = len(self.time)
            
            else:
                ii_end = ii_end[0] - 1
            
            #- Array of indexes corresponding to the i-th block
            iu = np.arange(ii_start, ii_end + 1)
        
            #- Computting the mean flux at the i-th block
            if len(iu) == 1:
                #- First block
                dt_use    = self.time[ii_start + 1] - self.time[ii_start]
                flux_mean = np.mean(self.flux[iu])
            
            else:
            
                if ii_end == len(self.time):
                    #- Last block
                    dt_use    = self.time[ii_end - 1] - self.time[ii_start - 1]
                    flux_mean = np.mean(self.flux[iu[:-1]])
                
                else:
                    #- All other blocks
                    dt_use    = self.time[ii_end +1] - self.time[ii_start]
                    flux_mean = np.mean(self.flux[iu])
            
            #- Storing data
            self.flux_vec[id_block] = flux_mean
            
            #- Plotting blocks
            plt.plot((tt_1, tt_1), (flux_old, flux_mean),  c=c, lw=lw, ls=ls)
            plt.plot((tt_1, tt_2), (flux_mean, flux_mean), c=c, lw=lw, ls=ls)
            
            if id_block == num_blocks - 1:
                #- The edge of the last block
                plt.plot((tt_2, tt_2), (flux_mean, 0),  c=c, lw=lw, ls=ls, label=label)
        
            #- Updating last value of flux
            flux_old = flux_mean
