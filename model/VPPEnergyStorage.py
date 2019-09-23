"""
Info
----
This file contains the basic functionalities of the VPPEnergyStorage class.

"""

from .VPPComponent import VPPComponent
import numpy as np

class VPPEnergyStorage(VPPComponent):

    def __init__(self, timebase, capacity, chargeEfficiency, dischargeEfficiency, maxPower, maxC, environment = None, userProfile = None):
        
        """
        Info
        ----
        The class "VPPEnergyStorage" adds functionality to implement an 
        electrical energy storage to the virtual power plant.
        
        
        Parameters
        ----------
        
        capacity [kWh]
        chargeEfficiency [-] (between 0 and 1)
        dischargeEfficiency [-] (between 0 and 1)
        maxPower [kW]
        maxC [-]
        	
        Attributes
        ----------
        
        The stateOfCharge [kWh] is set to zero by default.
        
        Notes
        -----
        
        ...
        
        References
        ----------
        
        ...
        
        Returns
        -------
        
        ...
        
        """

        # Call to super class
        super(VPPEnergyStorage, self).__init__(timebase, environment, userProfile)


        # Setup attributes
        self.capacity = capacity
        self.chargeEfficiency = chargeEfficiency
        self.dischargeEfficiency = dischargeEfficiency
        self.maxPower = maxPower
        self.maxC = maxC
        self.timebase = timebase
        self.stateOfCharge = 0





    def prepareTimeSeries(self):
    
        self.timeseries = np.zeros(int(365*24*60/self.timebase))




    # ===================================================================================
    # Observation Functions
    # ===================================================================================

    def observationsForTimestamp(self, timestamp):
        
        pass
        # TODO: Implement dataframe to return state of charge

    # ===================================================================================
    # Controlling functions
    # ===================================================================================

    def charge(self, energy, timebase, timestamp):
        
        """
        Info
        ----
        This function takes the energy [kWh] that should be charged and the timebase as
        parameters. The timebase [minutes] is neccessary to calculate if the maximum
        power is exceeded.
        
        Parameters
        ----------
        
        ...
        	
        Attributes
        ----------
        
        ...
        
        Notes
        -----
        
        ...
        
        References
        ----------
        
        ...
        
        Returns
        -------
        
        ...
        
        """

        # Check if power exceeds max power
        power = energy / (timebase / 60)
        is_viable = True
        
        if power > self.maxPower * self.maxC:
            energy = (self.maxPower * self.maxC) * (timebase / 60)


        # Check if charge exceeds capacity
        if self.stateOfCharge + energy * self.chargeEfficiency > self.capacity:
            energy = (self.capacity - self.stateOfCharge) * (1 / self.chargeEfficiency)
            is_viable = False

        # Update state of charge
        self.stateOfCharge += energy * self.chargeEfficiency
        if self.stateOfCharge > self.capacity:
            self.stateOfCharge = self.capacity
            is_viable = False
        
        # Check if data already exists
#        if self.timeseries[timestamp] == None:
#            self.append(energy)
#        else:
#            self.timeseries[timestamp] = energy
        return is_viable

    def discharge(self, energy, timebase, timestamp):
        
        """
        Info
        ----
        This function takes the energy [kWh] that should be discharged and the timebase as
        parameters. The timebase [minutes] is neccessary to calculate if the maximum
        power is exceeded.
        
        Parameters
        ----------
        
        ...
        	
        Attributes
        ----------
        
        ...
        
        Notes
        -----
        
        ...
        
        References
        ----------
        
        ...
        
        Returns
        -------
        
        ...
        
        """

        # Check if power exceeds max power
        power = energy / (timebase / 60)
        is_viable = True
        if power > self.maxPower * self.maxC:
            energy = (self.capacity - self.stateOfCharge) * (1 / self.chargeEfficiency)


        # Check if discharge exceeds state of charge
        if self.stateOfCharge - energy * (1 / self.dischargeEfficiency) < 0:
            energy = self.stateOfCharge * self.dischargeEfficiency
            is_viable = False

        # Update state of charge
        self.stateOfCharge -= energy * (1 / self.dischargeEfficiency)
        if self.stateOfCharge < 0:
            self.stateOfCharge = 0
            is_viable = False
        
        
        # Check if data already exists
#        if self.timeseries[timestamp] == None:
#            self.append(energy)
#        else:
#            self.timeseries[timestamp] = energy
        return is_viable





    # ===================================================================================
    # Balancing Functions
    # ===================================================================================

    # Override balancing function from super class.
    def valueForTimestamp(self, timestamp):

        return self.timeseries[timestamp]
