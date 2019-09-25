import time
from collections import deque

import numpy as np
import pandas as pd

from model.VPPBEV import VPPBEV
from model.VPPEnergyStorage import VPPEnergyStorage
from model.VPPHeatPump import VPPHeatPump
from model.VPPPhotovoltaic import VPPPhotovoltaic
from model.VPPThermalEnergyStorage import VPPThermalEnergyStorage
from model.VPPUserProfile import VPPUserProfile

log_global = deque(maxlen=50)
col_indexes = ["temperature", "el_loadprofile", "th_loadprofile", "heatpump", "bev_at_home", "bev_battery", "bev", "pv", "current_demand",
               "current_production", "temp_residual", "current_residual", "electrical_storage_charge",
               "electrical_storage_discharge", "done", "action", "reward", "bev_charge_flag", "heatpump_flag", "thermal_storage", "th_soc"]
def log_res(log):
    global log_global
    log_global.append(log)
    if len(log_global) == 50:
        pd.DataFrame(log_global, columns=col_indexes).to_csv("LOG_GLOBAL.csv")
        log_global = deque(maxlen=50)

class ComplexEMS:
    def __init__(self, ep_len=96):
        super(ComplexEMS, self).__init__()
        # gym declarations
        self.obs = 6
        self.ep_len = ep_len

        # operational variables
        self.time = 0
        self.cost_per_kwh = {"regular": 0.25, "PV": 0.08, "Wind": 0.12, "CHP": 0.15, "El_Store": 0.1}
        self.production_per_tech = {"regular": 0, "PV": 0, "Wind": 0, "CHP": 0, "El_Store": 0}
        self.tech = ["regular", "PV", "Wind", "CHP", "El_Store"]
        self.el_loadprofile = pd.read_csv("./Input_House/Base_Szenario/baseload_household.csv",
                                          delimiter="\t")["0"]/1000
        self.th_loadprofile = self.init_up()
        self.temperature = self.init_temperature()
        self.heatpump = self.init_heatpump(10)
        self.heatpump_flag = 0
        self.bev = self.init_bev()
        self.bev_charge_flag = 0
        self.pv = self.init_pv()
        self.el_storage = self.init_el_storage()
        self.th_storage = self.init_th_storage()


    def init_temperature(self):
        temperature = pd.read_csv('./Input_House/heatpump_model/mean_temp_hours_2017_indexed.csv', index_col="time")
        df = pd.DataFrame(index = pd.date_range("2017", periods=35040, freq='15min', name="time"))
        temperature.index = pd.to_datetime(temperature.index)
        df["quart_temp"] = temperature
        df.interpolate(inplace = True)
        return df

    def init_heatpump(self, power):
        start = '2017-01-01 00:00:00'
        end = '2017-12-31 23:45:00'
        rampUpTime = 1/15 #timesteps
        rampDownTime = 1/15 #timesteps
        minimumRunningTime = 1 #timesteps
        minimumStopTime = 2 #timesteps
        timebase = 15

        hp = VPPHeatPump(identifier="hp_1", timebase=timebase, heatpump_type="Air",
                         heat_sys_temp=60, environment=None, userProfile=None,
                         rampDownTime=rampDownTime, rampUpTime=rampUpTime,
                         minimumRunningTime=minimumRunningTime, minimumStopTime=minimumStopTime,
                         heatpump_power=power, full_load_hours=2100, heat_demand_year=None,
                         building_type='DE_HEF33', start=start, end=end, year='2017')

        hp.lastRampUp = self.time
        hp.lastRampDown = self.time
        return hp

    def init_bev(self):
        start = '2017-01-01 00:00:00'
        end = '2017-12-31 23:45:00'

        bev = VPPBEV(timebase=15/60, identifier='bev_1',
                     start = start, end = end, time_freq = "15 min",
                     battery_max = 16, battery_min = 0, battery_usage = 1,
                     charging_power = 11, chargeEfficiency = 0.98,
                     environment=None, userProfile=None)
        bev.prepareTimeSeries()
        return bev

    def init_pv(self):
        latitude = 50.941357
        longitude = 6.958307
        name = 'Cologne'

        weather_data = pd.read_csv("./Input_House/PV/2017_irradiation_15min.csv")
        weather_data.set_index("index", inplace = True)

        pv = VPPPhotovoltaic(timebase=15, identifier=name, latitude=latitude, longitude=longitude, modules_per_string=5, strings_per_inverter=1)
        pv.prepareTimeSeries(weather_data)
        pv.timeseries.fillna(0, inplace=True)
        return pv

    def init_el_storage(self):
        es = VPPEnergyStorage(15, "el_store_1", 25, 0.9, 0.9, 5, 1)
        # es.prepareTimeSeries()
        return es

    def init_up(self):
        yearly_heat_demand = 2500# kWh
        target_temperature = 60# °C
        up = VPPUserProfile(heat_sys_temp=target_temperature,
        yearly_heat_demand=yearly_heat_demand, full_load_hours=2100)
        up.get_heat_demand()
        return up.heat_demand

    def init_th_storage(self):
        #Values for Thermal Storage
        target_temperature = 60 # °C
        hysteresis = 5 # °K
        mass_of_storage = 300 # kg
        timebase = 15

        ts = VPPThermalEnergyStorage(timebase, mass=mass_of_storage, hysteresis=hysteresis,
                                     target_temperature=target_temperature)#, userProfile=self.th_loadprofile)
        return ts

    # def calc_cost(self, tech):
    #     return self.cost_per_kwh[tech] * self.production_per_tech[tech], self.production_per_tech[tech]
    #
    # def calc_cost_current_mix(self):
    #     costs_production = np.array(list(map(self.calc_cost, self.tech)))
    #     cost_sum, production_sum = np.sum(costs_production, axis=0)
    #     return cost_sum/production_sum

    def calc_cost_current_mix(self):
        cost, prod = 0, 0
        for tech in self.cost_per_kwh:
            cost += self.cost_per_kwh[tech] * self.production_per_tech[tech]
            prod += self.production_per_tech[tech]
        return cost/prod

    def reset(self):
        self.rand_start = int(np.random.rand()*(35040-self.ep_len))
        state = np.array(np.zeros(self.obs))
        self.time = self.rand_start
        self.heatpump.lastRampUp = self.time
        self.heatpump.lastRampDown = self.time
        self.variables = {"heatpump_flag": self.heatpump_flag, "store_temp": 0.5, "el_store": 0,
                     "bev_battery": 0, "bev_charge_flag": self.bev_charge_flag, "time": self.time}
        return state

    def step(self, action):
        """
        demand - non controllable     = LoadProfile
        production - non controllable = Photovoltaic, Wind
        demand - controllable         = Heatpump, BEV
        production - controllable     = CombinedHeatAndPower

        cost per kWh:
        regular mix             = 0.25 €/kWh
        Photovoltaic            = 0.08 €/kWh
        Wind                    = 0.12 €/kWh
        CombinedHeatAndPower    = 0.15 €/kWh
        Electrical Storage      = 0.10 €/kWh

        possibilities:
        reward_1 = current mixed price <-- way to go
        reward_2 = mixed price per episode

        actions:
        0: Nothing
        1: Heatpump on/off
        2: BEV on

        3: CHP on/off
        (4: Electrical Storage (?) Wie modelliert man Netzverträglichkeit?
        Ansonsten macht es keinen Sinn Speicher bei Überschuss nicht zu laden!)

        states:
        (time?), current_demand, current_production, thermal_storage_temp, (electrical_storage?),
        current_residual, bev at home

        reward shaping:
        main component = negative price of current mix, normalized between -0.08 and -0.25
        th_storage = gets negative as temp drops/rises from hysteresis, normal = 0
        bev = gets negative if user leaves with less than 99% - 90% charge, normal = 0
        invalid actions = -10 e.g. turning off heatpump too early

        problems:
        chp + heatpump need to be specifically configured for heat demand
        how to determine deviation of forecast to not achieve perfect forecast but 80% or sth.
        NaN's in pv
        Forecasting over several techs, storage temp, residual, bev etc.
        forecasting temp needs to be done iteratively
        forecasting bev and residual should be deterministic
        Is search depth limitation necessary?

        simplifications for first build:
        dismiss wind and chp

        todo:
        calculate min/max of forecast
        calculate sizes of all techs
        Implement BEV reward --> not fully loaded etc

        :return:
        """
        # step 1: apply actions
        heatpump_action, bad_action = 0, False
        if action == 1:
            # if self.heatpump_flag == 0:
            self.heatpump_flag= 1
            # else:
            #     self.heatpump_flag = 0
            heatpump_action = 1
        if action == 2:
            # if self.bev_charge_flag == 1:
            #     bad_action = True
            self.bev_charge_flag = 1

        if action == 3:
            self.bev_charge_flag = 1
            self.heatpump_flag= 1
            heatpump_action = 1

        # step 1: calculate all demands and productions
        temperature = self.temperature.iat[self.time, 0]
        el_loadprofile = self.el_loadprofile.iat[self.time]/1000
        th_loadprofile = self.th_loadprofile.iat[self.time, 0]
        heatpump = (self.heatpump.heatpump_power / self.heatpump.get_current_cop(temperature)) * self.heatpump_flag
        bev_at_home = self.bev.at_home.iat[self.time, 0]
        bev_battery, bev, self.bev_charge_flag = self.bev.charge_timestep(bev_at_home, self.bev_charge_flag)
        pv = self.pv.timeseries.iat[self.time, 0]*5
        current_demand = el_loadprofile + heatpump + bev
        current_production = pv
        temp_residual = current_demand - current_production

        #step 2. calculate storage states
        electrical_storage_charge, electrical_storage_discharge = 0, 0
        thermal_storage, th_soc, bad_action = self.operate_th_storage(th_loadprofile, heatpump_action)

        if temp_residual < 0 and (self.el_storage.stateOfCharge/self.el_storage.capacity) != 1:
            electrical_storage_charge = np.clip(abs(temp_residual), 0, self.el_storage.maxPower)
            self.el_storage.charge(electrical_storage_charge, 15, self.time)

        if temp_residual > 0 and (self.el_storage.stateOfCharge/self.el_storage.capacity) != 0:
            electrical_storage_discharge = np.clip(temp_residual, 0, self.el_storage.maxPower)
            self.el_storage.discharge(electrical_storage_discharge, 15, self.time)

        # step 3: calculate residual load
        current_demand += electrical_storage_charge
        current_production += electrical_storage_discharge
        current_residual = current_demand - current_production

        # step 4: calculate validity and reward
        self.production_per_tech = {"regular": current_residual, "PV": pv, "Wind": 0,
                                    "CHP": 0, "El_Store": electrical_storage_discharge}
        if current_residual < 0:
            self.production_per_tech = {"regular": 0, "PV": pv, "Wind": 0,
                                        "CHP": 0, "El_Store": electrical_storage_discharge}

        reward = 1-np.clip(self.calc_cost_current_mix()*4, 0, 1)

        if thermal_storage < 55 or thermal_storage > 65:
            bad_action = True

        if self.bev.at_home.iat[self.time-1, 0] == 0 and bev_at_home == 1:
            if bev_battery != 1:
                bad_action = True

        if bad_action:
            reward = -1

        # step 5: calculate states
        # normalizing factors:
        # maximum expected residual load = heatpump + bev + el_loadprofile = 5kW + 11kW + ~10kW = 26 kW
        # th_storage temp = 55 - 65 °C
        self.variables = {"heatpump_flag": self.heatpump_flag, "store_temp": th_soc, "el_store": self.el_storage.stateOfCharge,
                     "bev_battery": bev_battery, "bev_charge_flag": self.bev_charge_flag, "time": self.time}
        done = self.time >= self.rand_start + self.ep_len
        state = np.array([current_residual/26, (thermal_storage-55)/10, bev_at_home, bev_battery,
                          self.bev_charge_flag, self.heatpump_flag])
        self.time += 1
        # log = [temperature, el_loadprofile, th_loadprofile, heatpump, bev_at_home, bev_battery, bev, pv, current_demand,
        #        current_production, temp_residual, current_residual, electrical_storage_charge,
        #        electrical_storage_discharge, done, action, reward, self.bev_charge_flag, self.heatpump_flag,
        #        thermal_storage, th_soc]
        # log_res(log)
        return state, reward, done, self.variables

    def step_forecast(self, action, variables):
        # step 1: apply actions
        time = variables["time"]
        heatpump_action, bad_action = 0, False
        if action == 1:
            # if variables["heatpump_flag"] == 0:
            variables["heatpump_flag"] = 1
            # else:
            #     variables["heatpump_flag"] = 0
            heatpump_action = 1
        if action == 2:
            # if variables["bev_charge_flag"] == 1:
            #     bad_action = True
            variables["bev_charge_flag"] = 1
        if action == 3:
            variables["bev_charge_flag"] = 1
            variables["heatpump_flag"] = 1
            # else:
            #     variables["heatpump_flag"] = 0
            heatpump_action = 1

        # step 1: calculate all demands and productions
        temperature = self.temperature.iat[time, 0]
        el_loadprofile = self.el_loadprofile.iat[time]/1000
        th_loadprofile = self.th_loadprofile.iat[time, 0]
        heatpump = (self.heatpump.heatpump_power / self.heatpump.get_current_cop(temperature)) * heatpump_action
        bev_at_home = self.bev.at_home.iat[time, 0]
        bev_battery, bev, variables["bev_charge_flag"] = self.bev.charge_forecast(bev_at_home, variables["bev_charge_flag"], variables["bev_battery"])
        pv = self.pv.timeseries.iat[time, 0]*5
        current_demand = el_loadprofile + heatpump + bev
        current_production = pv
        temp_residual = current_demand - current_production

        #step 2. calculate storage states
        electrical_storage_charge, electrical_storage_discharge = 0, 0
        thermal_storage, th_soc, bad_action = self.forecast_th_storage(th_loadprofile, heatpump_action, variables)
        soc = variables["el_store"]
        if temp_residual < 0 and variables["el_store"]/self.el_storage.capacity != 1:
            electrical_storage_charge = np.clip(abs(temp_residual), 0, self.el_storage.maxPower)
            soc = self.el_storage.forecast_charge(electrical_storage_charge, 15, variables["el_store"])

        if temp_residual > 0 and variables["el_store"]/self.el_storage.capacity != 0:
            electrical_storage_discharge = np.clip(temp_residual, 0, self.el_storage.maxPower)
            soc = self.el_storage.forecast_charge(electrical_storage_discharge, 15, variables["el_store"])

        # step 3: calculate residual load
        current_demand += electrical_storage_charge
        current_production += electrical_storage_discharge
        current_residual = current_demand - current_production

        # step 4: calculate validity and reward
        self.production_per_tech = {"regular": current_residual, "PV": pv, "Wind": 0,
                                    "CHP": 0, "El_Store": electrical_storage_discharge}
        if current_residual < 0:
            self.production_per_tech = {"regular": 0, "PV": pv, "Wind": 0,
                                        "CHP": 0, "El_Store": electrical_storage_discharge}

        reward = 1-np.clip(self.calc_cost_current_mix()*4, 0, 1)

        if thermal_storage < 55 or thermal_storage > 65:
            bad_action = True

        if self.bev.at_home.iat[time-1, 0] == 0 and bev_at_home == 1:
            if bev_battery != 1:
                bad_action = True

        if bad_action:
            reward = -1

        # step 5: calculate states
        # normalizing factors:
        # maximum expected residual load = heatpump + bev + el_loadprofile = 5kW + 11kW + ~10kW = 26 kW
        # th_storage temp = 55 - 65 °C

        done = time >= self.rand_start + self.ep_len

        state = np.array([current_residual/26, (thermal_storage-55)/10, bev_at_home, bev_battery, variables["bev_charge_flag"], variables["heatpump_flag"]])
        time += 1
        variables = {"heatpump_flag": variables["heatpump_flag"], "store_temp": th_soc, "el_store": soc,
                     "bev_battery": bev_battery, "bev_charge_flag": variables["bev_charge_flag"], "time": time}


        return state, reward, done, variables

    def operate_th_storage(self, heat_demand, hp_action):
        feedback = True
        # if hp_action:
        #     if self.heatpump_flag:
        #         feedback = self.heatpump.rampUp(self.time)
        #     else:
        #         feedback = self.heatpump.rampDown(self.time)
        #     if feedback is None:
        #         feedback = True
        if hp_action:
            heat_production = self.heatpump.heatpump_power
        else:
            heat_production = 0
        temp, soc = self.th_storage.operate_storage_reinforcement(heat_demand, heat_production)
        return temp, soc, not feedback

    def forecast_th_storage(self, heat_demand, hp_action, vars):
        feedback = True
        # if hp_action:
        #     if variables["heatpump_flag"]:
        #         feedback = self.heatpump.rampUp(self.time)
        #     else:
        #         feedback = self.heatpump.rampDown(self.time)
        #     if feedback is None:
        #         feedback = True
        if hp_action:
            heat_production = self.heatpump.heatpump_power
        else:
            heat_production = 0
        temp, soc = self.th_storage.forecast_storage_reinforcement(heat_demand, heat_production, vars["store_temp"])
        return temp, soc, not feedback

def time_func(func, rounds=10000):
    tack = time.time()
    for _ in range(rounds):
        func()
    tick = time.time()
    print(tick-tack)

def x():
    return self.temperature.iat[10, 0]

if __name__ == "__main__":
    env = ComplexEMS()
    self = env
    # a = env.calc_cost_current_mix()
    time_func(x)
