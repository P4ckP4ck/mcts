import numpy as np
import pandas as pd
import gym

from model import VPPBEV, VPPCombinedHeatAndPower, VPPEnergyStorage, VPPHeatPump, VPPOperator, VPPPhotovoltaic, VPPThermalEnergyStorage, VPPWind
import time


class ComplexEMS:
    def __init__(self):
        self.cost_per_kwh = {"regular": 0.25, "PV": 0.08, "Wind": 0.12, "CHP": 0.15, "El_Store": 0.1}
        self.production_per_tech = {"regular": 0, "PV": 0, "Wind": 0, "CHP": 0, "El_Store": 0}
        self.tech = ["regular", "PV", "Wind", "CHP", "El_Store"]
        self.loadprofile = pd.read_csv("./Input_House/Base_Szenario/baseload_household.csv", delimiter="\t")["0"]
        self.heatpump = VPPHeatPump
        self.time = 0

    def init_heatpump(self):
        start = '2017-01-01 00:00:00'
        end = '2017-01-14 23:45:00'
        freq = "15 min"
        timestamp_int = 48
        timestamp_str = '2017-01-01 12:00:00'

        hp = VPPHeatPump(identifier="House 1", timebase=1, heatpump_type="Air",
                         heat_sys_temp=60, environment=None, userProfile=None,
                         heatpump_power=10.6, full_load_hours=2100, heat_demand_year=None,
                         building_type='DE_HEF33', start=start,
                         end=end, year='2017')
        return hp

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
        pass

    def step(self):
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
        2: CHP on/off
        3: BEV on/off (Only on?)

        (3: Electrical Storage (?) Wie modelliert man Netzverträglichkeit?
        Ansonsten macht es keinen Sinn Speicher bei Überschuss nicht zu laden!)

        states:
        (time?), current_demand, current_production, thermal_storage_temp, (electrical_storage?), current_residual

        reward shaping:
        main component = negative price of current mix
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
        dismiss wind

        todo:
        calculate min/max of forecast
        calculate sizes of all techs
        :return:
        """

        # step 1: calculate all demands
        loadprofile = self.loadprofile.iloc[self.time]
        heatpump = self.heatpump.get_
        current_demand = loadprofile + heatpump + bev
        current_production = pv + wind + chp
        current_
        thermal_storage
        electrical_storage
        current_residual
        validity_check


    def step_forecast(self):
        pass


def time_func(func, rounds=10000):
    tack = time.time()
    for _ in range(rounds):
        func()
    tick = time.time()
    print(tick-tack)


if __name__ == "__main__":
    env = ComplexEMS()
    a = env.calc_cost_current_mix()
    time_func(env.calc_cost_current_mix)