import numpy as np

class Chain(object):
    def __init__(self, power, app_factor, ratio, drive_speed):
        self.power = power
        self.app_factor = app_factor
        self.ratio = ratio
        self. drive_speed = drive_speed
        self.equivalent_power = self.power * self.app_factor
        pass
