from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import Queue
import threading
import math
import numpy as np
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from scene_generator import *
try:
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME",
                                                os.path.join(os.path.dirname(__file__), '..')), "tools"))
    from sumolib import checkBinary  # noqa
    import traci
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")


from Tkinter import *

eventQueue = Queue.Queue()
VERBOSE = False
TS = 0.1

def compute_cot(x1, x2, v1, v2):
    if np.abs(v1-v2) < 1E-4:
        return 10
    return - (x1 - x2) / (v1 - v2)


def downKey(event):
    eventQueue.put('slow')
    if VERBOSE:
        print("Down key pressed")


def upKey(event):
    eventQueue.put('fast')
    if VERBOSE:
        print("Up key pressed")


def leftKey(event):
    eventQueue.put('up')
    if VERBOSE:
        print("Left key pressed")


def rightKey(event):
    eventQueue.put('down')
    if VERBOSE:
        print("Right key pressed")

def nextCar(event):
    eventQueue.put('next_car')
    if VERBOSE:
        print("N key pressed")

def prevCar(event):
    eventQueue.put('prev_car')
    if VERBOSE:
        print("P key pressed")

def updateXML(trial, ref):
    routes = {}
    routes["n_routes"] = 2
    routes["routes"] = [{"id": "exiting", "edges": "0 1 4"}, {"id": "entering", "edges": "3 1 4"}]
    with open(os.path.abspath(os.path.join(os.pardir)) + '/data/scenario/scenario_' + str(trial)) as f:
        scenario = json.load(f)
    xml_string = generateXMLstring(scenario, routes)

    outfile = os.path.abspath(os.path.join(os.pardir)) + "/sumo-files/lanechange_" + str(int(ref)) + ".rou.xml"

    with open(outfile, "w") as f:
        f.write(xml_string)
    print("Writing XML file at " + outfile)    


class AdoVehicleAutoControl:
    def __init__(self, gui, sumocfg, reference_value, gain):
        self.sumocfg = sumocfg
        self.reference_value = reference_value   
        self.K = gain
        self.a_max = 4
        self.a_min = -6
        if gui:
            self.cb = "sumo-gui"
        else:
            self.cb = "sumo"   

    def run(self):
        try:
            print("Trying to start traci")
            traci.start([checkBinary(self.cb), "-c", self.sumocfg])
            traci.simulationStep()
            self.running = True
            print("traci started")
        except traci.FatalTraCIError:
            self.running = False
            pass

        try:
            print("traci running")
            t = 0
            self.history = defaultdict(lambda: defaultdict(list))
            while self.running:
                if self.K == 0:
                    t += TS
                    traci.simulationStep()
                else:
                    veh_list = traci.vehicle.getIDList()
                    # for every vehicle in the scene, loop through
                    for veh in veh_list:
                        x, y = traci.vehicle.getPosition(veh)
                        self.history[veh]["x"].append(x)
                        self.history[veh]["y"].append(y)
                        angle = traci.vehicle.getAngle(veh)
                        self.history[veh]["angle"].append(angle)
                        speed = traci.vehicle.getSpeed(veh)
                        self.history[veh]["speed"].append(speed)
                        self.history[veh]["t"].append(t)
                        # if it is an exit vehicle, then we want to control is
                        if "enter" in veh:
                            
                            other_veh = "exit." + veh.split(".")[-1]
                            if other_veh not in veh_list: 
                                continue
                            if traci.vehicle.getLaneID(other_veh) <= traci.vehicle.getLaneID(veh):
                                continue
                            ado_speed = max(traci.vehicle.getSpeed(veh), 7)
                            # print(veh + " get speed: ", ado_speed)
                            other_speed = traci.vehicle.getSpeed(other_veh)
                            x, y = traci.vehicle.getPosition(veh)
                            other_x, other_y = traci.vehicle.getPosition(other_veh)
                            # apply the feedback controller on cot
                            cot = compute_cot(other_x, x, other_speed, ado_speed)
                            d_cot = self.reference_value - cot
                            a = self.K * d_cot * np.sign(other_speed - ado_speed)
                            a_bar = min(max(self.a_min, a), self.a_max)
                            # print(a, a_bar)
                            speed = ado_speed + TS*a_bar
                            traci.vehicle.setSpeed(veh, speed)
                            # print(veh + " set speed: ", speed)
                    t += TS
                    traci.simulationStep()

                if len(traci.vehicle.getIDList()) == 0:
                    self.running = False
                    self.history["status"] = 1
                if t > 80:
                    self.running = False
                    self.history["status"] = 2
                    

            print("traci closed")
            traci.close()
        except traci.FatalTraCIError:
            pass
        self.running = False


class AdoVehicleHumanControl:

    """
    Launch the main part of the GUI and the worker thread. periodicCall and
    endApplication could reside in the GUI part, but putting them here
    means that you have all the thread controls in a single place.
    """

    def __init__(self, master, sumocfg, egoID):
        self.master = master
        self.sumocfg = sumocfg
        self.egoID = egoID
        self.running = True

        self.thread = threading.Thread(target=self.workerThread)
        self.thread.start()
        self.type = "fast"
        # Start the periodic call in the GUI to see if it can be closed
        self.periodicCall()

    def periodicCall(self):
        if not self.running:
            sys.exit(1)
        self.master.after(100, self.periodicCall)

    def workerThread(self):
        try:
            traci.start([checkBinary("sumo-gui"), "-c", self.sumocfg,
                         # "--lateral-resolution", "0.05",
                         # "--collision.action", "warn",
                         # "--step-length", str(TS)
                         ])
            # steal focus for keyboard input after sumo-gui has loaded
            # self.master.focus_force() # not working on all platforms
            # make sure ego vehicle is loaded
            traci.simulationStep()
            speed = traci.vehicle.getSpeed(self.egoID)
            angle = traci.vehicle.getAngle(self.egoID)
            traci.vehicle.setSpeedMode(self.egoID, 0)
            steerAngle = 0
            x, y = traci.vehicle.getPosition(self.egoID)
            traci.gui.trackVehicle(traci.gui.DEFAULT_VIEW, self.egoID)
            while traci.simulation.getMinExpectedNumber() > 0:
                try:
                    if eventQueue.qsize():
                        button = eventQueue.get(0)
                        if button == "next_car":
                            p = self.egoID.split('.')
                            self.egoID = p[0] + '.' + str(int(p[1])+1)
                            speed = traci.vehicle.getSpeed(self.egoID)
                            angle = traci.vehicle.getAngle(self.egoID)
                            traci.vehicle.setSpeedMode(self.egoID, 0)
                            steerAngle = 0
                            x, y = traci.vehicle.getPosition(self.egoID)
                            traci.gui.trackVehicle(traci.gui.DEFAULT_VIEW, self.egoID)
                            continue
                        elif button == "prev_car":
                            p = self.egoID.split('.')
                            self.egoID = p[0] + '.' + str(int(p[1])-1)
                            speed = traci.vehicle.getSpeed(self.egoID)
                            angle = traci.vehicle.getAngle(self.egoID)
                            traci.vehicle.setSpeedMode(self.egoID, 0)
                            steerAngle = 0
                            x, y = traci.vehicle.getPosition(self.egoID)
                            traci.gui.trackVehicle(traci.gui.DEFAULT_VIEW, self.egoID)
                            continue
                        else:
                            self.type = button
                            if self.type == "fast":
                                speed += 2
                            elif self.type == "slow":
                                speed -= 2
                            elif self.type == "up":
                                angle -= 5
                            elif self.type == "down":
                                angle += 5
                            else:
                                print("WHAT ARE YOU DOING?!?!")
                except Queue.Empty:
                    pass
                # angle += steerAngle
                angle = angle % 360
                rad = angle / 180 * math.pi + 0.5 * math.pi
                x2 = x - math.cos(rad) * TS * speed
                y2 = y + math.sin(rad) * TS * speed
                traci.vehicle.setSpeed(self.egoID, speed)
                traci.vehicle.moveToXY(self.egoID, "1", 0, x2, y2, angle, keepRoute=2)
                x3, y3 = traci.vehicle.getPosition(self.egoID)
                x, y = x2, y2
                traci.simulationStep()
                if VERBOSE:
                    print("old=%.2f,%.2f new=%.2f,%.2f found=%.2f,%.2f speed=%.2f steer=%.2f angle=%s rad/pi=%.2f cos=%.2f sin=%.2f" % (
                        x, y, x2, y2, x3, y3, speed, steerAngle, angle, rad / math.pi,
                        math.cos(rad), math.sin(rad)))
            traci.close()
        except traci.FatalTraCIError:
            pass
        self.running = False


def human_main(sumocfg="/home/karenleung/Documents/simulator/lanechange/sumo-files/lanechange_fancy.sumo.cfg", egoID="exit.0"):
    root = Tk()
    root.geometry('180x100+0+0')
    frame = Frame(root)
    Button(frame, text="Click here.\nControl with arrow keys").grid(row=0)
    root.bind('<Left>', leftKey)
    root.bind('<Right>', rightKey)
    root.bind('<Up>', upKey)
    root.bind('<Down>', downKey)
    root.bind('n', nextCar)
    root.bind('p', prevCar)
    frame.pack()

    client = AdoVehicleControl(root, sumocfg, egoID)
    root.mainloop()

def main(reference, gain, num_trials):
    filename = "/home/karen/projects/trams/explanatory_factors/sumo-files/lanechange_" + str(int(reference)) + ".sumo.cfg"
    gui = False
    data_dict = dict()

    avac = AdoVehicleAutoControl(gui, filename, reference, gain)
    for trial in range(num_trials):
        print("===== RUNNING TRIAL ", trial, "=====")
        updateXML(trial, reference)
    #     scenario = generateDataFilesControl(trial, record)
        avac.run()
        data_dict[trial] = pd.DataFrame(avac.history)

    data_df = pd.concat(data_dict)

    print("Data collection completed")
    data_df.to_pickle( os.path.abspath(os.path.join( os.pardir, "data/pickles/lanechange_control_" + str(reference) + "_" + str(gain) + ".pkl")))

reference = float(sys.argv[1])
num_trials = int(sys.argv[2])
if len(sys.argv) >= 4:
    gain = float(sys.argv[3])
    main(reference, gain, num_trials)
else:
    print("==================== gain = 0.1 ====================")
    main(reference, 0.1, num_trials)
    print("==================== gain = 0.5 ====================")
    main(reference, 0.5, num_trials)
    print("==================== gain = 1.0 ====================")
    main(reference, 1.0, num_trials)