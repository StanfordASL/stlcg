import json
import numpy as np
import os
from collections import OrderedDict, defaultdict

def generateEntriesString(keywords, header=''):
    string = '<' + header 
    for i in range(len(keywords)):
        string +='%s="%s" '
    string +='/>\n'
    return string


def generateLCVehicleParams(car_id):
    # lcStrategic The eagerness for performing strategic lane changing. Higher values result in earlier lane-changing. default: 1.0, range [0-inf[    LC2013, SL2015
    # lcCooperative   The willingness for performing cooperative lane changing. Lower values result in reduced cooperation. default: 1.0, range [0-1] LC2013, SL2015
    # lcSpeedGain The eagerness for performing lane changing to gain speed. Higher values result in more lane-changing. default: 1.0, range [0-inf[   LC2013, SL2015
    # lcKeepRight The eagerness for following the obligation to keep right. Higher values result in earlier lane-changing. default: 1.0, range [0-inf[    LC2013, SL2015
    # lcLookaheadLeft Factor for configuring the strategic lookahead distance when a change to the left is necessary (relative to right lookahead). default: 2.0, range ]0-inf[   LC2013, SL2015
    # lcSpeedGainRight    Factor for configuring the treshold asymmetry when changing to the left or to the right for speed gain. By default the decision for changing to the right takes more deliberation. Symmetry is achieved when set to 1.0. default: 0.1, range ]0-inf[    LC2013, SL2015
    # lcSublane   The eagerness for using the configured lateral alignment within the lane. Higher values result in increased willingness to sacrifice speed for alignment. default: 1.0, range [0-inf]   SL2015
    # lcPushy Willingness to encroach laterally on other drivers. default: 0, range 0 to 1    SL2015
    # lcPushyGap  Minimum lateral gap when encroaching laterally on other drives (alternative way to define lcPushy). default: minGapLat, range 0 to minGapLat    SL2015
    # lcAssertive Willingness to accept lower front and rear gaps on the target lane. The required gap is divided by this value. default: 1, range: positive reals    LC2013,SL2015
    # lcImpatience    dynamic factor for modifying lcAssertive and lcPushy. default: 0 (no effect) range -1 to 1. Impatience acts as a multiplier. At -1 the multiplier is 0.5 and at 1 the multiplier is 1.5.    SL2015
    # lcTimeToImpatience  Time to reach maximum impatience (of 1). Impatience grows whenever a lane-change manoeuvre is blocked.. default: infinity (disables impatience growth)  SL2015
    # lcAccelLat  maximum lateral acceleration per second. default: 1.0   SL2015
    # lcTurnAlignmentDistance Distance to an upcoming turn on the vehicles route, below which the alignment should be dynamically adapted to match the turn direction. default: 0.0 (i.e., disabled)  SL2015
    # lcMaxSpeedLatStanding   Upper bound on lateral speed when standing. default: maxSpeedLat (i.e., disabled)   SL2015
    # lcMaxSpeedLatFactor Upper bound on lateral speed while moving computed as lcMaxSpeedLatStanding + lcMaxSpeedLatFactor * getSpeed(). default: 1.0    SL2015

    # lcCooperative: lower values cause less cooperation during lane changing
    # lcSpeedGain: higher values cause more overtaking for speed
    # lcKeepRight: lower values cause less driving on the right
    # lcPushy: setting this to values between 0 and 1 causes aggressive lateral encroachment (only when using the Sublane Model)
    # lcAssertive: setting this values above 1 cause acceptance of smaller longitudinal gaps in proportion to driver impatience(only when using the Sublane Model)
    params = {}
    params["accel"] = 4
    params["decel"] = 6
    params["id"] = car_id
    params["length"] = "4.8"
    params["maxSpeed"] = str(round(np.random.randint(8,15) + np.random.randn()/2, 3))
    params["sigma"] = str(round(np.random.rand(), 3))
    params["lcStrategic"] = str(round(np.random.rand()*10, 3))   
    params["lcCoorperative"] = str(round(np.random.rand()*5, 3))
    params["lcSpeedGain"] = str(round(np.random.rand(), 3)) 
    params["lcKeepRight"] = str(round(np.random.rand(), 3)*0)
    params["lcPushy"] = str(round(np.random.rand(), 3))
    params["lcSpeedGainRight"] = str(1.0)
    # params["lcAccelLat"] = str(1.5)
    # params["actionStepLength"] = str(0.5)
    params["lcSublane"] = str(round(np.random.rand(), 3)) 
    params["lcAssertive"] = str(round(np.random.rand(), 3))
    params["speedFactor"] = "normc(1,0.1,0.2,2)"
    params["tau"] = 0.05
    return params 



def generateXMLstring(scenario, routes):
    xml_string = "<routes>\n" 
    flow_keywords = ["id",
                     "color",
                     "begin",
                     "end",
                     "period",
                     "type",
                     "route",
                     "departSpeed"]
    vehicle_keywords = ["depart",
                        "id",
                        "color",
                        "route",
                        "type",
                        "departLane",
                        "departPos",
                        "departSpeed",
                        "arrivalLane"]

    route_keywords = ["id", "edges"]
    # generate vType
    for i in range(scenario["n_cars"]):
        veh_keywords = scenario["cars"][i].keys()
        xml_string += '\t' + generateEntriesString(veh_keywords, header='vType ') % tuple(np.concatenate([[key, scenario["cars"][i][key]] for key in veh_keywords], 0))
    for i in range(routes["n_routes"]):
        route_keywords = routes["routes"][i].keys()
        xml_string += '\t' + generateEntriesString(route_keywords, 'route ') % tuple(np.concatenate([[key, routes['routes'][i][key]] for key in route_keywords]))
    for i in range(scenario["n_cars"]):
        xml_string += '\t' + generateEntriesString(flow_keywords, 'flow ') % tuple(np.concatenate([[key, scenario[key][i]] for key in flow_keywords], 0))
    xml_string += "</routes>\n"	
    return xml_string


def laneswapScenario(cars=[generateLCVehicleParams("Car1"), generateLCVehicleParams("Car2")], p=1):
    scenario = {}
    scenario["n_cars"] = len(cars)
    scenario["cars"] = cars 
    scenario["color"] = ["1,0,0", "0,1,0", "0,0,1"]
    scenario["type"] = ["Car1", "Car2", "Car0"]
    scenario["scenario"] = "laneswap"
    scenario["id"] = ["enter", "exit", "traffic"]
    scenario["route"] = ["entering", "exiting", "staying"]
    scenario["end"] = ["40"] * 3
    scenario["begin"] = ["0", "0", "3"]
    scenario["departSpeed"] = ["random"] * 3
    scenario["period"] = ["4", "4", "5"]
    scenario["color"] = ["1,0,0", "0,1,0", "0,0,1"]
    scenario["type"] = ["Car1", "Car2", "Car0"]


    # scenario["init_crossovertime"] = round(np.random.rand()*1.9 + 0.1, 3)
    # scenario["stagger"] = np.random.randint(0,2)  # if 0, car 1 starts behind. if 1, car 2 starts behind
    # min_max_speed = min(*[float(m["maxSpeed"]) for m in cars])
    # dx = 20
    # if scenario["stagger"]:  # if 1
    #     # car2 starts behind
    #     v2 = min_max_speed-p
    #     dv = np.random.rand()*2 + 1
    #     v1 = v2 - dv
    #     departSpeed = (str(round(v1, 3)), str(round(v2, 3)))

    #     x1 = dv * scenario["init_crossovertime"]
    #     departPos = (str(round(x1+dx, 3)), str(dx))

    # else:
    #     v1 = min_max_speed-p
    #     dv = np.random.rand()*2 + 1
    #     v2 = v1 - dv
    #     departSpeed = (str(round(v1, 3)), str(round(v2, 3)))

    #     x2 = dv * scenario["init_crossovertime"]
    #     departPos = (str(dx), str(round(x2+dx, 3)))

    # scenario["departPos"] = departPos
    # scenario["departSpeed"] = departSpeed
    return scenario


def generateDataFiles(trial, record=True):
    par_dir = os.path.abspath(os.path.join(os.pardir))
    # car 1 is the car in the left lane
    routes = {}
    routes["n_routes"] = 3
    routes["routes"] = [{"id": "exiting", "edges": "0 1 4"}, {"id": "entering", "edges": "3 1 2"},{"id": "staying", "edges": "0 1 2"}]
    car0 = {"id": "Car0", "type": "passenger", "length": "5", "accel": "4", "decel": "6", "sigma": "1.0" }
    cars=[generateLCVehicleParams("Car1"), generateLCVehicleParams("Car2"), car0]
    outfile_rou = par_dir + "/sumo-files/lanechange.rou.xml"
    outfile_scenario = par_dir + "/data/scenario/scenario_" + str(trial) +""

    scenario = laneswapScenario(cars)
    xml_string = generateXMLstring(scenario, routes)
    # writing out the rou file
    with open(outfile_rou, "w") as f:
        f.write(xml_string)
    print("Writing XML file at " + outfile_rou)    

    # saving the scenario dictionary as a json
    with open(outfile_scenario, 'w') as f:
        f.write(json.dumps(scenario, indent=2)) # use `json.loads` to do the reverse
    print("writing scenario data at " + outfile_scenario)

    # generate config file
    config_dict = configuration_dict(trial, record)
    generateConfigFile(config_dict)

       
    return scenario     


def generateDataFilesControl(trial, record=True):
    par_dir = os.path.abspath(os.path.join(os.pardir))
    # car 1 is the car in the left lane
    routes = {}
    routes["n_routes"] = 2
    routes["routes"] = [{"id": "exiting", "edges": "0 1 4"}, {"id": "entering", "edges": "3 1 4"},{"id": "staying", "edges": "0 1 2"}]
    # routes["routes"] = [{"id": "exiting", "edges": "0 1 4"}, {"id": "entering", "edges": "3 1 2"}]
    car0 = {"id": "Car0", "type": "passenger", "length": "5", "accel": "4", "decel": "6", "sigma": "1.0" }
    cars=[generateLCVehicleParams("Car1"), generateLCVehicleParams("Car2")]
    outfile_rou = par_dir + "/sumo-files/lanechange.rou.xml"
    outfile_scenario = par_dir + "/data/scenario/scenario_" + str(trial) +""

    scenario = laneswapScenario(cars)
    xml_string = generateXMLstring(scenario, routes)
    # writing out the rou file
    with open(outfile_rou, "w") as f:
        f.write(xml_string)
    print("Writing XML file at " + outfile_rou)    

    # saving the scenario dictionary as a json
    with open(outfile_scenario, 'w') as f:
        f.write(json.dumps(scenario, indent=2)) # use `json.loads` to do the reverse
    print("writing scenario data at " + outfile_scenario)

    # generate config file
    config_dict = configuration_dict(trial, record)
    generateConfigFile(config_dict)

       
    return scenario     





def generateConfigFile(config_dict):
    par_dir = os.path.abspath(os.path.join(os.pardir))
    config_string = '<configuration>\n'
    for (k,v) in config_dict.items():
        config_string += '\t<' + k + '>\n'
        for (kk,vv) in v.items():
            config_string += '\t\t<' + kk + ' value="' + vv + '"/>\n'
        config_string += '\t</' + k + '>\n'
    config_string += '</configuration>'


    outfile = par_dir + "/sumo-files/lanechange.sumo.cfg"

    with open(outfile, "w") as f:
        f.write(config_string)
    print("Writing config file at " + outfile)  



def configuration_dict(trial, record=True):
    par_dir = os.path.abspath(os.path.join(os.pardir))    
    d = defaultdict(lambda: defaultdict(str))

    tag = '_' + str(trial)
    d['input']["net-file"] = "lanechange.net.xml"
    d['input']["route-files"] = "lanechange.rou.xml"
    d['input']["gui-settings-file"] = "lanechange.settings.xml"
  

    d["time"]["step-length"] = "0.1" 
    d["time"]["begin"] = "0"
    d["time"]["end"] = "70"
    d["time"]["default.action-step-length"] = "0.3"

    d["processing"]["lateral-resolution"] = "0.05" 
    d["processing"]["collision.mingap-factor"] = "1.0" 
    d["processing"]["collision.action"] = "none"

    if record:
        d["output"]["vehroute-output.sorted"] ="true" 
        d["output"]["vehroute-output.write-unfinished"] ="true" 
        d["output"]["lanechange-output.started"] ="true"
        d["output"]["lanechange-output.ended"] ="true" 
        d["output"]["full-output"] = par_dir + "/data/fulloutputs/full-output" + tag + ".xml" 
        d["output"]["fcd-output"] = par_dir + "/data/fcdoutputs/fcd-output" + tag + ".xml" 
        d["output"]["lanechange-output"] = par_dir + "/data/lanechangeoutputs/lanechange-output" + tag + ".xml" 

        d["report"]["duration-log.statistics"] = "true" 
        d["report"]["message-log"] = par_dir + "/data/messagelogs/message-log" + tag + ".txt"
        d["report"]["error-log"] = par_dir + "/data/errorlogs/error-log" + tag + ".txt" 

    return d