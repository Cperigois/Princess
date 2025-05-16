from Princess.gwtools.Detector import Detector
from Princess.gwtools.Network import Network
import json
import os

from Princess.Run.settings import PARAMS_FILE

# Check PARAMS_FILE value
if not PARAMS_FILE or not os.path.exists(PARAMS_FILE):
    raise FileNotFoundError(f"The file parameter {PARAMS_FILE} is missing,. Execute Run.settings.Make_params_file() first.")

# Charge le fichier de paramètres
with open(PARAMS_FILE, "r") as f:
    params = json.load(f)


def initialization():

    for det in params['detector_list'].keys():
        detector = Detector(name=params['detector_list'][det]['name'],
                            configuration=params['detector_list'][det]['configuration'],
                            origin=params['detector_list'][det]['origin'],
                            reference=params['detector_list'][det]['reference'],
                            type= params['detector_list'][det]['type'])
        detector.save()
    for net in params['network_list'].keys():
        network = Network(name=params['network_list'][net]['name'],
                          compo=params['network_list'][net]['compo'],
                          pic_file=params['network_list'][net]['pic_file'],
                          efficiency=params['network_list'][net]['efficiency'],
                          SNR_thrs= params['network_list'][net]['SNR_thrs'])
        network.save()