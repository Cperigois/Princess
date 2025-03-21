from Princess.gwtools.Detector import Detector
from Princess.gwtools.Network import Network
import json
import importlib.resources


def initialization():
    # Import parameter file
    with importlib.resources.open_text("Princess.Run", "Params.json") as f:
        params = json.load(f)

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