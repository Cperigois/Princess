import gwtools.Detector as DET
import gwtools.Network as NET
import json

params = json.load(open('./Run/Params.json', 'r'))

def initialization():
    for det in params['detector_list'].keys():
        detector = DET.Detector(name=params['detector_list'][det]['name'],
                            configuration=params['detector_list'][det]['configuration'],
                            origin=params['detector_list'][det]['origin'],
                            reference=params['detector_list'][det]['reference'],
                            type= params['detector_list'][det]['type'])
        detector.save()
    for net in params['network_list'].keys():
        network = NET.Network(name=params['network_list'][net]['name'],
                          compo=params['network_list'][net]['compo'],
                          pic_file=params['network_list'][net]['pic_file'],
                          efficiency=params['network_list'][net]['efficiency'],
                          SNR_thrs= params['network_list'][net]['SNR_thrs'])
        network.save()