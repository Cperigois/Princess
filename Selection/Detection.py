
#Here calculate the optimal SNR for each sources of the catalogue, as well as the probability of detection.
def SNR_opt(self):
    """Calculate the SNR.
    Parameters
    ----------
    """
    fileexist = os.path.exists(self.cat_name + '.dat')
    if fileexist == False:
        self.makeCat()