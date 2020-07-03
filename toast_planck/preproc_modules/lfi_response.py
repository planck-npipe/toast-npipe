

class LFI_response:
    """ Class to store response curve in similar format to
    LFI_response data object
    """
    def __init__(self,keys,sky_Vi,sky_Vo,ref_Vi,ref_Vo):
        self.keys = keys             # Dictionary of keys and values
        self.sky_volt_in = sky_Vi
        self.sky_volt_out = sky_Vo
        self.load_volt_in = ref_Vi
        self.load_volt_out = ref_Vo


