import yaml

class Config:
    def __init__(self, file_path):
        self.file_path = file_path
        self.name = None
        self.input_dir = None
        self.matrix_file = None
        self.detectors_file = None
        self.config_template = None
        self.htcondor_config = None
        self.measurements = []
        self.calibration = {}
        self.labels = {}
        self.colours = {}
        self.plotting= {}

    def load(self):
        with open(self.file_path, 'r') as file:
            data = yaml.safe_load(file)
            self.name = data.get('name', None)
            self.input_dir = data.get('input_dir', None)
            self.matrix_file = data.get('matrix_file', None)
            self.detectors_file = data.get('detectors_file', None)
            self.config_template = data.get('config_template', None)
            self.htcondor_config = data.get('htcondor_config', None)
            self.measurements = data.get('measurements', [])
            self.calibration = data.get('calibration_energy_keV', {})
            self.labels = data.get('labels', {})
            self.colours = data.get('colours', {})
            self.plotting = data.get('plotting', {})

    def __str__(self):
        return f"Config:\n  Name: {self.name}\n  Input Dir: {self.input_dir}\n  Matrix File: {self.matrix_file}\n  Detectors File: {self.detectors_file}\n  Config Template: {self.config_template}\n  HTCondor Config: {self.htcondor_config}\n  Measurements: {self.measurements}"
