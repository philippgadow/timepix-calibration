import yaml

class Config:
    def __init__(self, file_path):
        self.file_path = file_path
        self.name = None
        self.timestamp = None
        self.input_dir = None
        self.matrix_file = None
        self.detectors_file = None
        self.config_template = None
        self.htcondor_config = None
        self.measurements = {}
        self.overwrite_threshold = {}
        self.columns = {}
        self.calibration = {}
        self.labels = {}
        self.colours = {}
        self.plotting= {}
        self.fitting = {}

    def load(self):
        with open(self.file_path, 'r') as file:
            data = yaml.safe_load(file)
            self.name = data.get('name', None)
            self.timestamp = data.get('timestamp', None)
            self.input_dir = data.get('input_dir', None)
            self.matrix_file = data.get('matrix_file', None)
            self.detectors_file = data.get('detectors_file', None)
            self.config_template = data.get('config_template', None)
            self.htcondor_config = data.get('htcondor_config', None)
            self.measurements = data.get('measurements', {})
            self.overwrite_threshold = data.get('overwrite_threshold', {})
            self.columns = data.get('columns', {})
            self.calibration = data.get('calibration_energy_keV', {})
            self.labels = data.get('labels', {})
            self.colours = data.get('colours', {})
            self.plotting = data.get('plotting', {})
            self.fitting = data.get('fitting', {})

    def __str__(self):
        return f'Config(name={self.name}, timestamp={self.timestamp}, input_dir={self.input_dir}, matrix_file={self.matrix_file}, detectors_file={self.detectors_file}, config_template={self.config_template}, htcondor_config={self.htcondor_config}, measurements={self.measurements}, overwrite_threshold={self.overwrite_threshold}, columns={self.columns}, calibration={self.calibration}, labels={self.labels}, colours={self.colours}, plotting={self.plotting}, fitting={self.fitting})'
