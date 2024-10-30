import yaml
from measurement import Measurement, BaselineMeasurement, SpectrumMeasurement

class Config:
    def __init__(self, file_path):
        self.file_path = file_path
        self.name = None
        self.timestamp = None
        self.measurements = {}
        self.plotting = {}

    def load(self):
        with open(self.file_path, 'r') as file:
            data = yaml.safe_load(file)
            self.name = data.get('name', None)
            self.timestamp = data.get('timestamp', None)
            self.group_measurements(data.get('measurements', {}))
            self.plotting = data.get('plotting', {})
        
    def __str__(self):
        return f'Config(name={self.name}, timestamp={self.timestamp}, measurements={self.measurements}, plotting={self.plotting})'

    def group_measurements(self, measurements):
        for name, data in measurements.items():
            if data['type'] == 'baseline':
                measurement = BaselineMeasurement(name, data)
            elif data['type'] == 'fit_spectrum':
                measurement = SpectrumMeasurement(name, data)
            else:
                measurement = Measurement(name, data)
            group = measurement.group
            if group not in self.measurements:
                self.measurements[group] = []
            self.measurements[group].append(measurement)
