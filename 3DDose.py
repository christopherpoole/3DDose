import numpy


class DoseFile(object):
    def __init__(self, file_name):
        if file_name[-3:] == 'npz':
            self._load_npz(file_name)
        elif file_name[-6:] == '3ddose':
            self._load_3ddose(file_name)
    
    def _load_npz(self, file_name):
        data = numpy.load(file_name)
        self.dose = data['dose']
        self.uncertainty = data['uncertainty']
        self.positions = [data['x_positions'], data['y_positions'], data['z_positions']]
        self.spacing = [numpy.diff(p) for p in self.positions]
        self.resolution = [s[0] for s in self.spacing if s.all()] 
        
        self.shape = self.dose.shape
        self.size = self.dose.size
    
    def _load_3ddose(self, file_name):
        data = file(file_name).read().split('\n')
        x, y, z = map(int, data[0].split())
        self.shape = (z, x, y)
        self.size = numpy.multiply.reduce(self.shape)
        
        self.positions = [numpy.fromstring(data[i], sep=' ') for i in range(1, 4)]
        self.spacing = [numpy.diff(p) for p in self.positions]
        self.resolution = [s[0] for s in self.spacing if s.all()]      
        assert len(self.resolution) == 3, "Non-linear resolution in either x, y or z."

        self.dose = numpy.fromstring(data[4], sep=' ')
        self.dose = self.dose.reshape((self.shape))
        assert self.dose.size == self.size, "Dose array size does not match that specified."

        self.uncertainty = numpy.fromstring(data[5], sep=' ')
        self.uncertainty = self.uncertainty.reshape((self.shape))
        assert self.uncertainty.size == self.size, "Uncertainty array size does not match that specified."

    def dump(self, file_name):
        numpy.savez(file_name, dose=self.dose, uncertainty=self.uncertainty,
            x_positions=self.positions[0], y_positions=self.positions[1],
            z_positions=self.positions[2])

    def max(self):
        return self.dose.max()
        
    def min(self):
        return self.dose.min()

    @property
    def x_extent(self):
        return self.positions[0][0], self.positions[0][-1]

    @property
    def y_extent(self):
        return self.positions[1][0], self.positions[1][-1]
    
    @property
    def z_extent(self):
        return self.positions[2][0], self.positions[2][-1]
        
        
if __name__ == "__main__":
    import pylab
    import sys   
 
    data = DoseFile(sys.argv[1])
    
    pylab.matshow(data.dose[5,:,:], cmap=pylab.cm.gray, extent=data.x_extent + data.y_extent)
    pylab.contour(data.dose[5,:,:], origin='upper', extent=data.x_extent + data.y_extent)
    pylab.show()
    
