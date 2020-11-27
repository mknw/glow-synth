#!/var/scratch/mao540/miniconda3/envs/maip-venv/bin/python
import yaml

class ConfWrap(object):
    def __init__(self, d=None, fn=None, create=True):
        if d is None and fn is None:
            d = {}
        elif fn:
            stream = open(fn, 'r')
            d = yaml.safe_load(stream)
            stream.close()
        assert isinstance(d, dict), 'only yaml dictionaries supported!\
                                        (i.e. without hyphen)'
        
        supr = super(ConfWrap, self)
        supr.__setattr__('_data', d)
        supr.__setattr__('__create', create)
        if fn:
            self.display()

    def __getattr__(self, name):
        try:
            value = self._data[name]
        except ValueError:
            if not super(ConfWrap, self).__getattribute__('__create'):
                raise #?
            value = {}
            self._data[name] = value
        except KeyError:
            raise KeyError(f'value {name} not found. Try instead: {list(self.keys())}')

        if hasattr(value, 'items'):
            create = super(ConfWrap, self).__getattribute__('__create')
            return ConfWrap(value, create=create)
        return value

    def __setattr__(self, name, value):
        self._data[name] = value

    def __getitem__(self, key):
        try:
            value = self._data[key]
        except KeyError:
            if not super(ConfWrap, self).__getattribute__('__create'):
                raise
            value = {} ##
            self._data[key] = value

        if hasattr(value, 'items'):
            create = super(ConfWrap, self).__getattribute__('__create')
            return ConfWrap(value, create=create)

    def __setitem__(self, key, value):
        self._data[key] = value

    # def __delitem__(self, key):
    # 	del self._data[key]

    def __delattr__(self, key):
        del self._data[key]
    # 	# print(str(self._data[key])+ 'is still here')
    
    def __iadd__(self, other):
        if self._data:
            raise TypeError("A Nested dict will only be replaced if it's empty")
        else:
            return other

    def __repr__(self):
        return str(super(ConfWrap, self).__getattribute__('_data'))

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()
    
    def display(self):
        print("-- Config wrapper --")
        for k, v in self._data.items():
            if isinstance(v, dict):
                print(f'{k}:')
                for l, w in v.items():
                    print(f'\t{l}: {w}')
            else:
                print(f'{k}: {v}')
    

if __name__ == "__main__":
    # C_ = ConfWrap(d={'one': 1, 'two': {'two_one': (2, 1), 'two_two': (2,2)}})
    C = ConfWrap(fn='config.yml')



