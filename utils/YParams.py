from ruamel.yaml import YAML
import logging

class YParams():
  """
  A class to parse and manage configuration parameters from a YAML file.

  Attributes:
      _yaml_filename (str): Path to the YAML configuration file.
      _config_name (str): The name of the configuration block to load from the file.
      params (dict): A dictionary to store configuration parameters.
  """
  def __init__(self, yaml_filename, config_name, print_params=False):
    self._yaml_filename = yaml_filename
    self._config_name = config_name
    self.params = {}

    if print_params:
      print("------------------ Configuration ------------------")

    with open(yaml_filename) as _file:

      for key, val in YAML().load(_file)[config_name].items():
        if print_params: print(key, val)
        if val =='None': val = None
        # Store the parameter in both the dictionary and as an attribute
        self.params[key] = val
        self.__setattr__(key, val)

    if print_params:
      print("---------------------------------------------------")

    # override setattr now so both the dict and the attrs get updated
    self.__setattr__ = self.__custom_setattr__

  def __custom_setattr__(self, key, val):
    """
    Custom method for setting attributes. Updates both the dictionary 
    (`params`) and the object's attributes.
    """
    self.params[key] = val
    super().__setattr__(key, val)

  def __getitem__(self, key):
    return self.params[key]

  def __setitem__(self, key, val):
    self.params[key] = val
    self.__setattr__(key, val)

  def __contains__(self, key):
    return (key in self.params)

  def update_params(self, config):
    for key, val in config.items():
      self.params[key] = val
      self.__setattr__(key, val)

  def log(self):
    """
    Log the configuration parameters using the `logging` module.
    """
    logging.info("------------------ Configuration ------------------")
    logging.info("Configuration file: "+str(self._yaml_filename))
    logging.info("Configuration name: "+str(self._config_name))
    for key, val in self.params.items():
        logging.info(str(key) + ' ' + str(val))
    logging.info("---------------------------------------------------")
