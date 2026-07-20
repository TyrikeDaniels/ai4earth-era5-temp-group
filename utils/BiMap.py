class BiMap:
    def __init__(self):
        self.key_to_value = {}
        self.value_to_key = {}

    def add(self, key, value):
        # Add a key-value pair, ensuring the mapping remains consistent.
        if key in self.key_to_value or value in self.value_to_key:
            raise ValueError("Key or value already exists.")
        self.key_to_value[key] = value
        self.value_to_key[value] = key

    def get_value(self, key):
        # Retrieve value using key.
        return self.key_to_value.get(key)

    def get_key(self, value):
        # Retrieve key using value.
        return self.value_to_key.get(value)

    def remove_by_key(self, key):
        # Remove a mapping by key.
        if key in self.key_to_value:
            value = self.key_to_value.pop(key)
            del self.value_to_key[value]

    def remove_by_value(self, value):
        # Remove a mapping by value.
        if value in self.value_to_key:
            key = self.value_to_key.pop(value)
            del self.key_to_value[key]


if __name__ == '__main__':
    # Example Usage
    bimap = BiMap()
    bimap.add("a", 1)
    bimap.add("b", 2)

    print(bimap.get_value("a"))  # Output: 1
    print(bimap.get_key(2))      # Output: b

    bimap.remove_by_key("a")
    print(bimap.get_value("a"))  # Output: None