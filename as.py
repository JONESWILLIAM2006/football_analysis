class MyClass:
    def __init__(self):
        self.public_attribute = "I am public"
        self._protected_attribute = "I am protected"
        self.__private_attribute = "I am private"

    def public_method(self):
        print("This is a public method.")

    def _protected_method(self):
        print("This is a protected method.")

    def __private_method(self):
        print("This is a private method.")

# Accessing members
obj = MyClass()
print(obj.public_attribute)  # Accessible
obj.public_method()          # Accessible

print(obj._protected_attribute) # Accessible, but convention suggests not to directly access
obj._protected_method()         # Accessible, but convention suggests not to directly access

# Attempting to access private attribute directly will raise an AttributeError
# print(obj.__private_attribute)

# Accessing private attribute through name mangling (discouraged)
print(obj._MyClass__private_attribute)