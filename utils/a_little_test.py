class Note:
    def __init__(self, length):
        self.length = length


# Create a list of Element objects with Note objects
notes = [
    Note(10),
    Note(20),
    Note(15)
]

# Find the element with the maximum note length
max_notel = (max(notes, key=lambda x: x.length)).length

print(f'The element with the maximum note length has a length of {max_notel}')
