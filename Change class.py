from os import listdir
from os.path import join

### You need to run this file from within the folder that was exported from label-studio ###

# Create a dictionary to map the current class-id to the id wich it should be
class_change_dict = {'0': '0',
                     '1': '5',
                     '2': '6',
                     '3': '7',
                     '4': '8',
                     '5': '9',
                     '6': '10',
                     '7': '1',
                     '8': '2',
                     '9': '3',
                     '10': '4',}

# Change the id for al the labels in all the files
for file in listdir('labels'):
    with open(join('labels', file), 'r') as f:
        text = f.read()
        new_text = ''
        lines = text.split('\n')
        for line in lines:
            old_index = line.split(' ')[0]
            if old_index != '':
                new_index = class_change_dict[old_index]
                new_line = line.replace(old_index, new_index, 1)
                new_text += new_line
            new_text += "\n"
        new_text = new_text[:-1]
    
    with open(join('labels', file), 'w') as f:
        f.write(new_text)

print('done')