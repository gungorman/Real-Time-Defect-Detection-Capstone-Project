from os import listdir
from os.path import join

### You need to run this file from within the folder that was exported from label-studio ###

old_class_index = '0 ' #Change for the current class index if neccecary but keep the space at the end
new_class_index = 'X ' #Fill in the new class indext and keep the space at the end

for file in listdir('labels'):
    with open(join('labels', file), 'r') as f:
        text = f.read()
        new_text = ''
        lines = text.split('\n')
        for line in lines:
            new_line = line.replace(old_class_index, new_class_index, 1)
            new_text += new_line
            new_text += "\n"
        new_text = new_text[:-1]
    
    with open(join('labels', file), 'w') as f:
        f.write(new_text)

print('done')