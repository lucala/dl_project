import numpy as np
import torchfile
import os
import glob


# Set the folder
folder = 'data_vqa_feat'

# Process image list first
for filename in glob.glob(os.path.join(folder, '*_imglist.dat')):
    
    # Load .dat
    data = torchfile.load(filename)

    # Extract the name
    names = []
    for el in data:
        names.append(el.decode('utf-8'))
    
    # Save to txt
    np.savetxt(filename + '.txt', names, fmt="%s")

# Process features
for filename in glob.glob(os.path.join(folder, '*_feat.dat')):

    # Load .dat
    X = torchfile.load(filename)

    # Write to npz
    np.save(filename + '.npy', X)
