from __future__ import print_function
import glob  # Import glob to easily loop over files
from Bio import SeqIO

import os

# import Utils
amino_acid_list = {"A": [89.1, 47, 41, 1.41, 0.72, 0.82, 2.4, 9.9],
                   "B": [132.118, -41, -28, 0.76, 0.48, 1.34, 2.1, 8.7],
                   "C": [121.16, 52, 49, 0.66, 2.4, 0.54, 1.9, 10.07],
                   "D": [133.1, -18, -55, 0.99, 0.39, 1.24, 2, 9.9],
                   "E": [147.13, 8, -13, 1.59, 0.52, 1.01, 2.1, 9.5],
                   "F": [165.19, 92, 100, 1.16, 1.33, 0.59, 2.2, 9.3],
                   "G": [75.07, 0, 0, 0.43, 0.58, 1.77, 2.4, 9.8],
                   "H": [155.16, -42, 8, 1.05, 0.8, 0.81, 1.8, 9.3],
                   "I": [131.18, 100, 99, 1.09, 1.67, 0.47, 2.3, 9.8],
                   "K": [146.188, -37, -23, 1.23, 0.69, 1.07, 2.2, 9.1],
                   "L": [131.18, 100, 97, 1.34, 1.22, 0.57, 2.3, 9.7],
                   "M": [149.21, 74, 74, 1.3, 1.14, 0.52, 2.1, 9.3],
                   "N": [132.118, -41, -28, 0.76, 0.48, 1.34, 2.1, 8.7],
                   "P": [115.13, -46, -46, 0.34, 0.31, 1.32, 2, 9.6],
                   "Q": [146.15, -18, -10, 1.27, 0.98, 0.84, 2.2, 9.1],
                   "R": [174.2, -26, -14, 1.21, 0.84, 0.9, 1.8, 9],
                   "S": [105.09, -7, -5, 0.57, 0.96, 1.22, 2.2, 9.2],
                   "T": [119.2, 13, 13, 0.76, 1.17, 0.96, 2.1, 9.1],
                   "V": [117.15, 79, 76, 0.9, 1.87, 0.41, 2.3, 9.7],
                   "W": [204.25, 84, 97, 1.02, 1.35, 0.65, 2.5, 9.4],
                   "Y": [181.19, 49, 63, 0.74, 1.45, 0.76, 2.2, 9.2],
                   "Z": [146.15, -18, -10, 1.27, 0.98, 0.84, 2.2, 9.1]}

import pickle


def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def ReadData(file_dir):
    protein_seq_dict = dict()
    protein_seq_name_dict = dict()
    i = 0
    for file_loc in glob.glob('{}/PF*.txt'.format(file_dir)):
        protein_name = os.path.splitext(os.path.basename(file_loc))[0].replace("_seed", "")
        fasta_sequences = SeqIO.parse(open(file_loc), 'fasta')
        protein_seq_dict[protein_name] = []
        protein_seq_name_dict[protein_name]=[]
        for fasta in fasta_sequences:
            name, sequence = fasta.id, str(fasta.seq)
            if len(sequence)>100:
                protein_seq_name_dict[protein_name].append(name)
                embedding = []
                for amino_acid in [v for v in list(sequence) if v in amino_acid_list.keys()]:
                    embedding.append(amino_acid_list[amino_acid][0:4])
                    if len(embedding) == 100:
                        break

                protein_seq_dict[protein_name].append(embedding)
        i += 1
        # if i > 1000:
        #     break


    return protein_seq_dict,protein_seq_name_dict

protein_seq_dict,protein_seq_name_dict = ReadData('target_sequences')
save_obj(protein_seq_dict, 'protein_seq_dict_60')
save_obj(protein_seq_name_dict, 'protein_seq_dict_name_60')

# protein_seq_dict = load_obj('protein_seq_dict_name')
# protein_seq_name_dict = sorted(protein_seq_name_dict.__getitem__())
for proteins in protein_seq_name_dict:
    print(proteins+","+str(len(protein_seq_name_dict[proteins])))


# np.savez('protein_seq_dict', protein_seq_dict=protein_seq_dict)
# data = np.load('protein_seq_dict.npy')

# data = np.load('protein_seq_dict.npz')
# with np.load('protein_seq_dict.npy') as data:
# print(data['PF000001'])
# print(len(protein_seq_dict))
