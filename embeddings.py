import pandas as pd
import numpy as np
from tqdm import tqdm
from ndfrt_analysis import read_embedding_matrix # https://github.com/clinicalml/embeddings

tqdm.pandas()
all_adms_df = pd.DataFrame([])

### Extract Embeddings
def separate_cuis(cui_to_idx):
    cui_diags = {}
    cui_procs = {}
    cui_drugs = {}
    
    for cui, idx in cui_to_idx.items():
        prefix, code = cui.split('_')
        if prefix == 'IDX':
            code = code.replace('.', '')
            cui_diags[code] = idx
        elif prefix == 'C':
            cui_procs[code] = idx
        elif prefix == 'N':
            cui_drugs[code] = idx
    
    return cui_diags, cui_procs, cui_drugs

def compare_codes(subset, codes):
    intersect = subset & codes
    diff = subset - codes
    print(f'Number of codes in subset: {len(subset)}')
    print(f'Present: {len(intersect)}')
    print(f'Not present: {len(diff)}')
    print()
    
    return list(intersect), list(diff)

embeds_file = 'claims_codes_hs_300.txt'
embedding_matrix, idx_to_cui, cui_to_idx = read_embedding_matrix(embeds_file)

cui_diags, cui_procs, cui_drugs = separate_cuis(cui_to_idx)
cui_diags_codes = set(cui_diags.keys())
cui_procs_codes = set(cui_procs.keys())
cui_drugs_codes = set(cui_drugs.keys())

diags_present, diags_not_present = compare_codes(diags_codes, cui_diags_codes)
procs_present, procs_not_present = compare_codes(procs_codes, cui_diags_codes | cui_procs_codes)
drugs_present, drugs_not_present = compare_codes(drugs_codes, cui_drugs_codes)

diags_present_idx = [cui_diags[i] for i in diags_present]
procs_present_idx = [cui_procs[i] if i in cui_procs else cui_diags[i] for i in procs_present]
drugs_present_idx = [cui_drugs[i] for i in drugs_present]

diags_embeddings = pd.DataFrame(embedding_matrix[diags_present_idx], index=diags_present)
procs_embeddings = pd.DataFrame(embedding_matrix[procs_present_idx], index=procs_present)
drugs_embeddings = pd.DataFrame(embedding_matrix[drugs_present_idx], index=[int(x) for x in drugs_present])


### Preprocess of Embeddings for Autoencoder
dxs = []
procs = []

def diagnoses_to_idx(row):
    indices = [diags_embeddings[diags_embeddings.code == str(row[dx])].index for dx in dxs]
    return pd.Series([idx[0] if len(idx) == 1 else 0 for idx in indices])

def diagnoses_to_mask(row):
    indices = [diags_embeddings[diags_embeddings.code == str(row[dx])].index for dx in dxs]
    return pd.Series([int(len(idx) == 1) for idx in indices])

def procedures_to_idx(row):
    indices = [procs_embeddings[procs_embeddings.code == str(row[proc])].index for proc in procs]
    return pd.Series([idx[0] if len(idx) == 1 else 0 for idx in indices])

def procedures_to_mask(row):
    indices = [procs_embeddings[procs_embeddings.code == str(row[proc])].index for proc in procs]
    return pd.Series([int(len(idx) == 1) for idx in indices])

def drugs_to_idx(row):
    embeds = drugs_embeddings[drugs_embeddings['code'] == row['NDC']]
    return embeds.index[0] if len(embeds == 1) else -1

# Diagnoses
diags_df = pd.DataFrame([])

diags_idx_df = pd.DataFrame([])
diags_idx_df[dxs] = diags_df.progress_apply(diagnoses_to_idx, axis=1)

diags_mask_df = pd.DataFrame([])
diags_mask_df[dxs] = diags_df.progress_apply(diagnoses_to_mask, axis=1)

# Procedures
procs_df = pd.DataFrame([])

procs_idx_df = pd.DataFrame([])
procs_idx_df[procs] = procs_df.progress_apply(procedures_to_idx, axis=1)

procs_mask_df = pd.DataFrame([])
procs_mask_df[procs] = procs_df.progress_apply(procedures_to_mask, axis=1)

# Drugs
drugs_idx_df = pd.DataFrame([])
drugs_idx_df['NDC'] = drugs_df.progress_apply(drugs_to_idx, axis=1)


### Average Embeddings
def avg_diag_embeddings(row):
    df = diags_embeddings[diags_embeddings.code.isin([str(row[dx]) for dx in dxs])].copy()
    df.set_index('code', inplace=True)
    
    if df.empty:
        return pd.Series([0] * len(df.columns), index=[str(i) for i in range(len(df.columns))])
    else:
        return pd.Series(df.mean(axis=0), index=[str(i) for i in range(len(df.columns))])
    
def avg_proc_embeddings(row):
    df = procs_embeddings[procs_embeddings.code.isin([str(row[proc]) for proc in procs])].copy()
    df.set_index('code', inplace=True)
    
    if df.empty:
        return pd.Series([0] * len(df.columns), index=[str(i) for i in range(len(df.columns))])
    else:
        return pd.Series(df.mean(axis=0), index=[str(i) for i in range(len(df.columns))])

diags_df = all_adms_df.copy()
diags_average_embeddings = all_adms_df.copy()
diags_average_embeddings[[str(i) for i in range(300)]] = diags_df.progress_apply(avg_diag_embeddings, axis=1)
diags_average_embeddings.to_csv('all_diagnoses_average_embeddings.csv', index=False)

procs_df = all_adms_df.copy()
procs_average_embeddings = all_adms_df.copy()
procs_average_embeddings[[str(i) for i in range(300)]] = procs_df.progress_apply(avg_proc_embeddings, axis=1)
procs_average_embeddings.to_csv('all_procedures_average_embeddings.csv', index=False)