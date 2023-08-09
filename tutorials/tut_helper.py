import json
import scanpy as sc 
from signaturescoring.utils.utils import check_signature_genes

def get_sigs_from_DGEX_list(adata, DE_of_celltypes, remove_overlapping=True):
    """
    Helper function to extract signatures for B-cells, monocytes and NK-cells based on list of differentially expressed genes from the original paper[1].
    
    [1] Hao, Yuhan, Stephanie Hao, Erica Andersen-Nissen, William M. Mauck 3rd, Shiwei Zheng, Andrew Butler, Maddie J. Lee, et al. 2021. “Integrated Analysis of Multimodal Single-Cell Data.” Cell 184 (13): 3573–87.e29.
    
    Args:
        adata (int): The preprocessed PBMC dataset.
        DE_of_celltypes (str): List of differentially expressed genes per celltype (column "Cell Type").
        remove_overlapping (bool): Indicates if overlapping signature genes should be removed from the signatures. 

    Returns:
        dict: Dictionary conatining for each cell-type a signature  (gene list).
    """
    
    # Get all subtypes belonging to granularity level 1 cell types
    available_subgroups = adata.obs[['celltype.l1', 'celltype.l2','celltype.l3']].value_counts().reset_index().sort_values(['celltype.l1', 'celltype.l2','celltype.l3'])
    
    available_subgroups = dict(available_subgroups.groupby('celltype.l1')['celltype.l3'].apply(lambda x: list(x.astype(str).unique())))
    
    sc.logging.info(f'Types and their subtypes:\n{json.dumps(available_subgroups, indent=4)}')
    
    # Group DGEX gene list by the celltype 
    grouped_DE_of_celltypes = DE_of_celltypes.groupby(by='Cell Type')
    
    # Union for each cell type the DGEX genes associated with its corresponding subtypes. 
    SG_subtypes = {}
    for key, subtypes in available_subgroups.items():
        sig_genes = set()
        for subtype in subtypes:
            if subtype not in grouped_DE_of_celltypes.groups.keys():
                continue
            group = grouped_DE_of_celltypes.get_group(subtype)
            sig_genes.update(group.sort_values(by='Average Log Fold Change', ascending=False)['Gene'].iloc[0:300])
        SG_subtypes[key] = sig_genes
     
    # Remove genes appearing in more than one signature 
    if remove_overlapping:
        intersec_B_Mono = SG_subtypes['B'].intersection(SG_subtypes['Mono'])
        intersec_B_NK = SG_subtypes['B'].intersection(SG_subtypes['NK'])
        intersec_Mono_NK = SG_subtypes['Mono'].intersection(SG_subtypes['NK'])
        
        overlap_B = intersec_B_Mono.union(intersec_B_NK)
        overlap_Mono = intersec_B_Mono.union(intersec_Mono_NK)
        overlap_NK = intersec_B_NK.union(intersec_Mono_NK)
        
        SG_subtypes['B'] = list(SG_subtypes['B'].difference(overlap_B))
        SG_subtypes['Mono'] = list(SG_subtypes['Mono'].difference(overlap_Mono))
        SG_subtypes['NK'] = list(SG_subtypes['NK'].difference(overlap_NK))
        
    # Check signatures, i.e., remove genes not avaiable in the dataset 
    for key, val in SG_subtypes.items():
        SG_subtypes[key] = list(val)
        SG_subtypes[key]  = check_signature_genes(adata.var_names, val)
        
    return SG_subtypes
        
     
        

    
    
    
    
    