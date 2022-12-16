import networkx as nx
import mygene
import os

import pandas as pd


def get_ppi_network(ppi_path):
    nx_ppi_network = nx.read_edgelist(ppi_path, delimiter=',')
    gene_ids = set()
    for i in nx_ppi_network.nodes:
        gene_ids.add(i)
    return nx_ppi_network, gene_ids


def get_id2names_from_id_list(gene_id_list):
    mg = mygene.MyGeneInfo()
    genelist = mg.querymany(gene_id_list, scopes='entrezgene', fields='symbol', species='human')
    # remove the genes that have no entrezID
    genelist = [i for i in genelist if 'symbol' in i.keys()]

    id2name = {gene['query']: gene['symbol'] for gene in genelist}
    name2id = {gene['symbol']: gene['query'] for gene in genelist}
    return id2name, name2id


def get_id2names_from_name_list(gene_name_list):
    mg = mygene.MyGeneInfo()
    genelist = mg.querymany(gene_name_list, scopes='symbol', fields='entrezgene', species='human')
    # remove the genes that have no entrezID
    genelist = [i for i in genelist if 'entrezgene' in i.keys()]

    id2name = {gene['query']: gene['entrezgene'] for gene in genelist}
    name2id = {gene['entrezgene']: gene['query'] for gene in genelist}
    return id2name, name2id


def get_candidate_gene(name_path, id_path):
    with open(name_path, 'r') as f:
        names = [gene.strip() for gene in f.readlines()]
    positive_gene_name_set = names
    if os.path.exists(id_path):
        with open(id_path, 'r') as f:
            positive_gene_id_set = [line.strip() for line in f.readlines()]
            positive_gene_id_set = set(positive_gene_id_set)
    else:
        positive_gene_id_set = convertname2ID(positive_gene_name_set)
        with open(id_path, 'w') as f:
            for line in positive_gene_id_set:
                f.write(line + '\n')

    return positive_gene_name_set, set(positive_gene_id_set)


def set_candidate_gene(filepath, k: int) -> set:
    df = pd.read_excel(filepath)

    dic = df.set_index(['id'])['count'].to_dict()
    can = [key for key, val in dic.items() if val >= k]
    return set(can), dic


def convertID2name(gene_id_list):
    mg = mygene.MyGeneInfo()
    genelist = mg.querymany(gene_id_list, scopes='entrezgene', fields='symbol', species='human')
    # remove the genes that have no entrezID
    genelist = [i for i in genelist if 'symbol' in i.keys()]
    genelist = [line['symbol'] for line in genelist]
    return set(genelist)


def convertname2ID(gene_name_list):
    mg = mygene.MyGeneInfo()
    genelist = mg.querymany(gene_name_list, scopes='symbol', fields='entrezgene', species='human')
    # remove the genes that have no entrezID
    genelist = [i for i in genelist if 'entrezgene' in i.keys()]
    genelist = [line['entrezgene'] for line in genelist]
    return set(genelist)
