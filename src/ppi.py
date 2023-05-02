import mygene
import networkx as nx
import pandas as pd


class PPI:
    def __init__(self, ppi_network_path, gene_count_path, k):
        """
        Initializes the PPI class with the given parameters.
        :param ppi_network_path: str, path to the PPI network file
        :param gene_count_path: str, path to the gene count file
        :param k: int, threshold for gene count
        """
        self.network, self.gene_id = self.get_ppi_network(ppi_network_path)
        # self.id_to_name, self.name_to_id = self.mapping_from_id(gene_id_list=list(self.gene_id))
        # self.gene_name = self.convert_id_to_name(list(self.gene_id))
        self.positive_name = None
        self.positive_id = None
        self.risk_level = None
        self.set_candidate_gene(gene_count_path, k)

    @staticmethod
    def get_ppi_network(ppi_network_path):
        """
        Returns the PPI network and gene id set.
        :param ppi_network_path: str, path to the PPI network file
        :return: network: networkx.classes.graph.Graph, PPI network
                 gene_id: set, set of gene ids
        """
        network = nx.read_edgelist(ppi_network_path, delimiter=',')
        gene_id = set(network.nodes)
        return network, gene_id

    @staticmethod
    def mapping_from_id(gene_id_list):
        """
        Returns the mapping between gene id and gene name.
        :param gene_id_list: list, list of gene ids
        :return: id_to_name: dict, mapping from gene id to gene name
                 name_to_id: dict, mapping from gene name to gene id
        """
        mg = mygene.MyGeneInfo()
        gene_name_list = mg.querymany(gene_id_list, scopes='entrezgene', fields='symbol', species='human')
        gene_name_list = [i for i in gene_name_list if 'symbol' in i.keys()]
        id_to_name = {gene['query']: gene['symbol'] for gene in gene_name_list}
        name_to_id = {gene['symbol']: gene['query'] for gene in gene_name_list}
        return id_to_name, name_to_id

    @staticmethod
    def mapping_from_name(gene_name_list):
        """
        Returns the mapping between gene name and gene id.
        :param gene_name_list: list, list of gene names
        :return: id_to_name: dict, mapping from gene id to gene name
                 name_to_id: dict, mapping from gene name to gene id
        """
        mg = mygene.MyGeneInfo()
        gene_id_list = mg.querymany(gene_name_list, scopes='symbol', fields='entrezgene', species='human')
        gene_id_list = [i for i in gene_id_list if 'entrezgene' in i.keys()]
        id_to_name = {gene['query']: gene['entrezgene'] for gene in gene_id_list}
        name_to_id = {gene['entrezgene']: gene['query'] for gene in gene_id_list}
        return id_to_name, name_to_id

    @staticmethod
    def convert_id_to_name(gene_id_list):
        """
        Converts gene id to gene name.
        :param gene_id_list: list, list of gene ids
        :return: set, set of gene names
        """
        mg = mygene.MyGeneInfo()
        gene_name_list = mg.querymany(gene_id_list, scopes='entrezgene', fields='symbol', species='human')
        gene_name_list = [i for i in gene_name_list if 'symbol' in i.keys()]
        gene_name = [line['symbol'] for line in gene_name_list]
        return set(gene_name)

    @staticmethod
    def convert_name_to_id(gene_name_list):
        """
        Converts gene name to gene id.
        :param gene_name_list: list, list of gene names
        :return: set, set of gene ids
        """
        mg = mygene.MyGeneInfo()
        gene_id_ist = mg.querymany(gene_name_list, scopes='symbol', fields='entrezgene', species='human')
        gene_id_ist = [i for i in gene_id_ist if 'entrezgene' in i.keys()]
        gene_id = set([line['entrezgene'] for line in gene_id_ist])
        return set(gene_id)

    def set_candidate_gene(self, gene_count_path, k: int):
        """
        Sets the candidate genes based on the given gene count file and threshold.
        :param gene_count_path: str, path to the gene count file
        :param k: int, threshold for gene count
        """
        self.risk_level = pd.read_excel(gene_count_path, index_col='id')['count'].to_dict()
        self.positive_id = {key for key, val in self.risk_level.items() if val >= k}
        # self.positive_name = self.convert_id_to_name(list(self.positive_id))


