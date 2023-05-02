import os
import subprocess

INPUT_NETWORK_PATH = r'C:\Users\dell\PycharmProjects\riskgene\data\network\PPI-Network.txt'
EMB_ROOT_PATH = r'C:\Users\dell\PycharmProjects\riskgene\data\emb'


def node2vec(output_file_name, p=1, q=1, dim=128, length=80, num=10):
    output_file_path = os.path.join(EMB_ROOT_PATH, 'node2vec', output_file_name)
    command_line = [
        'python', '-m', 'openne',
        '--method', 'node2vec',
        '--input', INPUT_NETWORK_PATH,
        '--graph-format', 'edgelist',
        '--output', output_file_path,
        '--q', str(q),
        '--p', str(p),
        '--number-walks', str(num),
        '--walk-length', str(length),
        '--representation-size', str(dim)
    ]
    subprocess.call(command_line, shell=False)
    return


# PARAM = {
#     'dim': [64, 256, 512],
#     'length': [40],
#     'num': [20],
#     'q': [0.25, 0.5, 1, 2],
#     'p': [0.25, 0.5, 1, 2]
# }
#
# for dim in PARAM['dim']:
#     for q in PARAM['q']:
#         for p in PARAM['p']:
#             output_file_name = 'n2v' + \
#                                '_d' + str(dim) + \
#                                '_l' + '40' + \
#                                '_n' + "20" + \
#                                '_q' + str(q) + \
#                                '_p' + str(p) + \
#                                '.emb'
#             node2vec(output_file_name=output_file_name)


def grafac(output_file_name, dim=128):
    output_file_path = os.path.join(EMB_ROOT_PATH, 'grafac', output_file_name)
    command_line = [
        'python', '-m', 'openne', \
        '--method', 'gf',
        '--input', INPUT_NETWORK_PATH,
        '--graph-format', 'edgelist',
        '--output', output_file_path,
        '--representation-size', str(dim)
    ]
    subprocess.call(command_line, shell=False)
    return


PARAM_gf = {
    'dim': [64, 128, 256, 512]
}
for dim in PARAM_gf['dim']:
    output_file_name = 'gf' + \
                       '_d' + str(dim) + \
                       '.emb'
    grafac(output_file_name=output_file_name)

# import os
# import subprocess
# INPUT_NETWORK_PATH = r'C:\Users\dell\PycharmProjects\riskgene\data\network\PPI-Network.txt'
# EMB_ROOT_PATH = r'C:\Users\dell\PycharmProjects\riskgene\data\emb'
# PARAM = {
#     'dim': [64, 128, 256, 512],
#     'length': [20, 40, 80],
#     'num': [20, 40, 80]
# }
# dim=128
# length=80
# num=10
# q=1
# p=1
# output_file_name = 'n2v' + \
#                                '_d' + str(dim) + \
#                                '_l' + str(length) + \
#                                '_n' + str(num) + \
#                                '_q1_p_1' + \
#                                '.emb'
# output_file_path = os.path.join(EMB_ROOT_PATH, 'node2vec', output_file_name)
# command_line = [
#         'python', '-m', 'openne', \
#         '--method', 'node2vec',
#         '--input', INPUT_NETWORK_PATH,
#         '--graph-format', 'edgelist',
#         '--output', output_file_path,
#         '--q', str(q),
#         '--p', str(p),
#         '--number-walks', str(num),
#         '--walk-length', str(length),
#         '--representation-size', str(dim)
#     ]
# subprocess.call(command_line, shell=False)
