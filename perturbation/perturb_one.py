import networkx as nx
from utils.graphutils import *
from preprocess.fragments_preprocess import *

def perturb_one(cpg,ast,fname):
    ast = cpg_to_ast(cpg,ast)

    fragment_list, fragment_graph = make_fragments(ast)

    ast_temp = nx.dfs_tree(ast, source=list(ast.nodes)[0])
    ast_temp.add_nodes_from((i,ast.nodes[i]) for i in ast_temp.nodes)

    if not len(fragment_list) == 1:

        random_tuple, degree = find_random_tuple_same_degree(fragment_graph,2)
        # random_tuple = [8,54]
        # print(random_tuple)

        # swap parents
        parent_a = list(fragment_graph.predecessors(random_tuple[0]))
        parent_b = list(fragment_graph.predecessors(random_tuple[1]))

        if parent_a and parent_b:

            c_as = list(ast_temp.successors(parent_a[0]))
            c_bs = list(ast_temp.successors(parent_b[0]))

            for c_a in c_as:
                ast_temp.remove_edge(parent_a[0],c_a)

            for c_b in c_bs:
                if (parent_b[0],c_b) in ast_temp.edges:
                    ast_temp.remove_edge(parent_b[0],c_b)

            for c_a in c_as:
                if c_a == random_tuple[0]:
                    ast_temp.add_edge(parent_a[0],random_tuple[1])
                else:
                    ast_temp.add_edge(parent_a[0],c_a)

            for c_b in c_bs:
                if c_b ==random_tuple[1]:
                    if (parent_b[0],random_tuple[0]) not in ast_temp.edges:
                        ast_temp.add_edge(parent_b[0],random_tuple[0])
                else:
                    if (parent_b[0],c_b) not in ast_temp.edges:
                        ast_temp.add_edge(parent_b[0],c_b)

        elif not parent_a:
            ast_temp.remove_edge(parent_b[0],random_tuple[1])
            ast_temp.add_edge(parent_b[0],random_tuple[0])

        elif not parent_b:
            ast_temp.remove_edge(parent_a[0],random_tuple[0])
            ast_temp.add_edge(parent_a[0],random_tuple[1])

        # swap grand children
        childs_a = list(fragment_graph.successors(random_tuple[0]))
        childs_b = list(fragment_graph.successors(random_tuple[1]))

        ast_temp = swap_grand_childs(childs_a,childs_b,ast_temp)
    else:
        print("ada nih",fname)

    ast_final = nx.dfs_tree(ast_temp, source=list(ast_temp.nodes)[0])
    ast_final.add_nodes_from((i,ast_temp.nodes[i]) for i in ast_final.nodes)

    return ast_final