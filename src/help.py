######################################
## For Rhino - calling without XFunc
######################################

def get_index(edges, keys):
    lis=[]
    for i in keys:
        for a, b in enumerate (edges):
            if b==i:
                lis.append(a)
    return lis

def get_values(lis, lvs):
    return [lvs[i] for i in lis]


########################
## Forces and lengths:
########################

def ith_node_coords(ni, vertices):
    return vertices[ni]


def element_node_indices(el):
    return el


def element_node_coords(el, vertices):
    return [ith_node_coords(i, vertices) for i in element_node_indices(el)]


def list_of_element_forces(lengths, q):
    return map(lambda x, y: x * y, lengths, q)

def unstrained_length(l, f, ae):
    return ae * l / (ae + f)

def list_of_unstrained_lengths(ll, ff, ae):
    return map(lambda x, y, z: z * x / (z + y), ll, ff, ae)

#################
## Constraints:
#################


def map_value_to_all_edges(edges,v):
    return [v for x in range(len(edges))]


def edge_constraints(ed_indices, ed_values):
    constrs = []
    constrs.extend(zip(ed_indices, ed_values))
    return constrs


def add_element_constraints(el_indices, el_values, constrs):
    constrs.extend(zip(el_indices, el_values))
    return constrs


def remove_element_constraints(el_indices, constrs):
    constrs2 = []
    for cnst in constrs:
        ei = cnst[0]
        if ei not in el_indices:
            constrs2.append(cnst)
    return constrs2