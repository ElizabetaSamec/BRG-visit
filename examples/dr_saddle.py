
from compas.datastructures import Network
from compas.hpc import drx_numba
from compas.numerical import drx_numpy
from compas.viewers import VtkViewer

from numpy import abs
from numpy import array
from numpy import max


# Network

path = '/home/al/downloads/BRG-visit-master/examples/'
network = Network.from_json('{0}saddle.json'.format(path))
k_i = network.key_index()
uv_i = network.uv_index()
fixed = list(network.vertices_where({'is_fixed': True}))
network.update_default_edge_attributes({'E': 0.00, 'A': 1, 'ct': 't', 's0': 1})
network.set_vertices_attributes(fixed, {'B': [0, 0, 0]})

# Run

X, f, l = drx_numpy(network=network, tol=0.001, refresh=100, update=False, factor=2)
X, f, l = drx_numba(network=network, tol=0.001, summary=1, update=True, factor=2)

# Plot

fa = array(f)
fs = fa / max(abs(fa))
col = [[fi * 255, 100, 100] for fi in fs]

data = {
    'vertices': {k_i[key]: network.vertex_coordinates(key) for key in network.vertices()},
    'edges': [{'u': k_i[u], 'v': k_i[v], 'color': col[uv_i[(u, v)]]} for u, v in network.edges()],
    'fixed': [k_i[key] for key in fixed],
}

viewer = VtkViewer(data=data)
viewer.start()
