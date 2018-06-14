from __future__ import division
from __future__ import print_function

import sys
import copy

try:
    from numpy import mean
    from numpy import asarray
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve
    from scipy.sparse.linalg import norm as sm_norm
    from scipy.sparse.linalg import inv
    from numpy.linalg import cond
    from numpy import array as array_m
    from numpy import zeros as zero_m
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import splu
    from math import sqrt
    from scipy.sparse.linalg import cg
    from numpy import empty as empty_m
    from copy import copy


except ImportError:
    if 'ironpython' not in sys.version.lower():
        raise

from compas.numerical import normrow

__all__ = [
    'IIFDM'
]

no_callbacks = (None, None, None)

_step_x = 0
_step_y = 0
_step_z = 0

def _fx(x):
    global _step_x
    _step_x += 1


def _fy(x):
    global _step_y
    _step_y += 1


def _fz(x):
    global _step_z
    _step_z += 1


def zero_steps():
    global _step_x, _step_y, _step_z
    _step_x = _step_y = _step_z = 0


def steps():
    global _step_x, _step_y, _step_z
    return _step_x, _step_y, _step_z

step_counts = (_fx, _fy, _fz)

def all_of(lst):
    return xrange(len(lst))

def get_index(edges, keys):
    lis=[]
    for i in keys:
        for a, b in enumerate (edges):
            if b==i:
                lis.append(a)
    return lis

def get_values(lis, lvs):
    return [lvs[i] for i in lis]

def replace(l,index,values):
    a=0
    for i in index:
        l[i]=values[a]
        a=+1
    return l

def support_node_index(fixed):
    return fixed

def table_of_nodal_DsOF(v, fixed):
    dof_per_node = 1
    dof_table = zero_m((v, dof_per_node), dtype=int)
    for s in fixed:
        dof_table[support_node_index(s)] = -1
    ndof = 0
    j = 0
    for i in xrange(v):
        if dof_table[i, j] != -1:
            dof_table[i, j] = ndof
            ndof += 1
    return (ndof, dof_table)


def element_DsOF(el_nd_ids, doft):
    nn = len(el_nd_ids)
    nd = len(doft[0])
    ix = zero_m((nn, nd), dtype=int)
    for i in xrange(nn):
        ix[i] = doft[el_nd_ids[i]]
    return ix.flatten()


def mDmDf(ndof, doft, vertices, edges, q):
    ii = []
    jj = []
    aij = []
    mDf = zero_m((ndof, 3))
    for j in xrange(len(edges)):
        eni = edges[j]
        i0, i1 = element_DsOF(eni, doft)
        if i0 != -1:
            ii.append(i0)
            jj.append(i0)
            aij.append(q[j])
            if i1 != -1:
                ii.append(i0)
                jj.append(i1)
                aij.append(-q[j])
            else:
                nc = vertices[eni[1]]
                mDf[i0, 0] -= q[j] * nc[0]
                mDf[i0, 1] -= q[j] * nc[1]
                mDf[i0, 2] -= q[j] * nc[2]
        if i1 != -1:
            ii.append(i1)
            jj.append(i1)
            aij.append(q[j])
            if i0 != -1:
                ii.append(i1)
                jj.append(i0)
                aij.append(-q[j])
            else:
                nc = vertices[eni[0]]
                mDf[i1, 0] -= q[j] * nc[0]
                mDf[i1, 1] -= q[j] * nc[1]
                mDf[i1, 2] -= q[j] * nc[2]
    mD = csc_matrix((aij, (ii, jj)), shape=(ndof, ndof))
    return mD, mDf


def __solve_lin_syst_d(mD, mDf):
    mDlu = splu(mD)
    cord = mDlu.solve(mDf)
    return cord

def table_of_nodal_coordinates(cord, vertices, doft, ndc=None):
    if ndc is None:
        ndc = zero_m((doft.shape[0], cord.shape[1]))
        for i in xrange(doft.shape[0]):
            if doft[i, 0] == -1:
                ndc[i] = vertices[i]
    for i in xrange(doft.shape[0]):
        if doft[i, 0] != -1:
            ndc[i] = cord[doft[i, 0]]
    return ndc

def nodal_coordinates_d(mD, mDf, vertices, doft):
    cc = __solve_lin_syst_d(mD, -mDf)
    ndc = table_of_nodal_coordinates(cc, vertices, doft)
    return ndc


def _FDM_d(ndof, tdof, vertices, edges, q):
    mD, mDf = mDmDf(ndof, tdof, vertices, edges, q)
    xyz = nodal_coordinates_d(mD, mDf, vertices, tdof)
    return xyz


def __solve_lin_syst_it(mD, mDf, x0, lin_solver, i_tol=1e-5, cord=None, callbacks=no_callbacks):
    (xx, infox) = lin_solver(mD, mDf[:, 0], x0[:, 0], tol=i_tol, callback=callbacks[0])
    (yy, infoy) = lin_solver(mD, mDf[:, 1], x0[:, 1], tol=i_tol, callback=callbacks[1])
    (zz, infoz) = lin_solver(mD, mDf[:, 2], x0[:, 2], tol=i_tol, callback=callbacks[2])
    if cord is None:
        cord = empty_m(mDf.shape)
    cord[:, 0] = xx
    cord[:, 1] = yy
    cord[:, 2] = zz
    return cord


def nodal_coordinates_it(mD, mDf, x0, vertices, doft, lin_solver=cg, i_tol=1e-5, callbacks=no_callbacks):
    cc = __solve_lin_syst_it(mD, -mDf, x0, lin_solver, i_tol, None, callbacks)
    ndc = table_of_nodal_coordinates(cc, vertices, doft)
    return ndc


def _FDM_it(ndof, tdof, vertices, edges, q, lin_solver=cg, i_tol=1e-5, callbacks=no_callbacks):
    mD, mDf = mDmDf(ndof, tdof, vertices, edges, q)
    x0 = zero_m(mDf.shape)
    xyz = nodal_coordinates_it(mD, mDf, x0, vertices, tdof, lin_solver, i_tol, callbacks)
    return xyz


def FDM(vertices, edges, fixed, q, lin_solver=splu, i_tol=1e-5, callbacks=no_callbacks):
    v = len(vertices)
    ndof, tdof = table_of_nodal_DsOF(v, fixed)
    if lin_solver == splu:
        return _FDM_d(ndof, tdof, vertices, edges, q)
    else:
        return _FDM_it(ndof, tdof, vertices, edges, q, lin_solver, i_tol, callbacks)


############################
##  Constraints:
############################


def map_value_to_all_edges(edges,v):
    return [v for x in range(len(edges))]


def edge_constraints(ed_indices, ed_values):
    constrs = []
    constrs.extend(zip(ed_indices, ed_values))
    return constrs


def add_edge_constraints(el_indices, el_values, constrs):
    constrs.extend(zip(el_indices, el_values))
    return constrs


def remove_edge_constraints(el_indices, constrs):
    constrs2 = []
    for cnst in constrs:
        ei = cnst[0]
        if ei not in el_indices:
            constrs2.append(cnst)
    return constrs2


#################################
# Multistep - recalculation of q
#################################


def _multistepFDM_d(vertices, edges, fixed, q, fcs=[], lcs=[], l0cs=[], steps=250):
    v = len(vertices)
    q0 = copy(q)
    ndof, tdof = table_of_nodal_DsOF(v, fixed)
    xyz = _FDM_d(ndof, tdof, vertices, edges, q0)

    l = list_of_element_lengths(edges, xyz)
    f = list_of_element_forces(l, q0)

    for i in xrange(2, steps + 1):
        for fj in fcs:
            q0[fj[0]] = fj[1] / l[fj[0]]
        for lj in lcs:
            q0[lj[0]] = f[lj[0]] / lj[1]
        for l0j in l0cs:
            i0 = l0j[0]
            l0 = l0j[1][0]
            ae0 = l0j[1][1]
            ffj = f[i0]
            ll = (ae0 + ffj) * l0 / ae0
            q0[i0] = ffj / ll
        xyz = _FDM_d(ndof, tdof,vertices, edges, q0)
        l = list_of_element_lengths(edges, xyz)
        f = list_of_element_forces(l, q0)

    return (xyz, f, q0)


def _multistepFDM_it(vertices, edges, fixed, q, fcs=[], lcs=[], l0cs=[], steps=250,
                     lin_solver=cg, i_tol=1e-5, callbacks=no_callbacks, cond_num=False):
    v = len(vertices)
    q0 = copy(q)
    ndof, tdof = table_of_nodal_DsOF(v, fixed)
    mD, mDf = mDmDf(ndof, tdof, vertices, edges, q0)
    if cond_num == True:
        print ('01', '  max =', mD.max(), '  nrm1 =', sm_norm(inv(mD), 1), '  nrmF =', sm_norm(inv(mD)), '  cn =', cond(
            mD.A, 1))
    x0 = zero_m(mDf.shape)
    cc = __solve_lin_syst_it(mD, -mDf, x0, lin_solver, i_tol, None, callbacks)
    xyz = table_of_nodal_coordinates(cc, vertices, tdof)

    l = list_of_element_lengths(edges, xyz)
    f = list_of_element_forces(l, q0)

    for i in xrange(2, steps + 1):
        for fj in fcs:
            q0[fj[0]] = fj[1] / l[fj[0]]
        for lj in lcs:
            q0[lj[0]] = f[lj[0]] / lj[1]
        for l0j in l0cs:
            i0 = l0j[0]
            l0 = l0j[1][0]
            ae0 = l0j[1][1]
            ffj = f[i0]
            ll = (ae0 + ffj) * l0 / ae0
            q0[i0] = ffj / ll
        mD, mDf = mDmDf(ndof, tdof, vertices, edges, q0)
        if i % 10 == 1:
            print (i, '  max =', mD.max(), '  nrm1 =', sm_norm(inv(mD), 1), '  nrmF =', sm_norm(inv(mD)), '  cn =', cond(
                mD.A, 1))
        cc = __solve_lin_syst_it(mD, -mDf, x0, lin_solver, i_tol, None, callbacks)
        xyz = table_of_nodal_coordinates(cc, vertices, tdof)
        l = list_of_element_lengths(edges, xyz)
        f = list_of_element_forces(l, q0)

    return (xyz, f, q0)

def multistepFDM(vertices, edges, fixed, q, fcs=[], lcs=[], l0cs=[], steps=250, lin_solver=splu, i_tol=1e-5, callbacks=no_callbacks, cond_num=False):
    if lin_solver == splu:
        return _multistepFDM_d(vertices, edges, fixed, q, fcs, lcs, l0cs, steps)
    else:
        return _multistepFDM_it(vertices, edges, fixed, q, fcs, lcs, l0cs, steps,
                                    lin_solver, i_tol, callbacks, cond_num)

############################
## for Rhino - no cg
############################
def multistepFDM_cg (vertices, edges, fixed, q, fcs=[], lcs=[], l0cs=[], steps=250,
                     lin_solver=cg, i_tol=1e-5, callbacks=no_callbacks, cond_num=False):
    v = len(vertices)
    q0 = copy(q)
    ndof, tdof = table_of_nodal_DsOF(v, fixed)
    mD, mDf = mDmDf(ndof, tdof, vertices, edges, q0)
    if cond_num == True:
        print ('01', '  max =', mD.max(), '  nrm1 =', sm_norm(inv(mD), 1), '  nrmF =', sm_norm(inv(mD)), '  cn =', cond(
            mD.A, 1))
    x0 = zero_m(mDf.shape)
    cc = __solve_lin_syst_it(mD, -mDf, x0, lin_solver, i_tol, None, callbacks)
    xyz = table_of_nodal_coordinates(cc, vertices, tdof)

    l = list_of_element_lengths(edges, xyz)
    f = list_of_element_forces(l, q0)

    for i in xrange(2, steps + 1):
        for fj in fcs:
            q0[fj[0]] = fj[1] / l[fj[0]]
        for lj in lcs:
            q0[lj[0]] = f[lj[0]] / lj[1]
        for l0j in l0cs:
            i0 = l0j[0]
            l0 = l0j[1][0]
            ae0 = l0j[1][1]
            ffj = f[i0]
            ll = (ae0 + ffj) * l0 / ae0
            q0[i0] = ffj / ll
        mD, mDf = mDmDf(ndof, tdof, vertices, edges, q0)
        if i % 10 == 1:
            print (i, '  max =', mD.max(), '  nrm1 =', sm_norm(inv(mD), 1), '  nrmF =', sm_norm(inv(mD)), '  cn =', cond(
                mD.A, 1))
        cc = __solve_lin_syst_it(mD, -mDf, x0, lin_solver, i_tol, None, callbacks)
        xyz = table_of_nodal_coordinates(cc, vertices, tdof)
        l = list_of_element_lengths(edges, xyz)
        f = list_of_element_forces(l, q0)

    return (xyz, f, q0)

###########################
# With tolerance
###########################


def _multistepFDM_wtol_d(vertices, edges, fixed, q, fcs=[], lcs=[], l0cs=[], tol_f=1e-3, tol_l=1e-3, abs_error=True, steps=250):
    if abs_error == False:

        def force_diff(f, fs):
            df = []
            for ff in fs:
                df.append(abs(f[ff[0]] - ff[1]) / ff[1])
            return max(df)

        def length_diff(l, ls):
            dl = []
            for ll in ls:
                dl.append(abs(l[ll[0]] - ll[1]) / ll[1])
            return max(dl)

        def l0_diff(l, f, l0s):
            dl = []
            for ll in l0s:
                i0 = ll[0]
                l0 = ll[1][0]
                ae0 = ll[1][1]
                dl.append(abs(unstrained_length(l[i0], f[i0], ae0) - l0) / l0)
            return max(dl)

    else:

        def force_diff(f, fs):
            df = []
            for ff in fs:
                df.append(abs(f[ff[0]] - ff[1]))
            return max(df)

        def length_diff(l, ls):
            dl = []
            for ll in ls:
                dl.append(abs(l[ll[0]] - ll[1]))
            return max(dl)

        def l0_diff(l, f, l0s):
            dl = []
            for ll in l0s:
                i0 = ll[0]
                l0 = ll[1][0]
                ae0 = ll[1][1]
                dl.append(abs(unstrained_length(l[i0], f[i0], ae0) - l0))
            return max(dl)

    q0 = copy(q)
    v= len(vertices)
    ndof, tdof = table_of_nodal_DsOF(v, fixed)
    xyz = _FDM_d(ndof, tdof, vertices, edges, q0)

    l = list_of_element_lengths(edges, xyz)
    f = list_of_element_forces(l, q0)

    rfd = rld = rl0d = 0
    if fcs != []:
        rfd = force_diff(f, fcs)
    if lcs != []:
        rld = length_diff(l, lcs)
    if l0cs != []:
        rl0d = l0_diff(l, f, l0cs)
    print (1, rfd, rld)

    if rfd < tol_f and rld < tol_l and rl0d < tol_l:
        print ('steps:', 1)
        if fcs != []:
            print ('maximal force error: ', rfd)
        if lcs != []:
            print ('maximal length error: ', rld)
        if l0cs != []:
            print ('maximal unstrained length error: ', rl0d)
        return (xyz, f, q0)

    for i in xrange(2, steps + 1):
        if rfd >= tol_f:
            for fj in fcs:
                q0[fj[0]] = fj[1] / l[fj[0]]
        if rld >= tol_l:
            for lj in lcs:
                q0[lj[0]] = f[lj[0]] / lj[1]
        if rl0d >= tol_l:
            for l0j in l0cs:
                i0 = l0j[0]
                l0 = l0j[1][0]
                ae0 = l0j[1][1]
                ffj = f[i0]
                ll = (ae0 + ffj) * l0 / ae0
                q0[i0] = ffj / ll

        xyz = _FDM_d(ndof, tdof, vertices, edges, q0)
        l = list_of_element_lengths(edges, xyz)
        f = list_of_element_forces(l, q0)

        if fcs != []:
            rfd = force_diff(f, fcs)
        if lcs != []:
            rld = length_diff(l, lcs)
        if l0cs != []:
            rl0d = l0_diff(l, f, l0cs)
        if i % 10 == 1:
            print (i, rfd, rld)

        if rfd < tol_f and rld < tol_l and rl0d < tol_l:
            break

    if i == steps:
        print ('WARNING: maximal number of steps (' + str(steps) + ') reached !')
    else:
        print ('steps:', i)
    if fcs != []:
        print ('maximal force error: ', rfd)
    if lcs != []:
        print ('maximal length error: ', rld)
    if l0cs != []:
        print ('maximal unstrained length error: ', rl0d)

    return (xyz, f, q0)


def _multistepFDM_wtol_it(vertices, edges, fixed, q, fcs=[], lcs=[], l0cs=[], tol_f=1e-3, tol_l=1e-3, abs_error=True, lin_solver=cg,
                          i_tol=None, latest=True, steps=250, callbacks=no_callbacks):
    if abs_error == False:

        def force_diff(f, fs):
            df = []
            for ff in fs:
                df.append(abs(f[ff[0]] - ff[1]) / ff[1])
            return max(df)

        def length_diff(l, ls):
            dl = []
            for ll in ls:
                dl.append(abs(l[ll[0]] - ll[1]) / ll[1])
            return max(dl)

        def l0_diff(l, f, l0s):
            dl = []
            for ll in l0s:
                i0 = ll[0]
                l0 = ll[1][0]
                ae0 = ll[1][1]
                dl.append(abs(unstrained_length(l[i0], f[i0], ae0) - l0) / l0)
            return max(dl)

    else:

        def force_diff(f, fs):
            df = []
            for ff in fs:
                df.append(abs(f[ff[0]] - ff[1]))
            return max(df)

        def length_diff(l, ls):
            dl = []
            for ll in ls:
                dl.append(abs(l[ll[0]] - ll[1]))
            return max(dl)

        def l0_diff(l, f, l0s):
            dl = []
            for ll in l0s:
                i0 = ll[0]
                l0 = ll[1][0]
                ae0 = ll[1][1]
                dl.append(abs(unstrained_length(l[i0], f[i0], ae0) - l0))
            return max(dl)

    q0 = copy(q)
    v = len(vertices)
    ndof, tdof = table_of_nodal_DsOF(v, fixed)
    mD, mDf = mDmDf(ndof, tdof, vertices, edges, q0)
    x0 = zero_m(mDf.shape)
    nb = 5e-3
    i_tol = i_tol if i_tol != None else min(tol_f, tol_l) * nb
    print ('i_tol =', i_tol)
    cc = __solve_lin_syst_it(mD, -mDf, x0, lin_solver, i_tol, None, callbacks)
    xyz = table_of_nodal_coordinates(cc, vertices, tdof)

    l = list_of_element_lengths(edges, xyz)
    f = list_of_element_forces(l, q0)

    rfd = rld = rl0d = 0
    if fcs != []:
        rfd = force_diff(f, fcs)
    if lcs != []:
        rld = length_diff(l, lcs)
    if l0cs != []:
        rl0d = l0_diff(l, f, l0cs)
    print (1, rfd, rld)

    if rfd < tol_f and rld < tol_l and rl0d < tol_l:
        print ('steps:', 1)
        if fcs != []:
            print ('maximal force error: ', rfd)
        if lcs != []:
            print ('maximal length error: ', rld)
        if l0cs != []:
            print ('maximal unstrained length error: ', rl0d)
        return (xyz, f, q0)

    for i in xrange(2, steps + 1):
        if rfd >= tol_f:
            for fj in fcs:
                q0[fj[0]] = fj[1] / l[fj[0]]
        if rld >= tol_l:
            for lj in lcs:
                q0[lj[0]] = f[lj[0]] / lj[1]
        if rl0d >= tol_l:
            for l0j in l0cs:
                i0 = l0j[0]
                l0 = l0j[1][0]
                ae0 = l0j[1][1]
                ffj = f[i0]
                ll = (ae0 + ffj) * l0 / ae0
                q0[i0] = ffj / ll

        mD, mDf = mDmDf(ndof, tdof, vertices, edges, q0)
        x1 = cc if latest else x0
        cc = __solve_lin_syst_it(mD, -mDf, x1, lin_solver, i_tol, None, callbacks)
        xyz = table_of_nodal_coordinates(cc, vertices, tdof)
        l = list_of_element_lengths(edges, xyz)
        f = list_of_element_forces(l, q0)

        if fcs != []:
            rfd = force_diff(f, fcs)
        if lcs != []:
            rld = length_diff(l, lcs)
        if l0cs != []:
            rl0d = l0_diff(l, f, l0cs)
        if i % 10 == 1:
            print (i, rfd, rld)

        if rfd < tol_f and rld < tol_l and rl0d < tol_l:
            break

    if i == steps:
        print ('WARNING: maximal number of steps (' + str(steps) + ') reached !')
    else:
        print ('steps:', i)
    if fcs != []:
        print ('maximal force error: ', rfd)
    if lcs != []:
        print ('maximal length error: ', rld)
    if l0cs != []:
        print ('maximal unstrained length error: ', rl0d)

    return (xyz, f, q0)


def multistepFDM_wtol(vertices, edges, fixed, q, fcs=[], lcs=[], l0cs=[], tol_f=1e-3, tol_l=1e-3, abs_error=True, lin_solver=splu,
                      i_tol=None, latest=True, steps=250, callbacks=no_callbacks):
    if lin_solver == splu:
        return _multistepFDM_wtol_d(vertices, edges, fixed, q, fcs, lcs, l0cs,
                                    tol_f, tol_l, abs_error, steps)
    else:
        return _multistepFDM_wtol_it(vertices, edges, fixed, q, fcs, lcs, l0cs, tol_f, tol_l,
                                     abs_error, lin_solver, i_tol, latest, steps, callbacks)

####################
# for Rhino - no cg
####################

def multistepFDM_wtol_cg(vertices, edges, fixed, q, fcs=[], lcs=[], l0cs=[], tol_f=1e-3, tol_l=1e-3, abs_error=True, lin_solver=cg,
                          i_tol=None, latest=True, steps=250, callbacks=no_callbacks):
    if abs_error == False:

        def force_diff(f, fs):
            df = []
            for ff in fs:
                df.append(abs(f[ff[0]] - ff[1]) / ff[1])
            return max(df)

        def length_diff(l, ls):
            dl = []
            for ll in ls:
                dl.append(abs(l[ll[0]] - ll[1]) / ll[1])
            return max(dl)

        def l0_diff(l, f, l0s):
            dl = []
            for ll in l0s:
                i0 = ll[0]
                l0 = ll[1][0]
                ae0 = ll[1][1]
                dl.append(abs(unstrained_length(l[i0], f[i0], ae0) - l0) / l0)
            return max(dl)

    else:

        def force_diff(f, fs):
            df = []
            for ff in fs:
                df.append(abs(f[ff[0]] - ff[1]))
            return max(df)

        def length_diff(l, ls):
            dl = []
            for ll in ls:
                dl.append(abs(l[ll[0]] - ll[1]))
            return max(dl)

        def l0_diff(l, f, l0s):
            dl = []
            for ll in l0s:
                i0 = ll[0]
                l0 = ll[1][0]
                ae0 = ll[1][1]
                dl.append(abs(unstrained_length(l[i0], f[i0], ae0) - l0))
            return max(dl)

    q0 = copy(q)
    v = len(vertices)
    ndof, tdof = table_of_nodal_DsOF(v, fixed)
    mD, mDf = mDmDf(ndof, tdof, vertices, edges, q0)
    x0 = zero_m(mDf.shape)
    nb = 5e-3
    i_tol = i_tol if i_tol != None else min(tol_f, tol_l) * nb
    print ('i_tol =', i_tol)
    cc = __solve_lin_syst_it(mD, -mDf, x0, lin_solver, i_tol, None, callbacks)
    xyz = table_of_nodal_coordinates(cc, vertices, tdof)

    l = list_of_element_lengths(edges, xyz)
    f = list_of_element_forces(l, q0)

    rfd = rld = rl0d = 0
    if fcs != []:
        rfd = force_diff(f, fcs)
    if lcs != []:
        rld = length_diff(l, lcs)
    if l0cs != []:
        rl0d = l0_diff(l, f, l0cs)
    print (1, rfd, rld)

    if rfd < tol_f and rld < tol_l and rl0d < tol_l:
        print ('steps:', 1)
        if fcs != []:
            print ('maximal force error: ', rfd)
        if lcs != []:
            print ('maximal length error: ', rld)
        if l0cs != []:
            print ('maximal unstrained length error: ', rl0d)
        return (xyz, f, q0)

    for i in xrange(2, steps + 1):
        if rfd >= tol_f:
            for fj in fcs:
                q0[fj[0]] = fj[1] / l[fj[0]]
        if rld >= tol_l:
            for lj in lcs:
                q0[lj[0]] = f[lj[0]] / lj[1]
        if rl0d >= tol_l:
            for l0j in l0cs:
                i0 = l0j[0]
                l0 = l0j[1][0]
                ae0 = l0j[1][1]
                ffj = f[i0]
                ll = (ae0 + ffj) * l0 / ae0
                q0[i0] = ffj / ll

        mD, mDf = mDmDf(ndof, tdof, vertices, edges, q0)
        x1 = cc if latest else x0
        cc = __solve_lin_syst_it(mD, -mDf, x1, lin_solver, i_tol, None, callbacks)
        xyz = table_of_nodal_coordinates(cc, vertices, tdof)
        l = list_of_element_lengths(edges, xyz)
        f = list_of_element_forces(l, q0)

        if fcs != []:
            rfd = force_diff(f, fcs)
        if lcs != []:
            rld = length_diff(l, lcs)
        if l0cs != []:
            rl0d = l0_diff(l, f, l0cs)
        if i % 10 == 1:
            print (i, rfd, rld)

        if rfd < tol_f and rld < tol_l and rl0d < tol_l:
            break

    if i == steps:
        print ('WARNING: maximal number of steps (' + str(steps) + ') reached !')
    else:
        print ('steps:', i)
    if fcs != []:
        print ('maximal force error: ', rfd)
    if lcs != []:
        print ('maximal length error: ', rld)
    if l0cs != []:
        print ('maximal unstrained length error: ', rl0d)

    return (xyz, f, q0)


#########################################
# Inexact - containing l0 in optimisation
#########################################

def multistepFDM_inexact(nodes, elems, supports, qs, fcs=[], lcs=[], l0cs=[],
                         tol_f=1e-3, tol_l=1e-3, abs_error=True, lin_solver=cg,
                         i_tol_min=None, i_tol_max=0.1, damping=1e-2,
                         latest=True, steps=250, callbacks=no_callbacks):
    if abs_error == False:

        def force_diff(f, fs):
            df = []
            for ff in fs:
                df.append(abs(f[ff[0]] - ff[1]) / ff[1])
            return max(df)

        def length_diff(l, ls):
            dl = []
            for ll in ls:
                dl.append(abs(l[ll[0]] - ll[1]) / ll[1])
            return max(dl)

        def l0_diff(l, f, l0s):
            dl = []
            for ll in l0s:
                i0 = ll[0]
                l0 = ll[1][0]
                ae0 = ll[1][1]
                dl.append(abs(unstrained_length(l[i0], f[i0], ae0) - l0) / l0)
            return max(dl)

    else:

        def force_diff(f, fs):
            df = []
            for ff in fs:
                df.append(abs(f[ff[0]] - ff[1]))
            return max(df)

        def length_diff(l, ls):
            dl = []
            for ll in ls:
                dl.append(abs(l[ll[0]] - ll[1]))
            return max(dl)

        def l0_diff(l, f, l0s):
            dl = []
            for ll in l0s:
                i0 = ll[0]
                l0 = ll[1][0]
                ae0 = ll[1][1]
                dl.append(abs(unstrained_length(l[i0], f[i0], ae0) - l0))
            return max(dl)

    qs0 = copy(qs)
    ndof, tdof = table_of_nodal_DsOF(len(nodes), supports)
    mD, mDf = mDmDf(ndof, tdof, nodes, elems, qs0)
    x0 = zero_m(mDf.shape)
    # nb1 = max (map (bind_2nd (v_norm, inf), mDf))
    # nb1 = 1/nb1 if nb1 > 100 else 1e-2
    # nb1 /= 2*sqrt (3.)
    nb = 5e-3
    i_tol0 = i_tol_min if i_tol_min != None else min(tol_f, tol_l) * nb
    print ('i_tol_min =', i_tol0)
    cc = __solve_lin_syst_it(mD, -mDf, x0, lin_solver, i_tol0, None, callbacks)
    nc = table_of_nodal_coordinates(cc, nodes, tdof)

    l = list_of_element_lengths(elems, nc)
    f = list_of_element_forces(l, qs0)

    rfd = rld = rl0d = 1e-13
    if fcs != []:
        rfd1 = rfd
        rfd = force_diff(f, fcs)
    if lcs != []:
        rld1 = rld
        rld = length_diff(l, lcs)
    if l0cs != []:
        rl0d1 = rl0d
        rl0d = l0_diff(l, f, l0cs)
    print(1, i_tol0, rfd, rld, rl0d)

    if rfd < tol_f and rld < tol_l and rl0d < tol_l:
        print ('steps:', 1)
        if fcs != []:
            print ('maximal force error: ', rfd)
        if lcs != []:
            print ('maximal length error: ', rld)
        if l0cs != []:
            print ('maximal unstrained length error: ', rl0d)
        return (nc, f, qs0)

    i_tol1 = i_tol_max
    gamma_f = i_tol0 * (1 - sqrt(tol_f)) / tol_f ** 2
    gamma_l = i_tol0 * (1 - sqrt(tol_l)) / tol_l **2

    for i in xrange(2, steps + 1):
        if rfd >= tol_f:
            for fj in fcs:
                qs0[fj[0]] = fj[1] / l[fj[0]]
        if rld >= tol_l:
            for lj in lcs:
                qs0[lj[0]] = f[lj[0]] / lj[1]
        if rl0d >= tol_l:
            for l0j in l0cs:
                i0 = l0j[0]
                l0 = l0j[1][0]
                ae0 = l0j[1][1]
                ffj = f[i0]
                ll = (ae0 + ffj) * l0 / ae0
                qs0[i0] = ffj / ll

        mD, mDf = mDmDf(ndof, tdof, nodes, elems, qs0)
        # nb1 = max (map (bind_2nd (v_norm, inf), -mDf))
        # nb1 = 1/nb1 if nb1 > 100 else 1e-2
        # nb1 /= 2*sqrt (3.)
        # i_tol0 = i_tol if i_tol != None else min (tol_f, tol_l) * nb
        x1 = cc if latest else x0
        minrfd = minrld = minrl0d = 0.0
        if fcs != []:
            minrfd = min(damping * rfd ** 3 / rfd1 ** 2, gamma_f * rfd ** 2)
        if lcs != []:
            minrld = min(damping * rld ** 3 / rld1 ** 2, gamma_l * rld ** 2)
        if l0cs != []:
            minrl0d = min(damping * rl0d ** 3 / rl0d1 ** 2, gamma_l * rl0d ** 2)
        i_tol1 = min(i_tol_max, i_tol1, max(minrfd, minrld, minrl0d, i_tol0))
        cc = __solve_lin_syst_it(mD, -mDf, x1, lin_solver, i_tol1, None, callbacks)
        # cc = __solve_lin_syst_it (mD, -mDf, x1, lin_solver, i_tol1, x1, callbacks)
        nc = table_of_nodal_coordinates(cc, nodes, tdof)
        # nc = table_of_nodal_coordinates (cc, nodes, tdof, nc)
        l = list_of_element_lengths(elems, nc)
        f = list_of_element_forces(l, qs0)

        if fcs != []:
            rfd1 = rfd
            rfd = force_diff(f, fcs)
        if lcs != []:
            rld1 = rld
            rld = length_diff(l, lcs)
        if l0cs != []:
            rl0d = l0_diff(l, f, l0cs)

        if i % 10 == 1:
            print (i, i_tol1, rfd, rld, rl0d)

        if rfd < tol_f and rld < tol_l and rl0d < tol_l:
            break

    if i == steps:
        print ('WARNING: maximal number of steps (' + str(steps) + ') reached !')
    else:
        print ('steps:', i)
    if fcs != []:
        print ('maximal force error: ', rfd)
    if lcs != []:
        print ('maximal length error: ', rld)
    if l0cs != []:
        print ('maximal unstrained length error: ', rl0d)

    return (nc, f, qs0)

###############################
#without l0 in optimisation
###############################

# def multistepFDM_inexact(vertices, edges, fixed, qs, fcs=[], lcs=[], l0cs=[],
#                          tol_f=1e-3, tol_l=1e-3, abs_error=True, lin_solver=cg,
#                          i_tol_min=None, i_tol_max=0.1, damping=1e-2,
#                          latest=True, steps=250, callbacks=no_callbacks):
#     if abs_error == False:
#
#         def force_diff(f, fs):
#             df = []
#             for ff in fs:
#                 df.append(abs(f[ff[0]] - ff[1]) / ff[1])
#             return max(df)
#
#         def length_diff(l, ls):
#             dl = []
#             for ll in ls:
#                 dl.append(abs(l[ll[0]] - ll[1]) / ll[1])
#             return max(dl)
#
#         def l0_diff(l, f, l0s):
#             dl = []
#             for ll in l0s:
#                 i0 = ll[0]
#                 l0 = ll[1][0]
#                 ae0 = ll[1][1]
#                 dl.append(abs(unstrained_length(l[i0], f[i0], ae0) - l0) / l0)
#             return max(dl)
#
#     else:
#
#         def force_diff(f, fs):
#             df = []
#             for ff in fs:
#                 df.append(abs(f[ff[0]] - ff[1]))
#             return max(df)
#
#         def length_diff(l, ls):
#             dl = []
#             for ll in ls:
#                 dl.append(abs(l[ll[0]] - ll[1]))
#             return max(dl)
#
#         def l0_diff(l, f, l0s):
#             dl = []
#             for ll in l0s:
#                 i0 = ll[0]
#                 l0 = ll[1][0]
#                 ae0 = ll[1][1]
#                 dl.append(abs(unstrained_length(l[i0], f[i0], ae0) - l0))
#             return max(dl)
#
#     qs0 = copy(qs)
#     ndof, tdof = table_of_nodal_DsOF(len(vertices), fixed)
#     mD, mDf = mDmDf(ndof, tdof, vertices, edges, qs0)
#     x0 = zero_m(mDf.shape)
#     nb = 5e-3
#     i_tol0 = i_tol_min if i_tol_min != None else min(tol_f, tol_l) * nb
#     print ('i_tol_min =', i_tol0)
#     cc = __solve_lin_syst_it(mD, -mDf, x0, lin_solver, i_tol0, None, callbacks)
#     xyz = table_of_nodal_coordinates(cc, vertices, tdof)
#
#     l = list_of_element_lengths(edges, xyz)
#     f = list_of_element_forces(l, qs0)
#
#     rfd = rld = rl0d = 1e-16
#     rfd1 = rld1 = 1e-16
#     if fcs != []:
#         rfd = force_diff(f, fcs)
#     if lcs != []:
#         rld = length_diff(l, lcs)
#     if l0cs != []:
#         rl0d = l0_diff(l, f, l0cs)
#     print (1, i_tol0, rfd, rld)
#
#     if rfd < tol_f and rld < tol_l and rl0d < tol_l:
#         print ('steps:', 1)
#         if fcs != []:
#             print ('maximal force error: ', rfd)
#         if lcs != []:
#             print ('maximal length error: ', rld)
#         if l0cs != []:
#             print ('maximal unstrained length error: ', rl0d)
#         return (xyz, f)
#
#     i_tol1 = i_tol_max
#     gamma_f = i_tol0 * (1 - sqrt(tol_f)) / tol_f ** 2
#     gamma_l = i_tol0 * (1 - sqrt(tol_l)) / tol_l ** 2
#
#     for i in xrange(2, steps + 1):
#         if rfd >= tol_f:
#             for fj in fcs:
#                 qs0[fj[0]] = fj[1] / l[fj[0]]
#         if rld >= tol_l:
#             for lj in lcs:
#                 qs0[lj[0]] = f[lj[0]] / lj[1]
#         if rl0d >= tol_l:
#             for l0j in l0cs:
#                 i0 = l0j[0]
#                 l0 = l0j[1][0]
#                 ae0 = l0j[1][1]
#                 ffj = f[i0]
#                 ll = (ae0 + ffj) * l0 / ae0
#                 qs0[i0] = ffj / ll
#
#         mD, mDf = mDmDf(ndof, tdof, vertices, edges, qs0)
#         x1 = cc if latest else x0
#         if rld == 0:
#             i_tol1 = min(i_tol_max, i_tol1,
#                          max(min(damping * rfd ** 3 / rfd1 ** 2, gamma_f * rfd ** 2), i_tol0))
#         else:
#             i_tol1 = min(i_tol_max, i_tol1,
#                          max(min(damping * rfd ** 3 / rfd1 ** 2, gamma_f * rfd ** 2),
#                              min(damping * rld ** 3 / rld1 ** 2, gamma_l * rld ** 2), i_tol0))
#         cc = __solve_lin_syst_it(mD, -mDf, x1, lin_solver, i_tol1, None, callbacks)
#         nc = table_of_nodal_coordinates(cc, vertices, tdof)
#         l = list_of_element_lengths(edges, nc)
#         f = list_of_element_forces(l, qs0)
#
#         if fcs != []:
#             rfd1 = rfd
#             rfd = force_diff(f, fcs)
#         if lcs != []:
#             rld1 = rld
#             rld = length_diff(l, lcs)
#         if l0cs != []:
#             rl0d = l0_diff(l, f, l0cs)
#
#         if i % 10 == 1:
#             print (i, i_tol1, rfd, rld)
#
#         if rfd < tol_f and rld < tol_l and rl0d < tol_l:
#             break
#
#     if i == steps:
#         print ('WARNING: maximal number of steps (' + str(steps) + ') reached !')
#     else:
#         print ('steps:', i)
#     if fcs != []:
#         print ('maximal force error: ', rfd)
#     if lcs != []:
#         print ('maximal length error: ', rld)
#     if l0cs != []:
#         print ('maximal unstrained length error: ', rl0d)
#
#     return (nc, f, qs0)


###############################
# Forces and lengths
###############################


def ith_node_coords(ni, vertices):
    return vertices[ni]


def element3D_length(el_nd_coords):
    [[x1, y1, z1], [x2, y2, z2]] = el_nd_coords
    return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1))


def element_node_indices(el):
    return el


def element_node_coords(el, vertices):
    return [ith_node_coords(i, vertices) for i in element_node_indices(el)]


def list_of_element_lengths(edges, vertices):
    return [element3D_length(element_node_coords(el, vertices)) for el in edges]


def list_of_element_forces(lengths, q):
    return map(lambda x, y: x * y, lengths, q)

def unstrained_length(l, f, ae):
    return ae * l / (ae + f)


def list_of_unstrained_lengths(ll, ff, ae):
    return map(lambda x, y, z: z * x / (z + y), ll, ff, ae)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':

    import compas

    from compas.datastructures import Network
    from compas.viewers import NetworkViewer

    from compas.numerical import fd_numpy

    network = Network.from_obj(compas.get('saddle.obj'))

    dva = {'is_anchor': False, 'px': 0.0, 'py': 0.0, 'pz': 0.0}
    dea = {'q': 1.0}

    network.update_default_vertex_attributes(dva)
    network.update_default_edge_attributes(dea)

    for key in network.vertices():
        network.vertex[key]['is_anchor'] = network.is_vertex_leaf(key)

    key_index = network.key_index()

    xyz = network.get_vertices_attributes(('x', 'y', 'z'))
    loads = network.get_vertices_attributes(('px', 'py', 'pz'))
    q = network.get_edges_attribute('q')
    fixed = network.vertices_where({'is_anchor': True})
    fixed = [key_index[key] for key in fixed]
    edges = [(key_index[u], key_index[v]) for u, v in network.edges()]

    xyz = FDM(xyz, edges, fixed, q)

    for key, attr in network.vertices(True):
        index = key_index[key]
        attr['x'] = xyz[index, 0]
        attr['y'] = xyz[index, 1]
        attr['z'] = xyz[index, 2]

    viewer = NetworkViewer(network)
    viewer.setup()
    viewer.show()