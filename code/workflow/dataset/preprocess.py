# This file is part of the Reproducible Open Benchmarks for Data Analysis
# Platform (ROB).
#
# Copyright (C) [2019-2020] NYU.
#
# ROB is free software; you can redistribute it and/or modify it under the
# terms of the MIT License; see LICENSE file for more details.

"""Collection of helper function for the default preprocessing step.

This is an adoped version of code in:
https://github.com/SebastianMacaluso/TreeNiN
"""

import logging
import numpy as np
import fastjet as fj
import os
import pickle


# -- Preprocessing Helper Functions (from preprocess_functions.py) --------------

def make_dictionary(tree, content, mass, pt, charge=None, abs_charge=None, muon=None):
    """Create a dictionary with all the jet tree info (topology, constituents
    features: eta, phi, pT, E, muon label). Keep only the leading jet.
    """
    jet = {}
    jet["root_id"] = 0
    # Labels for the jet constituents in the tree
    jet["tree"] = tree[0]
    # Use this format if using Pytorch
    jet["content"] = np.reshape(content[0],(-1,4))
    jet["mass"] = mass
    jet["pt"] = pt
    jet["energy"] = content[0][0, 3]
    # The jet is the first entry of content. And then we have (px,py,pz,E)
    px = content[0][0, 0]
    py = content[0][0, 1]
    pz = content[0][0, 2]
    p = (content[0][0, 0:3] ** 2).sum() ** 0.5
     # pseudorapidity eta
    eta = 0.5 * (np.log(p + pz) - np.log(p - pz))
    phi = np.arctan2(py, px)
    jet["eta"] = eta
    jet["phi"] = phi
    if charge:
        jet["charge"]=charge[0]
        jet["abs_charge"]=abs_charge[0]
    if muon:
        jet["muon"]=muon[0]
    return jet


def make_pseudojet(event):
    """Create a fastjet pseudojet from the 4-vector entries"""
    event_particles = list()
    for n,t in enumerate(event):
        # Data Format:(E,px,py,pz)
        # FastJet pseudojet format should be: (px,py,pz,E)
        temp=fj.PseudoJet(t[1],t[2],t[3],t[0])
        event_particles.append(temp)
    return event_particles


def make_tree_list(out_jet):
    """Create the lists with the trees."""
    jets_tree = list()
    for i in range(len(out_jet)):
        traversal_result = traverse_tree(out_jet[i], extra_info=False)
        tree, content, charge, abs_charge, muon = traversal_result
        tree=np.asarray([tree])
        tree=np.asarray([np.asarray(e).reshape(-1,2) for e in tree])
        content=np.asarray([content])
        content=np.asarray([np.asarray(e).reshape(-1,4) for e in content])
        mass=out_jet[i].m()
        pt=out_jet[i].pt()
        jets_tree.append((tree, content, mass, pt))
        if i > 0:
            logging.info('More than 1 reclustered jet')
        # TODO: Is it intended that the loop always returns after the first
        # entry???
        return jets_tree


def recluster(particles, Rjet,jetdef_tree):
    """Recluster the jet constituents"""
    # Recluster the jet constituents and access the clustering history
    #set up our jet definition and a jet selector
    if jetdef_tree=='antikt':
        tree_jet_def = fj.JetDefinition(fj.antikt_algorithm, Rjet)
    elif jetdef_tree=='kt':
        tree_jet_def = fj.JetDefinition(fj.kt_algorithm, Rjet)
    elif jetdef_tree=='CA':
        tree_jet_def = fj.JetDefinition(fj.cambridge_algorithm, Rjet)
    else:
        logging.info('Missing jet definition')
    # Recluster the jet constituents, they should give us only 1 jet if using
    # the original jet radius
    out_jet = fj.sorted_by_pt(tree_jet_def(particles))
    return out_jet


# -- Recursive Tree Traversal (from tree_cluster_hist.py) ----------------------

def traverse_tree(root, extra_info=True):
    """This function calls the recursive traverse function to make the trees
    starting from the root.

    Parameters
    ----------
    root: should be a fj.PseudoJet
    """
    tree = list()
    content = list()
    charge = list()
    abs_charge = list()
    muon = list()
    # We start from the root=jet 4-vector
    traverse_tree_rec(
        root,
        parent_id=-1,
        is_left=False,
        tree=tree,
        content=content,
        charge=charge,
        abs_charge=abs_charge,
        muon=muon,
        extra_info=extra_info
    )
    return tree, content, charge, abs_charge, muon


def traverse_tree_rec(
    root, parent_id, is_left, tree, content, charge, abs_charge, muon,
    extra_info=True
):
    """Recursive function to access fastjet clustering history and make the
    tree.

    Parameters
    ----------
    root: should be a fj.PseudoJet
    """
    id = int(len(tree)/2)
    if parent_id >= 0:
        if is_left:
            # We set the location of the lef child in the content array of the
            # 4-vector stored in content[parent_id]. So the left child will be
            # content[tree[2 * parent_id]].
            tree[2 * parent_id] = id
        else:
            # We set the location of the right child in the content array of
            # the 4-vector stored in content[parent_id]. So the right child
            # will be content[tree[2 * parent_id+1]]. This is correct because
            # with each 4-vector we increase the content array by one element
            # and the tree array by 2 elements. But then we take
            # id=tree.size()//2, so the id increases by 1. The left and right
            # children are added one after the other.
            tree[2 * parent_id + 1] = id
    # We insert 2 new nodes to the vector that constitutes the tree. In the
    # next iteration we will replace this 2 values with the location of the
    # parent of the new nodes
    tree.append(-1)
    tree.append(-1)
    # We fill the content vector with the values of the node
    content.append(root.px())
    content.append(root.py())
    content.append(root.pz())
    content.append(root.e())
    # We move from the root down until we get to the leaves. We do this
    # recursively
    if root.has_pieces():
        if extra_info:
            charge.append('inner')
            abs_charge.append('inner')
            muon.append('inner')
        # Call the function recursively
        pieces = root.pieces()
        # pieces[0] is the left child
        traverse_tree_rec(
            root=pieces[0],
            parent_id=id,
            is_left=True,
            tree=tree,
            content=content,
            charge=charge,
            abs_charge=abs_charge,
            muon=muon,
            extra_info=extra_info
        )
        # pieces[1] is the right child
        traverse_tree_rec(
            root=pieces[1],
            parent_id=id,
            is_left=False,
            tree=tree,
            content=content,
            charge=charge,
            abs_charge=abs_charge,
            muon=muon,
            extra_info=extra_info
        )
    else:
        if extra_info:
            charge.append(root.python_info().Charge)
            abs_charge.append(np.absolute(root.python_info().Charge))
            muon.append(root.python_info().Muon)


# -- Main preprocessing function (from toptag_reference_dataset_Tree.py) -------

def run(card_file, input_jets_file, out_file):
    """Load and recluster the jet constituents. Create binary trees with the
    clustering history of the jets and output a dictionary for each jet that
    contains the root_id, tree, content (constituents 4-momentum vectors),
    mass, pT, energy, eta and phi values (also charge, muon ID, etc. depending
    on the information contained in the dataset)
    """
    # Read card file and extract command list from file content
    with open(card_file) as f:
       commands=f.readlines()
    commands = [x.strip().split('#')[0].split() for x in commands]
    # Evaluate commands in the extracted list
    ptmin = -9999999.
    ptmax = 9999999.
    maxeta = 9999999.
    matchdeltaR = 9999999.
    mergedeltaR = 9999999.
    N_jets=np.inf
    for command in commands:
        if len(command)>=2:
            if command[0] == 'TRIMMING':
                Trimming = int(command[1])
            if command[0] == 'JETDEF':
                jetdef_tree = str(command[1])
            if command[0] == 'PTMIN':
                ptmin = float(command[1])
            if command[0] == 'PTMAX':
                ptmax = float(command[1])
            if command[0] == 'ETAMAX':
                etamax = float(command[1])
            if command[0] == 'MATCHDELTAR':
                matchdeltaR = float(command[1])
            if command[0] == 'MERGEDELTAR':
                mergedeltaR = float(command[1])
            if command[0] == 'RJET':
                Rjet = float(command[1])
            if command[0] == 'RTRIM':
                Rtrim = float(command[1])
            if command[0] == 'MINPTFRACTION':
                MinPtFraction = float(command[1])
            if command[0] == 'PREPROCESS':
                preprocess_label = command[1]
            if command[0] == 'MERGE':
                jetmergeflag = int(command[1])
            if command[0] == 'NPOINTS':
                npoints = int(command[1])
            if command[0] == 'DRETA':
                DReta = float(command[1])
            if command[0] == 'DRPHI':
                DRphi = float(command[1])
            if command[0] == 'NCOLORS':
                Ncolors = int(command[1])
            if command[0] == 'KAPPA':
                kappa = float(command[1])
    logging.info('Loading files for subjets')
    # Expected subjet array format is:
    # ([
    #    [[pTsubj1],[pTsubj2],...],
    #    [[etasubj1],[etasubj2],...],
    #    [[phisubj1],[phisubj2],...]]
    # ])
    logging.info('Loading subjet file {}'.format(input_jets_file))
    # ???
    images=[]
    jetmasslist=[]
    counter=0
    ## Loop over the data
    with open(input_jets_file, 'rb') as f:
        jets_file_content = pickle.load(f)
    reclustered_jets = list()
    # Loop over all the events
    for element in jets_file_content:
        event = make_pseudojet(element[0])
        label = element[1]
        # Recluster jet constituents
        out_jet = recluster(event, 0.8,'kt')
        # Create a dictionary with all the jet tree info (topology,
        # constituents features: eta, phi, pT, E, muon label)
        jets_tree = make_tree_list(out_jet)
        # Keep only the leading jet
        for tree, content, mass, pt in [jets_tree[0]]:
            jet = make_dictionary(tree, content, mass, pt)
            reclustered_jets.append((jet, label))
            counter += 1
    # Save output to file
    logging.info('out_filename={}'.format(out_file))
    with open(out_file, "wb") as f:
        pickle.dump(reclustered_jets, f, protocol=2)
    logging.info('counter={}'.format(counter))
