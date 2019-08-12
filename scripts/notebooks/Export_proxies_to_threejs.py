#!/usr/bin/env python
# coding: utf-8

# How can we load proxy in threejs when there is no proxy class. Here's a simple idea:
# 
# What if we just attach it to the human object and map it to the bones that the weighted vertices are attached to? That would make offsets work, I'm not sure if it would make distortion work. Might be worth testing
# - get a proxy
# - collect weighting of attached vertices, merge dups, get the 4 most significant
# - export as json object with skin weights, attach to human
# 

# In[8]:


# %load_ext autoreload
# %autoreload 2


# In[9]:


from tqdm import tqdm
from path import Path
import json
import sys
import time
from pprint import pprint

import numpy as np
import pandas as pd
import re
# get_ipython().run_line_magic('precision', '6')

from collections import OrderedDict
import logging
import tempfile
import os

basedir=Path(r'../../scripts/').abspath()


# In[ ]:





# In[10]:


# from .mh_helpers import clean, short_hash, clean_modifier

mhpath = Path(os.path.abspath("../vendor/makehuman-commandline/makehuman"))

#===============================================================================
# Import Makehuman resources, needs to be with makehuman dir as current dir
#===============================================================================

appcwd = os.path.abspath(os.curdir)
sys.path.append(mhpath)
sys.path.append(appcwd)
sys.path.append('.')

def getHuman():
    """Load a human model with modifiers."""
    with mhpath:
        # maxFaces *uint* Number of faces per vertex (pole), None for default (min 4)
        human = Human(files3d.loadMesh(
            getpath.getSysDataPath("3dobjs/base.obj"),
            maxFaces=5))
        # load modifiers onto human
        humanmodifier.mods_loaded = False
        modifiers = humanmodifier.loadModifiers(
            getpath.getSysDataPath('modifiers/modeling_modifiers.json'), human)
        return human

with mhpath:
    import makehuman
    oldpath = os.sys.path
    makehuman.set_sys_path()
    # make makehuman paths absolute by going through newest paths and making abs
    for i in range(len(os.sys.path)):
        p = os.sys.path[i]
        if p[0:2] == './':
            os.sys.path[i] = os.path.join(
                os.path.abspath('.'), p.replace('./', ''))
        else:
            break

    makehuman.init_logging()
    logging.getLogger().setLevel(logging.CRITICAL)
    #import image_pil as image_lib
    #
    import proxy as mhproxy
    import humanargparser
    import targets as mhtargets
    from human import Human
    import files3d
    import getpath
    import humanmodifier
    from core import G
    import headless
    import autoskinblender
    import export
    
    # Init console app
    with mhpath:
        G.app = headless.ConsoleApp()
    G.app.selectedHuman = human = getHuman()
    headless.OBJExporter = None
    headless.MHXExporter = None
    headless.MhxConfig = None
    humanargparser.mods_loaded = False
    
    from makehuman import LicenseInfo
    mh_licence=LicenseInfo()


# In[ ]:


# import proxies
def _listDataFiles(foldername,
                   extensions,
                   onlySysData=False,
                   recursive=True):
    with mhpath:  # sadly makehuman seems hardcoded
        if onlySysData:
            paths = [getpath.getSysDataPath(foldername)]
        else:
            paths = [getpath.getDataPath(foldername),
                     getpath.getSysDataPath(foldername)]
    return list(getpath.search(paths, extensions, recursive))

def clean(s):
    """Remove invalid characters."""
    s = re.sub('[^0-9a-zA-Z_]', '_', s)
    return s

with mhpath:
    mhproxy.ProxyTypes
    proxies = OrderedDict()
    for proxyType in mhproxy.ProxyTypes+['Genitals']:
        # print("proxyType = ", proxyType)
        files = list(_listDataFiles(proxyType.lower(),
                                         ['.proxy', '.mhclo']))

        # print("files = ", files)
        for f in files:
            if proxyType not in proxies.keys():
                proxies[proxyType] = OrderedDict()
            filesname = clean(os.path.splitext(os.path.basename(f))[0])
            proxies[proxyType][filesname] = f


# In[ ]:


# proxies['Clothes']


# In[ ]:


os.sys.path.append(basedir)
from convert_obj_three import convert_ascii, parse_mtl
from export_makehuman import material_to_mtl, vertex_weights_to_skin_weights, parse_skeleton_bones, NP_MH_Encoder, copyAndCompress


# # export obj

# In[ ]:





# In[ ]:


# also export

# TODO compress this, it's too big. Could convert to Int16, or limit decimal places, or restrict information
def get_proxy_metadata(prxy):
    """A function to get the metadata we wish to add to the json file"""
    data=dict(
        description=prxy.description
    )
    keys=[
#      '_material_file',
#      '_obj_file',
#      '_vertexBoneWeights_file',
     'basemesh',
     'deleteVerts',
     'description',
     'file',
    # #  'human',
     'license',
    # #  'material',
#      'max_pole',
#      'mtime',
#      'name',
    # #  'object',
     'offsets',
     'ref_vIdxs',
     'tags',
#      'tmatrix',
     'type',
     'uuid',
#      'uvLayers',
     'version',
#      'vertWeights',
#      'vertexBoneWeights',
     'weights',
     'z_depth'
    ]
    for key in keys:
        v = getattr(prxy,key)

        # keys must be strings
        if isinstance(v,dict):
            if v.keys() and not isinstance(v.keys()[0],str):
                v = dict((str(k),vv) for k,vv in v.items())
        if hasattr(v,'dtype') and v.dtype==np.dtype('bool'):
            v=v.astype(int)
        data[key]= v

    return data

# test it works and saves as json
# data=get_proxy_metadata(prxy)
# s=json.dumps(data, cls=SetEncoder)
# ss=json.loads(s)
# print(s)


# In[ ]:


logging.getLogger('export_makehuman').setLevel(logging.WARN)


# In[ ]:


proxies.keys()


# In[ ]:


outdir = Path(tempfile.mkdtemp(suffix='Convert_proxy_to_threejs_json'))
rig_file = 'data/rigs/default.mhskel'
print(outdir)
import material


# In[4]:



proxyGroups = [
#     # 'Proxymeshes',
    'Clothes',
    # 'Hair',
    'Eyes',
    # 'Eyebrows',
    # 'Eyelashes',
    # 'Teeth',
    # 'Tongue',
    'Genitals'
]
for group in proxyGroups:
    for proxy_name in proxies[group]:
        proxy_file = proxies[group][proxy_name]
        
        # ignore the debug proxies
        if proxy_file.find('__') != -1: continue
        
        # load data
        basehuman = getHuman()
        humanargparser.addRig(basehuman, rig_file)
        prxy = mhproxy.loadProxy(basehuman, proxy_file)
        mesh, obj = prxy.loadMeshAndObject(basehuman)

        print("prxy = ", prxy)
        # export
        infile = Path(prxy.obj_file)
        outfile = outdir.joinpath(group.lower()).joinpath(prxy.name,prxy.name + '.json')
        outfile.dirname().makedirs_p()
        if outfile.isfile(): continue
        
        convert_ascii(
            infile=infile,
            morphfiles='',
            colorfiles='',
            outfile=outfile,
            licence=json.dumps(LicenseInfo().asDict()),
            mtllib=material_to_mtl(prxy.material, texdir=os.path.dirname(outfile))
        )
        print(outfile)
        
        # some extra data to add to the file
        skeleton = basehuman.getSkeleton()
        bones = parse_skeleton_bones(skeleton)
        skeletonMetadata  = {
            "name": skeleton.name,
            "version": skeleton.version,
            "description": skeleton.description,
            "plane_map_strategy": skeleton.plane_map_strategy,
            "license": skeleton.license.asDict(),
        }
        vertex_weights = prxy.getVertexWeights(skeleton.getVertexWeights())
        influencesPerVertex = int(min(vertex_weights._nWeights, 4))
        skinIndices, skinWeights = vertex_weights_to_skin_weights(vertex_weights, skeleton, influencesPerVertex=influencesPerVertex)
        licence = json.dumps(mh_licence.asDict())
        
        # now add extra data to file
        metadata = get_proxy_metadata(prxy)
        metadata['skeletonMetadata']=skeletonMetadata
        data = json.load(open(outfile))
        data['metadata'].update(metadata)
        
        data['skinIndices']=skinIndices
        data['skinWeights']=skinWeights
        data['offsets']=prxy.offsets
        data['ref_vIdxs']=prxy.ref_vIdxs
        data['weights']=prxy.weights
        data['bones']=bones
        data['influencesPerVertex']=influencesPerVertex
        
        assert len(data['ref_vIdxs'])==len(data['weights'])
        assert len(data['ref_vIdxs'])>0
        assert len(data['offsets'])>0
        assert len(data['skinIndices'])==len(data['skinIndices'])
        assert len(data['skinIndices'])>0
        assert len(data['skinIndices'])%data['influencesPerVertex']==0
        
        
        # load alternative materials
        materials=[]
        if prxy.material_file:
            for material_file in Path(prxy.material_file).dirname().glob('*.mhmat'):
                material_name = str(Path(material_file).basename().splitext()[0])
                mat = material.Material(material_name)
                mat.fromFile(material_file)
                mtl = parse_mtl(material_to_mtl(mat, texdir=os.path.dirname(outfile)))
                mtl = mtl[mtl.keys()[0]]
                mtl['name']=material_name
                materials.append(mtl)
        data["materials"] = materials
        
        json.dump(data, open(outfile, 'w'), cls=NP_MH_Encoder, separators=(',', ':'))
        
        
        # copy thumbnail
        thumbnail = Path(infile.replace('.obj','.thumb'))
        if thumbnail.isfile():
            copyAndCompress(thumbnail,outfile.replace('.json','.thumb.png'))
        for thumbnail in [p.replace('.mhmat','.thumb') for p in Path(prxy.material_file).dirname().glob('*.mhmat')]:
            if Path(thumbnail).isfile():
                outnail = outfile.dirname().joinpath(Path(thumbnail).basename()).replace('.thumb','.thumb.png')
                copyAndCompress(thumbnail,outnail)
        
        print('nb_materials {nb_materials:} nb_bones {nb_bones:}'
          .format(
            nb_materials=len(materials),
            nb_bones=len(bones),
            nb_ref_vIdxs=len(data['ref_vIdxs'])
         ))


# In[ ]:




