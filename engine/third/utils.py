from skimage import color
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import os
import sklearn.neighbors as nn
import warnings
import configparser
# *****************************
# ***** Utility functions *****
# *****************************

def check_value(inds, val):
    ''' Check to see if an array is a single element equaling a particular value
    for pre-processing inputs in a function '''
    if(np.array(inds).size==1):
        if(inds==val):
            return True
    return False

def na(): # shorthand for new axis
    return np.newaxis

def flatten_nd_array(pts_nd,axis=1):
    ''' Flatten an nd array into a 2d array with a certain axis
    INPUTS
        pts_nd       N0xN1x...xNd array
        axis         integer
    OUTPUTS
        pts_flt     prod(N \ N_axis) x N_axis array     '''
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0,NDIM),np.array((axis))) # non axis indices
    NPTS = np.prod(SHP[nax])
    axorder = np.concatenate((nax,np.array(axis).flatten()),axis=0)
    pts_flt = pts_nd.transpose((axorder))
    pts_flt = pts_flt.reshape(NPTS,SHP[axis])
    return pts_flt

def unflatten_2d_array(pts_flt,pts_nd,axis=1,squeeze=False):
    ''' Unflatten a 2d array with a certain axis
    INPUTS
        pts_flt     prod(N \ N_axis) x M array
        pts_nd      N0xN1x...xNd array
        axis        integer
        squeeze     bool     if true, M=1, squeeze it out
    OUTPUTS
        pts_out     N0xN1x...xNd array        '''
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0,NDIM),np.array((axis))) # non axis indices
    NPTS = np.prod(SHP[nax])

    if(squeeze):
        axorder = nax
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        # print NEW_SHP
        # print pts_flt.shape
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)
    else:
        axorder = np.concatenate((nax,np.array(axis).flatten()),axis=0)
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        NEW_SHP.append(M)
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)

    return pts_out


# ***************************
# ***** SUPPORT CLASSES *****
# ***************************
class PriorFactor():
    ''' Class handles prior factor '''
    def __init__(self,alpha,gamma=0,verbose=False,priorFile=''):
        # INPUTS
        #   alpha           integer     prior correction factor, 0 to ignore prior, 1 to divide by prior, alpha to divide by prior**alpha
        #   gamma           integer     percentage to mix in uniform prior with empirical prior
        #   priorFile       file        file which contains prior probabilities across classes

        # settings
        self.alpha = alpha
        self.gamma = gamma
        self.verbose = verbose

        # empirical prior probability
        self.prior_probs = np.load(priorFile)

        # define uniform probability
        self.uni_probs = np.zeros_like(self.prior_probs)
        self.uni_probs[self.prior_probs!=0] = 1.
        self.uni_probs = self.uni_probs/np.sum(self.uni_probs)

        # convex combination of empirical prior and uniform distribution       
        self.prior_mix = (1-self.gamma)*self.prior_probs + self.gamma*self.uni_probs

        # set prior factor
        self.prior_factor = self.prior_mix**-self.alpha
        self.prior_factor = self.prior_factor/np.sum(self.prior_probs*self.prior_factor) # re-normalize

        # implied empirical prior
        self.implied_prior = self.prior_probs*self.prior_factor
        self.implied_prior = self.implied_prior/np.sum(self.implied_prior) # re-normalize

        if(self.verbose):
            self.print_correction_stats()

    def print_correction_stats(self):
        print('Prior factor correction:')
        print ('  (alpha,gamma) = (%.2f, %.2f)'%(self.alpha,self.gamma))
        print ('  (min,max,mean,med,exp) = (%.2f, %.2f, %.2f, %.2f, %.2f)'%(np.min(self.prior_factor),np.max(self.prior_factor),np.mean(self.prior_factor),np.median(self.prior_factor),np.sum(self.prior_factor*self.prior_probs)))

    def forward(self,data_ab_quant,axis=1):
        data_ab_maxind = np.argmax(data_ab_quant,axis=axis)
        corr_factor = self.prior_factor[data_ab_maxind]
        if(axis==0):
            return corr_factor[na(),:]
        elif(axis==1):
            return corr_factor[:,na(),:]
        elif(axis==2):
            return corr_factor[:,:,na(),:]
        elif(axis==3):
            return corr_factor[:,:,:,na()]

def _prior_boost(gt_ab_313):
  '''
  Args:
    gt_ab_313: (N, H, W, 313)
  Returns:
    prior_boost: (N, H, W, 1)
  '''
  enc_dir = './resources'
  gamma = 0.5
  alpha = 1.0

  pc = PriorFactor(alpha, gamma, priorFile=os.path.join(enc_dir, 'prior_probs.npy'))

  gt_ab_313 = np.transpose(gt_ab_313, (0, 3, 1, 2))
  prior_boost = pc.forward(gt_ab_313, axis=1)

  prior_boost = np.transpose(prior_boost, (0, 2, 3, 1))
  return prior_boost


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.expand_dims(np.max(x, axis=-1), axis=-1))
    return e_x / np.expand_dims(e_x.sum(axis=-1), axis=-1) # only difference


def decode(data_l, conv8_313, rebalance=1):
  """
  Args:
    data_l   : [1, height, width, 1]
    conv8_313: [1, height/4, width/4, 313]
  Returns:
    img_rgb  : [height, width, 3]
  """
  data_l = data_l + 50
  _, height, width, _ = data_l.shape
  data_l = data_l[0, :, :, :]
  conv8_313 = conv8_313[0, :, :, :]
  enc_dir = '../resources'
  conv8_313_rh = conv8_313 * rebalance
  class8_313_rh = softmax(conv8_313_rh)

  cc = np.load(os.path.join(enc_dir, 'pts_in_hull.npy'))
  
  data_ab = np.dot(class8_313_rh, cc)
  data_ab = resize(data_ab, (height, width))
  img_lab = np.concatenate((data_l, data_ab), axis=-1)
  img_rgb = color.lab2rgb(img_lab)

  return img_rgb

def get_data_l(image_path):
  """
  Args:
    image_path  
  Returns:
    data_l 
  """
  data = imread(image_path)
  data = data[None, :, :, :]
  img_lab = color.rgb2lab(data)
  img_l = img_lab[:, :, :, 0:1]
  data_l = img_l - 50
  data_l = data_l.astype(dtype=np.float32)
  return data, data_l

def process_config(conf_file):
  """process configure file to generate CommonParams, DataSetParams, NetParams 
  Args:
    conf_file: configure file path 
  Returns:
    CommonParams, DataSetParams, NetParams, SolverParams
  """
  common_params = {}
  dataset_params = {}
  net_params = {}
  solver_params = {}

  #configure_parser
  config = configparser.ConfigParser()
  config.read(conf_file)

  #sections and options
  for section in config.sections():
    #construct common_params
    if section == 'Common':
      for option in config.options(section):
        common_params[option] = config.get(section, option)
    #construct dataset_params
    if section == 'DataSet':
      for option in config.options(section):
        dataset_params[option] = config.get(section, option)
    #construct net_params
    if section == 'Net':
      for option in config.options(section):
        net_params[option] = config.get(section, option)
    #construct solver_params
    if section == 'Solver':
      for option in config.options(section):
        solver_params[option] = config.get(section, option)

  return common_params, dataset_params, net_params, solver_params