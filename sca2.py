#!/usr/bin/python
# -*- coding: utf-8 -*-
print('importing my sca functions...')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import butter, lfilter, buttord
from scipy import signal
from numpy.fft import fft, ifft, fft2, ifft2, fftshift
import scipy
import scipy.stats
import scipy.io as spio
from scipy.io import loadmat
from scipy.signal import freqz
from scipy.signal import butter, lfilter
from scipy.signal import buttord
import sys
import os, re
import math
import time
import datetime
from sklearn.metrics import classification_report
from sklearn import cluster, datasets
import scipy
import scipy.stats
from scipy.stats import multivariate_normal
from sklearn.neighbors import KNeighborsClassifier as knn
import scipy.cluster.hierarchy as hcluster
import pandas as pd
import importlib # for reload function
from numpy.lib.stride_tricks import as_strided
import my_aes

exec(open('global_settings.py').read())
exec(open('def_design.py').read())
#klen=256;is_gpu=0;pmdir='./powerModel/';design_ext='AESP';fs=5e09;

def cov(x, y):
    # Find the covariance between two 1D lists (x and y).
    # Note that var(x) = cov(x, x)
    return np.cov(x, y)[0][1]

# Useful utilities
sbox=(
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16)

inv_sbox=(
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D)

shift_row=(
	0, 4, 8, 12,
	5, 9, 13, 1,
	10, 14, 2, 6,
	15, 3, 7, 11)

inv_shift_row=(
	0, 4, 8, 12,
	13, 1, 5, 9,
	10, 14, 2, 6,
	7, 11, 15, 3)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    #    N, Wn = buttord([low, high], [0.8*low, 1.2*high], 1, 60, True)
    N, Wn = buttord(wp=[low, high], ws=[0.8*low, 1.2*high], gpass=1, gstop=60)
    #print(N)
    b, a = butter(order, Wn, btype='band')
    #    b, a = butter(order, [low, high], [low*0.8, high*1.2], gpass=1, gstop=60, btype='band')
    #    b, a = butter(order, [low, high], btype='band')
    return b, a

def trace_collector(keyIndex, NoofTraces, attackType, collectTraces):
	if attackType=='public':
		keymsg_file=public_keymsg_file
		index_file=public_index_file
		NoofTraces=publicDB_NoofTraces;
	elif attackType=='template':
		keymsg_file=template_keymsg_file
		index_file=template_index_file
		NoofTraces=templateDB_NoofTraces;
	
	tsta=time.time()
	index_file_handle=open(index_file, 'r')
	index_list=index_file_handle.readlines()
	index_file_handle.close()
	
	all_traces_dict={}
	all_ct_dict={}
	
	for t in range(len(index_list)):
		index_line=index_list[t].split(' ');
		key=index_line[0]; msg=index_line[1]; ct=index_line[2]; trace_file=index_line[3];
		all_traces_dict[key, msg]=trace_file.rstrip();
		all_ct_dict[key, msg]=ct;
		#print(key, msg, ct);
	
	tsto=time.time(); print('elapsed time for dict creation: {}'.format(tsto-tsta)); tsta=time.time()
	
	keymsg_file_handle=open(keymsg_file, 'r')
	keymsg_list=keymsg_file_handle.readlines()
	keymsg_file_handle.close()
	
	#for t in range(len(keymsg_list)):
	
	traceArray=np.zeros((NoofTraces, NoofSamples), dtype=np.float16)
	ct_dict={}
	keymsgct_dict={}
	if attackType=='public':
		keymsg_start=keyIndex*publicDB_NoofTraces
		keymsg_end=keymsg_start+NoofTraces
	else:
		keymsg_start=0
		keymsg_end=templateDB_NoofTraces
	for t in range(keymsg_start, keymsg_end):
		line=keymsg_list[t].split('\t');
		key=line[0].rstrip(); msg=line[1].rstrip();
		if collectTraces:
			trace_file=all_traces_dict[key, msg];
			if attackType=='public':
				traceArray[t-keymsg_start,:]=pd.read_csv(public_idir+trace_file, header=None).values[24:].reshape(1, NoofSamples)
			else:
				traceArray[t-keymsg_start,:]=pd.read_csv(template_idir+trace_file, header=None).values[24:].reshape(1, NoofSamples)
		ct_dict[t-keymsg_start]=all_ct_dict[key,msg]
		keymsgct_dict[t-keymsg_start]=[key, msg, all_ct_dict[key,msg]]
		#print(keymsgct_dict[t-keymsg_start])
		#print(key, msg, all_ct_dict[key, msg])
	if attackType=='public':
		IKey=key
	
	tsto=time.time(); print('elapsed time for traceArray creation: {}'.format(tsto-tsta)); tsta=time.time()

	return (keymsgct_dict, traceArray)

# a and b are expected to be integers
def hamming(a,b):
	return bin(a^b).count('1')

def hamming2(s1, s2):
    """Calculate the Hamming distance between two bit strings"""
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def last_round_power_model(keymsgct_dict, keyIndex, NoofTraces, attackType, debug, verbose):	
	KEY_NAME='powerModel'
	trace_interval=10000;

	if attackType=='public':
		startKey=0; endKey=256;
	elif attackType=='template':
		startKey=0; endKey=1;
	
	keySize = dict(SIZE_128=16, SIZE_192=24, SIZE_256=32)
	size=16; nbrRounds=10;
	expandedKeySize = 16*(nbrRounds+1)
	if attackType=='public':
		klen = 256;
	elif attackType=='template':
		klen = 1;
	
	for t_i in range (0,int(NoofTraces/trace_interval)):
		powerModelHD=np.zeros((16, trace_interval, klen), dtype=np.int8)
		tsta=time.time(); 
		for t_j in range(0, trace_interval):
			trace=t_i*trace_interval+t_j;
			for k in range(startKey, endKey, 1):
				if attackType=='public':
					key='';
					kfb="0x%0.2X"%k;
					kfb=kfb[2:]
					key=[kfb]*16
					if debug:	print(key)
				elif attackType=='template':
					key=keymsgct_dict[trace][0]

				ct = keymsgct_dict[trace][2];
				key_int = [[]for i in range(int(len(key)))];
				ct_int = [[]for i in range(int(len(ct)/2))];
				
				for i in range(int(len(ct)/2)):
					ct_int[i] = int(ct[2*i:2*(i+1)],16);
				
				for i in range(int(len(key))):
					key_int[i] = int(key[i],16);

				if attackType=='template':
					expandedKey=my_aes.expandKey(key_int, size, expandedKeySize)
					key_int=expandedKey[-16:];

				for tgtKey in range(16):
					rowKey = tgtKey%4;
					columnKey = int(tgtKey/4);
					tgtKey_mapped=rowKey*4+columnKey;	
					ct0 = ct_int[tgtKey];
					ct1 = ct_int[shift_row[tgtKey_mapped]];
					intermediate_state=inv_sbox[key_int[tgtKey]^ct0]
					#hw=hamming(0, intermediate_state)
					hd=hamming(ct1, intermediate_state)
					if verbose:	print('HD: {}, HW: {}'.format(hd, hw))
					#powerModel[trace, k]=hw;
					powerModelHD[tgtKey, t_j, k]=hd;
		for tgtKey in range(16):
				if attackType=='public':
					PMHDF=pmdir+'powerModel_AESP_lastRd_HD_Public_keyIndex_'+str(keyIndex)+'_B'+str(tgtKey+1)+'_'+str(int(publicDB_NoofTraces/1000))+'k.h5'
				elif attackType=='template':
					PMHDF=pmdir+'powerModel_AESP_lastRd_HD_Template_Train_B'+str(tgtKey+1)+'_'+str(int(publicDB_NoofTraces/1000))+'k.h5'
				col_idx=['k'+str(x) for x in range(startKey, endKey)]
				df=pd.DataFrame(powerModelHD[tgtKey, :, :], columns=col_idx, dtype=np.int8);
				#df=pd.DataFrame(powerModel, dtype='int8');
				if t_i==0:
					df.to_hdf(PMHDF, key=KEY_NAME, format='table', append=False, mode='w', index=None)
				else:
					df.to_hdf(PMHDF, key=KEY_NAME, format='table', append=True, index=None)
		tsto=time.time(); print('t_i: {}, tgtKey: {}, Key guess: {}, Elased Time (s): {}'.format(t_i, tgtKey, k, tsto-tsta)); tsta=time.time()



def inc_cpa(P_t, P_fft, startKey, endKey, start_trace, NoofTraces, m_traces_in, mtd_npts, keyIndex, correctKeyDec, Fp1, Fp2, plot_en, idir, ofile_ext, is_filter, is_dpa, verbose):
	temp_idir=idir+'temp/'
	chunk_key=klen;	
	#trace_interval=int(P_t[:,0].size/mtd_npts)
	trace_interval=int(NoofTraces/mtd_npts)
	#if verbose:	print('m_traces_in: {}, trace_interval: {}'.format(m_traces_in, trace_interval))

	for trace_idx in range(mtd_npts):
		m_traces=m_traces_in*mtd_npts+trace_idx;
		n1=int((m_traces)*trace_interval);n2=int((m_traces+1)*trace_interval);
		#if verbose:	print(n1, n2, trace_interval);
		for tgtKey in np.arange(startKey, endKey, 1):
			results_idir=idir+'results/Byte_'+str(tgtKey)+'/'
			if is_dpa:
				#PMHDF=pmdir+'powerModel_AESS_Round1_HW_B'+str(tgtKey+1)+'_4_10000k.h5'
				PMHDF=pmdir+'powerModel_AESP_lastRd_DPA_B'+str(tgtKey+1)+'_4_10000k.h5'
			else:
				PMHDF=pmdir+'powerModel_AESP_lastRd_HD_Public_keyIndex_'+str(keyIndex)+'_B'+str(tgtKey+1)+'_'+str(int(publicDB_NoofTraces/1000))+'k.h5'
			#if verbose:	print(PMHDF)
			n_k=int(klen/chunk_key)
			for k in range(n_k):
				start_key=k*chunk_key; end_key=start_key+chunk_key;
				key_columns=['k'+str(x) for x in range(start_key, end_key)]
				O=pd.read_hdf(PMHDF, start=n1, stop=n2, columns=key_columns, mode='r', dtype=np.int8).values;
				
				# Time Domain
				is_freq=0;
				if is_freq:
					P=P_fft;
				else:
					P=P_t;
				trace=P[0, :]
				mtd=start_trace; # default value
				trace_size=trace.size; w_size=trace_size; n_w=1;
				if is_freq:
					w_size=200; n_w=int(trace_size/(w_size));
				corr_index=np.zeros((klen, trace_size), dtype=np.float16)
				temp_corr_index=np.zeros((klen, w_size), dtype=np.float16)
				#print(trace_size)
				if is_freq:	freq=np.array([np.fft.fftfreq(w_size, d=1/fs)]*int(n_w)); freq=np.ravel(freq);
				#if is_freq:	print('w_size: {}, n_w: {}'.format(w_size, n_w))
				#if verbose:	print('m_traces: {}'.format(m_traces))
				#print(results_idir, temp_idir, idir)
				start_t=0; end_t=start_t+trace_size;
				s1_file=temp_idir+'s1_'+design_ext+'_Byte_'+str(tgtKey)+'_'+str(int(Fp1/1000000))+'M_'+str(int(Fp2/1000000))+'M_'+str(int(NoofTraces))+'_tw_'+str(start_t)+'-'+str(end_t)+'_freq_'+str(is_freq)+ofile_ext+'filter_'+str(is_filter)+'_dpa_'+str(is_dpa)+'.h5';
				s2_file=temp_idir+'s2_'+design_ext+'_Byte_'+str(tgtKey)+'_'+str(int(Fp1/1000000))+'M_'+str(int(Fp2/1000000))+'M_'+str(int(NoofTraces))+'_tw_'+str(start_t)+'-'+str(end_t)+'_freq_'+str(is_freq)+ofile_ext+'filter_'+str(is_filter)+'_dpa_'+str(is_dpa)+'.h5';
				s3_file=temp_idir+'s3_'+design_ext+'_Byte_'+str(tgtKey)+'_'+str(int(Fp1/1000000))+'M_'+str(int(Fp2/1000000))+'M_'+str(int(NoofTraces))+'_tw_'+str(start_t)+'-'+str(end_t)+'_freq_'+str(is_freq)+ofile_ext+'filter_'+str(is_filter)+'_dpa_'+str(is_dpa)+'.h5';
				s4_file=temp_idir+'s4_'+design_ext+'_Byte_'+str(tgtKey)+'_'+str(int(Fp1/1000000))+'M_'+str(int(Fp2/1000000))+'M_'+str(int(NoofTraces))+'_tw_'+str(start_t)+'-'+str(end_t)+'_freq_'+str(is_freq)+ofile_ext+'filter_'+str(is_filter)+'_dpa_'+str(is_dpa)+'.h5';
				s5_file=temp_idir+'s5_'+design_ext+'_Byte_'+str(tgtKey)+'_'+str(int(Fp1/1000000))+'M_'+str(int(Fp2/1000000))+'M_'+str(int(NoofTraces))+'_tw_'+str(start_t)+'-'+str(end_t)+'_freq_'+str(is_freq)+ofile_ext+'filter_'+str(is_filter)+'_dpa_'+str(is_dpa)+'.h5';
				if m_traces==0:
					s1=0; s2=0; s3=0; s4=0; s5=0; num_ones=0;
				else:
					s1=pd.read_hdf(s1_file, dtype=np.float32).values; s1=np.ravel(s1);
					s2=pd.read_hdf(s2_file, dtype=np.float32).values; s2=np.ravel(s2);
					s3=pd.read_hdf(s3_file, dtype=np.float32).values; s3=np.ravel(s3);
					s4=pd.read_hdf(s4_file, dtype=np.float32).values; s4=np.ravel(s4);
					s5=pd.read_hdf(s5_file, dtype=np.float32).values;
				start_time=time.time()
				#if verbose:	end_time=time.time(); print('Elapsed time for loading powerModel is: {0:.3f}'.format(end_time-start_time)); start_time=time.time()
				x_time=np.arange(start_t, end_t, 1)
				x_time=x_time.reshape(1, len(x_time))
				P=P.astype(np.float32)
				O=O.astype(np.float32)
				if is_dpa==0:
					#(s1, s2, s3, s4, s5, temp_corr_index)=inc_corr(P[n1%1000:n2%1000, :], O, s1, s2, s3, s4, s5, n2);
					(s1, s2, s3, s4, s5, temp_corr_index)=inc_corr(P[n1:n1+trace_interval, :], O, s1, s2, s3, s4, s5, n2);
				else:
					(s1, s3, temp_corr_index)=inc_corr(P, O, s1, s3, n2);
				df=pd.DataFrame(s1, dtype=np.float32);df.to_hdf(s1_file, key='s1')
				df=pd.DataFrame(s2, dtype=np.float32);df.to_hdf(s2_file, key='s2')
				df=pd.DataFrame(s3, dtype=np.float32);df.to_hdf(s3_file, key='s3')
				df=pd.DataFrame(s4, dtype=np.float32);df.to_hdf(s4_file, key='s4')
				df=pd.DataFrame(s5, dtype=np.float32);df.to_hdf(s5_file, key='s5')
				corr_index[start_key:end_key, start_t:end_t]=temp_corr_index;
				#if verbose:	end_time=time.time(); print('Elapsed time for corr computation is: {0:.3f}'.format(end_time-start_time)); start_time=time.time()
				o_file=results_idir+'corr_index_'+design_ext+'_Byte_'+str(tgtKey)+'_'+str(int(Fp1/1000000))+'M_'+str(int(Fp2/1000000))+'M_'+str(int(n2))+'_tw_'+str(start_t)+'-'+str(end_t)+'_freq_'+str(is_freq)+ofile_ext+'filter_'+str(is_filter)+'_dpa_'+str(is_dpa)+'.npz';
				#if is_freq:	print('freq.shape, corr_index.shape: {}, {}'.format(freq.shape, corr_index.shape))
				#if is_freq:	np.savetxt(o_file, np.append(freq.reshape(1,len(freq)), corr_index, axis=0), fmt='%1.2e');
				if is_freq:	np.savez_compressed(o_file, np.append(freq.reshape(1,len(freq)), corr_index, axis=0));
				#else:	np.savetxt(o_file, np.append(x_time, corr_index, axis=0), fmt='%1.2e');
				else:	np.savez_compressed(o_file, np.append(x_time, corr_index, axis=0));
				end_time=time.time()
				end_time=time.time()
				(rank_correct_key, corr_ratio, mcorr, temp_ranks)=analyze_cpa_window(corr_index, correctKeyDec, int(tgtKey/1), Fp1, Fp2, n_w, w_size, 0, 0)
				if verbose:	print('Byte: {}, start: {}, end: {}, Fp1: {}M, Fp2: {}M, traces: {}, DPA: {}, Freq: {}, ranks : {}, ratio: {}, elapsed time: {}'.format(tgtKey, start_t, end_t, Fp1/1000000, Fp2/1000000, n2, is_dpa, is_freq, rank_correct_key, corr_ratio, round(end_time-start_time,2)))
	
				# Freq Domain
				is_freq=1;
				if is_freq:
					P=P_fft;
				else:
					P=P_t;
				trace=P[0, :]
				mtd=start_trace; # default value
				trace_size=trace.size; w_size=trace_size; n_w=1;
				if is_freq:
					w_size=40; n_w=int(trace_size/(w_size));
				corr_index=np.zeros((klen, trace_size), dtype=np.float16)
				temp_corr_index=np.zeros((klen, w_size), dtype=np.float16)
				#print(trace_size)
				if is_freq:	freq=np.array([np.fft.fftfreq(w_size, d=1/fs)]*int(n_w)); freq=np.ravel(freq);
				#if is_freq:	print('w_size: {}, n_w: {}'.format(w_size, n_w))
				#if verbose:	print('m_traces: {}'.format(m_traces))
				start_t=0; end_t=start_t+trace_size;
				s1_file=temp_idir+'s1_'+design_ext+'_Byte_'+str(tgtKey)+'_'+str(int(Fp1/1000000))+'M_'+str(int(Fp2/1000000))+'M_'+str(int(NoofTraces))+'_tw_'+str(start_t)+'-'+str(end_t)+'_freq_'+str(is_freq)+ofile_ext+'filter_'+str(is_filter)+'_dpa_'+str(is_dpa)+'.h5';
				s2_file=temp_idir+'s2_'+design_ext+'_Byte_'+str(tgtKey)+'_'+str(int(Fp1/1000000))+'M_'+str(int(Fp2/1000000))+'M_'+str(int(NoofTraces))+'_tw_'+str(start_t)+'-'+str(end_t)+'_freq_'+str(is_freq)+ofile_ext+'filter_'+str(is_filter)+'_dpa_'+str(is_dpa)+'.h5';
				s3_file=temp_idir+'s3_'+design_ext+'_Byte_'+str(tgtKey)+'_'+str(int(Fp1/1000000))+'M_'+str(int(Fp2/1000000))+'M_'+str(int(NoofTraces))+'_tw_'+str(start_t)+'-'+str(end_t)+'_freq_'+str(is_freq)+ofile_ext+'filter_'+str(is_filter)+'_dpa_'+str(is_dpa)+'.h5';
				s4_file=temp_idir+'s4_'+design_ext+'_Byte_'+str(tgtKey)+'_'+str(int(Fp1/1000000))+'M_'+str(int(Fp2/1000000))+'M_'+str(int(NoofTraces))+'_tw_'+str(start_t)+'-'+str(end_t)+'_freq_'+str(is_freq)+ofile_ext+'filter_'+str(is_filter)+'_dpa_'+str(is_dpa)+'.h5';
				s5_file=temp_idir+'s5_'+design_ext+'_Byte_'+str(tgtKey)+'_'+str(int(Fp1/1000000))+'M_'+str(int(Fp2/1000000))+'M_'+str(int(NoofTraces))+'_tw_'+str(start_t)+'-'+str(end_t)+'_freq_'+str(is_freq)+ofile_ext+'filter_'+str(is_filter)+'_dpa_'+str(is_dpa)+'.h5';
				
				if m_traces==0:
					s1=0; s2=0; s3=0; s4=0; s5=0; num_ones=0;
				else:
					s1=pd.read_hdf(s1_file, dtype=np.float32).values; s1=np.ravel(s1);
					s2=pd.read_hdf(s2_file, dtype=np.float32).values; s2=np.ravel(s2);
					s3=pd.read_hdf(s3_file, dtype=np.float32).values; s3=np.ravel(s3);
					s4=pd.read_hdf(s4_file, dtype=np.float32).values; s4=np.ravel(s4);
					s5=pd.read_hdf(s5_file, dtype=np.float32).values;
				start_time=time.time()
	
				#if verbose:	end_time=time.time(); print('Elapsed time for loading powerModel is: {0:.3f}'.format(end_time-start_time)); start_time=time.time()
				x_time=np.arange(start_t, end_t, 1)
				x_time=x_time.reshape(1, len(x_time))
				P=P.astype(np.float32)
				O=O.astype(np.float32)
				if is_dpa==0:
					#(s1, s2, s3, s4, s5, temp_corr_index)=inc_corr(P, O, s1, s2, s3, s4, s5, n2);
					#(s1, s2, s3, s4, s5, temp_corr_index)=inc_corr(P[n1%1000:n1%1000+trace_interval, :], O, s1, s2, s3, s4, s5, n2);
					(s1, s2, s3, s4, s5, temp_corr_index)=inc_corr(P[n1:n1+trace_interval, :], O, s1, s2, s3, s4, s5, n2);
				else:
					(s1, s3, temp_corr_index)=inc_corr(P, O, s1, s3, n2);
				df=pd.DataFrame(s1, dtype=np.float32);df.to_hdf(s1_file, key='s1')
				df=pd.DataFrame(s2, dtype=np.float32);df.to_hdf(s2_file, key='s2')
				df=pd.DataFrame(s3, dtype=np.float32);df.to_hdf(s3_file, key='s3')
				df=pd.DataFrame(s4, dtype=np.float32);df.to_hdf(s4_file, key='s4')
				df=pd.DataFrame(s5, dtype=np.float32);df.to_hdf(s5_file, key='s5')
				corr_index[start_key:end_key, start_t:end_t]=temp_corr_index;
				#if verbose:	end_time=time.time(); print('Elapsed time for corr computation is: {0:.3f}'.format(end_time-start_time)); start_time=time.time()
				o_file=results_idir+'corr_index_'+design_ext+'_Byte_'+str(tgtKey)+'_'+str(int(Fp1/1000000))+'M_'+str(int(Fp2/1000000))+'M_'+str(int(n2))+'_tw_'+str(start_t)+'-'+str(end_t)+'_freq_'+str(is_freq)+ofile_ext+'filter_'+str(is_filter)+'_dpa_'+str(is_dpa)+'.npz';
				#if is_freq:	print('freq.shape, corr_index.shape: {}, {}'.format(freq.shape, corr_index.shape))
				#if is_freq:	np.savetxt(o_file, np.append(freq.reshape(1,len(freq)), corr_index, axis=0), fmt='%1.2e');
				if is_freq:	np.savez_compressed(o_file, np.append(freq.reshape(1,len(freq)), corr_index, axis=0));
				#else:	np.savetxt(o_file, np.append(x_time, corr_index, axis=0), fmt='%1.2e');
				else:	np.savez_compressed(o_file, np.append(x_time, corr_index, axis=0));
				end_time=time.time()
				end_time=time.time()
				(rank_correct_key, corr_ratio, mcorr, temp_ranks)=analyze_cpa_window(corr_index, correctKeyDec, int(tgtKey/1), Fp1, Fp2, n_w, w_size, 0, 0)
				if verbose:	print('Byte: {}, start: {}, end: {}, Fp1: {}M, Fp2: {}M, traces: {}, DPA: {}, Freq: {}, ranks : {}, ratio: {}, elapsed time: {}'.format(tgtKey, start_t, end_t, Fp1/1000000, Fp2/1000000, n2, is_dpa, is_freq, rank_correct_key, corr_ratio, round(end_time-start_time,2)))
			#ranks[m_traces]=rank_correct_key;
			#mtd_array[m_traces, 1:n_w+1]=temp_ranks.transpose();
			#ratio[m_traces]=corr_ratio;
			#max_corr[m_traces,:]=mcorr;
			#if verbose:	print('Byte: {}, start: {}, end: {}, Fp1: {}M, Fp2: {}M, traces: {}K, DPA: {}, Freq: {}, ranks : {}, ratio: {}, elapsed time: {}'.format(tgtKey, start_t, end_t, Fp1/1000000, Fp2/1000000, traces/1000, is_dpa, is_freq, rank_correct_key, corr_ratio, round(end_time-start_time,2)))
			#if ranks[m_traces]==1:
			#	mtd_pt=1;
			#else:
			#	if m_traces < num_pts-1:
			#		mtd_pt=0;
			#		mtd=traces_array[m_traces+1]
			#	else:
			#		mtd=traces_array[m_traces]	
			#if num_pts > 1:	
			#	if verbose:	print('Byte: {}, start: {}, end: {}, Fp1: {}M, Fp2: {}M, traces: {}K, DPA: {}, Freq: {}, ranks : {}, ratio: {}, MTD: {}K, elapsed time: {}'.format(tgtKey, start_t, end_t, Fp1/1000000, Fp2/1000000, traces/1000, is_dpa, is_freq, rank_correct_key, corr_ratio, mtd/1000, round(end_time-start_time,2)))
			#mtd_file=idir+'mtd_array_for_windows_'+design_ext+'_Byte_'+str(tgtKey)+'_'+str(int(Fp1/1000000))+'M_'+str(int(Fp2/1000000))+'M_'+str(int(traces))+'_tw_'+str(start_t)+'-'+str(end_t)+'_freq_'+str(is_freq)+ofile_ext+'ref_'+str(ref_trace_idx)+'_corr_offset_'+str(corr_offset)+'_filter_'+str(is_filter)+'_zpf_'+str(is_zpf)+'_align_'+str(is_align)+'_trig_'+str(is_trig)+'_rise_'+str(is_rise)+'_dpa_'+str(is_dpa)+'.npz';
			#np.savez_compressed(mtd_file, mtd_array);

	return (rank_correct_key, corr_ratio, mcorr)

def inc_corr(P, O, s1, s2, s3, s4, s5, n2):
	s1 = s1+np.einsum("nt->t", P, dtype='float32', optimize=True);
	s2 = s2+np.einsum("nt,nt->t", P, P, dtype='float32', optimize=True);
	s3 = s3+np.einsum("nk->k", O, dtype='float32', optimize=True);
	s4 = s4+np.einsum("nk,nk->k", O, O, dtype='float32', optimize=True);
	s5 = s5+np.einsum("nt,nk->tk", P, O, dtype='float32', optimize=True); # this takes the most amount of time (88s for 100K), others are negligible for 100K (ms) float16 doesn't work and float32 works and gives 32% speedup
	#print('s1.shape: {}, s2.shape: {}, s3.shape: {}, s4.shape: {}, s5.shape: {}'.format(s1.shape, s2.shape, s3.shape, s4.shape, s5.shape))
	tmp1 = np.multiply(s5, np.double(n2), dtype='float32')
	tmp2 = np.einsum("n,k->nk", s1,s3, dtype='float32', optimize=True);
	numerator = np.subtract(tmp1, tmp2, dtype='float32');
	tmp1 = np.einsum("t,t->t",s1,s1, dtype='float32', optimize=True);
	tmp2 = np.einsum("k,k->k",s3,s3, dtype='float32', optimize=True);
	tmp1 = np.subtract(np.multiply(s2, np.double(n2), dtype='float32'), tmp1, dtype='float32')
	tmp2 = np.subtract(np.multiply(s4, np.double(n2), dtype='float32'), tmp2, dtype='float32')
	tmp = np.einsum("t,k->tk", tmp1, tmp2, dtype='float32', optimize=True);
	denominator = np.sqrt(tmp);
	temp_corr_index = numerator / denominator; #del numerator, denominator;
	temp_corr_index=temp_corr_index.transpose(); temp_corr_index = np.nan_to_num(temp_corr_index);

	return (s1, s2, s3, s4, s5, temp_corr_index);

def inc_dom(P, O, s1, s3, num_ones, n2):
	#print('s1.shape: {}, s3.shape: {}'.format(s1.shape, s3.shape))
	s1 = s1+np.einsum("nt,nk->tk", P, O, dtype='float32', optimize=True); 
	s3 = s3+np.einsum("nt,nk->tk", P, 1-O, dtype='float32', optimize=True); 
	num_ones=num_ones+np.sum(O, axis=0)
	#print('s1.shape: {}, s3.shape: {}, num_ones.shape: {}, n2: {}'.format(s1.shape, s3.shape, num_ones.shape, n2))
	temp_corr_index = (s1/num_ones-s3/(n2-num_ones)); # difference of means
	temp_corr_index=temp_corr_index.transpose(); 
	temp_corr_index = np.nan_to_num(temp_corr_index);

	return (s1, s3, num_ones, temp_corr_index);

def analyze_cpa_window(corr_index, correctKeyDec, tgtKey, Fp1, Fp2, n_w, w_size, plot_en, verbose):
	temp_ratio=np.zeros((n_w,1), dtype=float);
	temp_ranks=np.ones((n_w,1), dtype=int)*klen;
	temp_max_corr=np.zeros((n_w, klen), dtype=float);
	start_t=0; end_t=corr_index.shape[1];
	for w in range(n_w):
		start_t=w*w_size; end_t=start_t+w_size;
		temp_corr_index=corr_index[:,start_t:end_t]
		(rank_correct_key, corr_ratio, mcorr)=analyze_cpa(temp_corr_index, correctKeyDec, tgtKey, Fp1, Fp2, 0, 0)
		temp_ranks[w, 0]=rank_correct_key;
		temp_ratio[w, 0]=corr_ratio;
		temp_max_corr[w, :]=mcorr;
		if plot_en:	plt.plot(corr_index.transpose()[:,:], 'gray'); plt.plot(corr_index[correctKeyDec[tgtKey],:], 'k'); plt.show()
		if verbose:	print('Byte: {}, window: {}, Fp1: {}M, Fp2: {}M, ranks : {}, ratio: {}'.format(tgtKey, w, Fp1/1000000, Fp2/1000000, rank_correct_key, round(corr_ratio,3)))
	temp_ratio=np.nan_to_num(temp_ratio);temp_max_corr=np.nan_to_num(temp_max_corr)
	min_rank=np.min(temp_ranks);
	max_index=np.argmax(temp_ratio);
	start_t=max_index*w_size; end_t=start_t+w_size;
	#print('window for max corr ratio: {}, start: {}ns, end: {}ns'.format(max_index, start_t, end_t))
	#print(corr_ratio)
	corr_ratio=temp_ratio[max_index];
	max_corr=temp_max_corr[max_index, :]
	if verbose:	print('Byte: {}, Fp1: {}M, Fp2: {}M, ratio: {}, ranks : {}'.format(tgtKey, Fp1/1000000, Fp2/1000000, corr_ratio, min_rank))
	return (min_rank, corr_ratio, max_corr, temp_ranks);

def analyze_cpa(corr_index, correctKeyDec, tgtKey, Fp1, Fp2, plot_en, verbose):
	#correctKeyDec=[9]
	max_corr=np.max(abs(corr_index), axis=1); # ma0:guesses
	#print('max_corr.shape: {}'.format(max_corr.shape))
	#plt.plot(max_corr); plt.show()
	temp=max_corr.argsort()
	peak_1=max_corr[temp[klen-1]];
	peak_2=max_corr[temp[klen-2]];
	#if verbose:	print('correctKeyDec: '.format(correctKeyDec[tgtKey]))
	peak_correct=max_corr[correctKeyDec[tgtKey]];
	ranks=np.empty(len(max_corr),int)
	ranks[temp]=np.arange(len(max_corr))
	rank_correct_key=klen-1-ranks[correctKeyDec[tgtKey]]+1
	if rank_correct_key==1:	
		corr_ratio=peak_correct/peak_2;
	else:
		corr_ratio=peak_correct/peak_1;
	if plot_en==1:
		for i in range(klen):	plt.plot(corr_index[i,:], 'b'); plt.plot(corr_index[correctKeyDec[tgtKey], :], 'r'); 
		plt.show();
	if verbose:
		if rank_correct_key==1:
			print ('ATTACK HAPPENS!! tgtKey: '+str(tgtKey)+', tgtKeyDec: '+str(correctKeyDec[tgtKey])+', TOP 5: '+str(max_corr.argsort()[-5:])+', RankTgtKey: '+str(rank_correct_key)+', Ratio: '+str(corr_ratio))
		else:
			print ('NO ATTACK!! tgtKey: '+str(tgtKey)+', tgtKeyDec: '+str(correctKeyDec[tgtKey])+', TOP 5: '+str(max_corr.argsort()[-5:])+', RankTgtKey: '+str(rank_correct_key)+', Ratio: '+str(corr_ratio))
	return (rank_correct_key, corr_ratio, max_corr);

