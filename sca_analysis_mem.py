import numpy as np
import sca2
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
import scipy.stats
import scipy.io as spio
from scipy.io import loadmat
from scipy.signal import freqz
from scipy.signal import butter, lfilter
from scipy.signal import buttord
from scipy import signal
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
import h5py
import importlib
import my_aes
import argparse
#import pycorrelate as pyc

parser=argparse.ArgumentParser(prog='sca_analysis_mem',	formatter_class=argparse.MetavarTypeHelpFormatter)

#parser.add_argument('--idir', type=str)
#parser.add_argument('--odir', type=str)
parser.add_argument('--attackType', type=str)
parser.add_argument('--NoofTraces', type=int)
parser.add_argument('--mtd_start_trace', type=int)
parser.add_argument('--mtd_npts', type=int)
parser.add_argument('--single_band', type=int)
parser.add_argument('--start_band', type=int)
parser.add_argument('--end_band', type=int)
parser.add_argument('--keyIndex', type=int)
parser.add_argument('--collectTraces', type=int)
parser.add_argument('--generatePowerModel', type=int)
parser.add_argument('--run_cpa_attack', type=int)
parser.add_argument('--is_filter', type=int)
parser.add_argument('--is_dpa', type=int)
parser.add_argument('--startKey', type=int)
parser.add_argument('--endKey', type=int)

args = parser.parse_args()

run_template_attack=1-args.run_cpa_attack
run_svm_attack=0; 
run_dnn_attack=0;

exec(open('def_design.py').read())
exec(open('global_settings.py').read())

#python -i sca_analysis_mem.py --attackType public --NoofTraces 20000 --mtd_start_trace 2000 --mtd_npts 10 --single_band 1 --start_band 0 --end_band 5 --keyIndex 0 --collectTraces 1 --generatePowerModel 1 --run_cpa_attack 1 --is_filter 1 --is_dpa 0 --startKey 0 --endKey 16


debug=0; verbose=1; plot_en=0; plot_en=0; plot_en2=0;

# template attacks
#numPOIs=4; POIspacing=40;
numPOIs=2; POIspacing=200;
NoofTestTraces=5000;
keyIndexTest=1;
isHD_Dist=1;
collectTestTraces=1;
run_pca_analysis=0;
normalize_data=0;
is_freq=0;


#start_t=2500; end_t=2900;
start_t=2500; end_t=3000;
#start_t=0; end_t=NoofSamples;


if args.attackType=='public':
	powerTraceFile='powerTrace/powerTrace_Public_keyIndex_'+str(args.keyIndex)+'_'+str(int(publicDB_NoofTraces/1000))+'k.h5'
elif args.attackType=='template':
	powerTraceFile='powerTrace/powerTrace_Template_Train_'+str(int(templateDB_NoofTraces/1000))+'k.h5'

(keymsgct_dict, traceArray)=sca2.trace_collector(args.keyIndex, args.NoofTraces, args.attackType, args.collectTraces)
tsta=time.time()
IKey=keymsgct_dict[0][0]
#IKey='13198a2e03707344a4093822299f31d0' #keyIndex=1;
print(IKey)
if args.collectTraces:
	#np.savez_compressed(powerTraceFile, traceArray)
	KEY_NAME='train'
	col_idx=['t'+str(x) for x in range(0,len(traceArray[0]))]
	df=pd.DataFrame(traceArray, columns=col_idx, dtype='float16');
	df.to_hdf(powerTraceFile, key=KEY_NAME, format='table', index=None, append=False, mode='w')
else:
	#traceArray=np.load(powerTraceFile); traceArray=traceArray['arr_0']; traceArray=traceArray[0:args.NoofTraces, :]
	col_idx=['t'+str(x) for x in range(start_t, end_t)]
	#col_idx=['t'+str(x) for x in range(0, NoofSamples)]
	traceArray=pd.read_hdf(powerTraceFile, start=0, stop=args.NoofTraces, columns=col_idx, dtype=np.float16).values;

tsto=time.time(); print('Elapsed Time for storing/loading powerTrace is {0:.05f}'.format(tsto-tsta)); tsta=time.time();

size=16; nbrRounds=10;
expandedKeySize = 16*(nbrRounds+1)

IKey_int = [[]for i in range(int(len(IKey)/2))];

for i in range(int(len(IKey)/2)):
	IKey_int[i] = int(IKey[2*i:2*(i+1)],16);

expandedKey=my_aes.expandKey(IKey_int, size, expandedKeySize)
correctKeyDec=expandedKey[-16:];

if args.generatePowerModel:
	print('dbg - generating power model...');
	sca2.last_round_power_model(keymsgct_dict, args.keyIndex, args.NoofTraces, args.attackType, 0, 0)

if not os.path.exists(odir+'/results'):
	os.makedirs(odir+'/results/')

for tgtKey in range(16):
	results_idir=odir+'results/Byte_'+str(tgtKey)+'/'
	if not os.path.exists(results_idir):
		os.makedirs(results_idir);

temp_idir=odir+'temp/'
if not os.path.exists(temp_idir):
	os.makedirs(temp_idir);

#args.single_band=0; args.end_band=5; args.start_band=0;
bands=np.zeros((args.end_band-args.start_band, 1))
#args.startKey=0; args.endKey=16;
#args.startKey=6; args.endKey=7;

# mov mean settings
do_mov_mean=0; N_movmean=100;

t_array_time=np.zeros((args.end_band-args.start_band, 1))
t_array_freq=np.zeros((args.end_band-args.start_band, 1))

if not args.is_filter:	args.single_band=1;

if args.single_band:	num_bands=1; args.start_band=0; args.end_band=1;
else:	num_bands=args.end_band-args.start_band;

# window settings
w_offset=100; w_size=200; n_w=(int)(((traceArray[0].size)/w_size)*(w_size/w_offset));
if verbose:	print('window size: {}, window offset: {}, # of windows: {}'.format(w_size, w_offset, n_w))

ofile_ext='_';

# template


tsta=time.time()	
for filt_idx in range(args.start_band, args.end_band):
	if args.single_band:	Fp1=5e06; Fp2=45e06;
	else:	Fp1=band_offset+(filt_idx)*band; Fp2=Fp1+band;
	if args.is_filter:
		print('Fp1: {}, Fp2: {}'.format(Fp1/1e6, Fp2/1e6))
		b, a=sca2.butter_bandpass(Fp1, Fp2, fs, order=3)
		traceArray=lfilter(b, a, traceArray,axis=1) # axis-1 horizontal
	else:
		traceArray=traceArray;
	#traceArray=traceArray[:, start_t:end_t]
	traceArray_fft=np.zeros([args.NoofTraces, n_w*w_offset])
	if do_mov_mean:	
		for t_i in range(len(traceArray)):
			traceArray[t_i, :]=np.convolve(traceArray[t_i,:], np.ones((N_movmean,))/N_movmean, mode='same');
	#if is_freq:	traceArray=np.fft.fft(traceArray, n=None, norm=None, axis=1); traceArray=traceArray[:, 0:int(traceArray[0].size/2)]
	if is_freq:
		for w in range(n_w):
			temp_start_t=w*w_offset; temp_end_t=temp_start_t+w_size;
			traceArray_fft[:, w*w_offset:(w+1)*w_offset]=np.abs(np.fft.fft(traceArray[:,temp_start_t:temp_end_t], n=None, norm=None, axis=1)[:,0:w_offset]); # axis 1 is row, axis 0 is column
		if run_template_attack:
			traceArray=traceArray_fft; del traceArray_fft;
	tsto=time.time(); print('Elapsed Time for Filtering and FFT is {0:.05f}'.format(tsto-tsta)); tsta=time.time();

	if args.attackType=='template':
		(testKeymsgct_dict, testTraceArray)=sca2.trace_collector(keyIndexTest, NoofTestTraces, 'public', collectTestTraces)
		testTraceArray=testTraceArray[:, start_t:end_t]

		IKey=testKeymsgct_dict[0][0]
		IKey_int = [[]for i in range(int(len(IKey)/2))];

		for k_i in range(int(len(IKey)/2)):
			IKey_int[k_i] = int(IKey[2*k_i:2*(k_i+1)],16);

		expandedKey=my_aes.expandKey(IKey_int, size, expandedKeySize)
		correctKeyDec=expandedKey[-16:];
		
		print(IKey, correctKeyDec);
		if args.is_filter:
			testTraceArray=lfilter(b, a, testTraceArray,axis=1) # axis-1 horizontal
		else:
			testTraceArray=testTraceArray;

		if do_mov_mean:	
			for t_i in range(len(testTraceArray)):
				testTraceArray[t_i, :]=np.convolve(testTraceArray[t_i,:], np.ones((N_movmean,))/N_movmean, mode='same');
		#if is_freq:	testTraceArray=np.fft.fft(testTraceArray, n=None, norm=None, axis=1); testTraceArray=testTraceArray[:, 0:int(testTraceArray[0].size/2)]
		testTraceArray_fft=np.zeros([NoofTestTraces, n_w*w_offset])
		if is_freq:
			for w in range(n_w):
				temp_start_t=w*w_offset; temp_end_t=temp_start_t+w_size;
				testTraceArray_fft[:, w*w_offset:(w+1)*w_offset]=abs(np.fft.fft(testTraceArray[:,temp_start_t:temp_end_t], n=None, norm=None, axis=1)[:,0:w_offset]); # axis 1 is row, axis 0 is column
			testTraceArray=testTraceArray_fft; del testTraceArray_fft;

	if args.run_cpa_attack==1:
		#(ranks, ratio, max_corr)=sca2.inc_cpa(traceArray, traceArray_fft, args.startKey, args.endKey, args.mtd_start_trace, args.NoofTraces, 0, args.mtd_npts, args.keyIndex, correctKeyDec, Fp1, Fp2, plot_en, odir, ofile_ext, args.is_filter, args.is_dpa, 1)
		(ranks, ratio, max_corr)=sca2.inc_cpa(traceArray, traceArray_fft, args.startKey, args.endKey, args.mtd_start_trace, args.NoofTraces, 0, args.mtd_npts, args.keyIndex, correctKeyDec, Fp1, Fp2, plot_en, odir, ofile_ext, args.is_filter, args.is_dpa, verbose)
		tsto=time.time(); 
		if verbose:	print("Elapsed Time for CPA is %s seconds " %(tsto-tsta)); tsta=time.time()
	else:
		mtd_array=np.zeros((args.endKey-args.startKey,1), dtype=np.int32)
		for tgtKey in range(args.startKey, args.endKey):
		#for tgtKey in range(5,7):
			#tgtKey=5;
			rowKey = tgtKey%4;columnKey = int(tgtKey/4);tgtKey_mapped=rowKey*4+columnKey;

			tsta=time.time();
			traceArrayHD = [[] for _ in range(9)];
			trainHD = [[]for i in range(0,args.NoofTraces)];
			PMHDF=pmdir+'powerModel_AESP_lastRd_HD_Template_Train_B'+str(tgtKey+1)+'_'+str(int(templateDB_NoofTraces/1000))+'k.h5'
			#trainHD=pd.read_hdf(PMHDF, start=0, stop=args.NoofTraces, dtype=np.int8).values;
			f = open('./temp_9and10round_Template_RandomKey_RandomPT_Train_1M.txt','r');	
			line = f.readlines();
			f.close();
			for i in range(args.NoofTraces):
				#rowKey = tgtKey%4;columnKey = int(tgtKey/4);tgtKey_mapped=rowKey*4+columnKey;
				line_temp = line[i].split(',');
				intermediate = line_temp[0].split(); # 9th round
				ct = line_temp[1].split(); # ciphertext
				intermediate_bin = '{0:08b}'.format(int(intermediate[tgtKey]));
				#ct_bin = '{0:08b}'.format(int(ct[sca2.shift_row[tgtKey_mapped]]));
				ct_bin = '{0:08b}'.format(int(ct[tgtKey]));
				if isHD_Dist:
					trainHD[i] = [sca2.hamming2(intermediate_bin, ct_bin)];
				else:
					trainHD[i] = [sca2.hamming2(intermediate_bin, '00000000')];

			for i in range(len(traceArray)):
				traceArrayHD[trainHD[i][0]].append(i)

			if run_template_attack or run_svm_attack:
				#traceArrayHD = [np.array(traceArray[traceArrayHD[HD]) for HD in range(9)];

				tempMeans = np.zeros((9, len(traceArray[0])));
				
				for i in range(9):
				    tempMeans[i] = np.average(traceArray[traceArrayHD[i]], 0);
				
				tempSumDiff = np.zeros(len(traceArray[0]));
				
				for i in range(9):
				    for j in range(i):
				        tempSumDiff += np.abs(tempMeans[i] - tempMeans[j])
				
				tsto = time.time();
				elapsed_time = tsto - tsta;
				#print (elapsed_time,'s');
				#print (len(tempSumDiff));
				if plot_en2:	plt.plot(tempSumDiff); plt.grid(); plt.show()
				
				POIs = []
				
				for i in range(numPOIs):
				    nextPOI = tempSumDiff.argmax()
				    POIs.append(nextPOI)
				    
				    poiMin = max(0, nextPOI - POIspacing)
				    poiMax = min(nextPOI + POIspacing, len(tempSumDiff))
				    for j in range(poiMin, poiMax):
				        tempSumDiff[j] = 0
				#POIs=np.array(POIs); POIs=POIs[POIs>50]; POIs=POIs[POIs<550];POIs=POIs.tolist();
				if verbose:	print ('POIs: {}'.format(POIs))
				if run_pca_analysis:	POIs = range(numPOIs);
				numPOIs=len(POIs);
				#POIs=[54, 79, 128, 104, 20, 86, 29, 113, 48, 98, 134, 73, 35, 61, 122, 92]	
				#POIs=[128, 104, 113, 134, 122]	
				meanMatrix = np.zeros((9, numPOIs))
				for HD in range(9):
				    for i in range(numPOIs):
				        meanMatrix[HD][i] = tempMeans[HD][POIs[i]]
				
				#np.cov(a, b) = [[cov(a, a), cov(a, b)],
				#                [cov(b, a), cov(b, b)]]
				
				covMatrix  = np.zeros((9, numPOIs, numPOIs))
				for HD in range(9):
				    for i in range(numPOIs):
				        for j in range(numPOIs):
				            x = traceArray[traceArrayHD[HD], POIs[i]]
				            y = traceArray[traceArrayHD[HD], POIs[j]]
				            #y = traceArrayHD[HD][:,POIs[j]]
				            covMatrix[HD,i,j] = sca2.cov(x, y)
				#print (meanMatrix)
				#print (covMatrix[0]);
				

			if run_svm_attack:
				traceArray=traceArray[:, POIs]
				# Train the model
				tr_data=np.array(traceArray).astype(np.float32)
				tr_labels=np.asarray(tempHD, dtype=np.int32);

				print("train_data")
				#train_data = DA_ShiftDeform_trace(tr_data, 500, 500, 5)
				train_data=tr_data; del tr_data;
				# normalization of the train data, subtract mean and divide by std deviation
			#	if normalize_data:
			#		train_data_mean=np.mean(train_data, axis=0);
			#		train_data_std=np.std(train_data, axis=0);
			#		train_data=train_data-train_data_mean;
			#		train_data=train_data/train_data_std;
				print(train_data.shape)
				
				print("train_labels")
				#train_labels = DA_ShiftDeform_labels(tr_labels, 5)
				train_labels=tr_labels; del tr_labels;
				print(train_labels)
				print(train_labels.shape)
				label_freq=np.zeros((9,1));
				class_wt={};
				for label in range(9):
					label_freq[label, 0]=np.sum(train_labels==label);
					class_wt[label]=1/label_freq[label,0];
			
				#class_wt=np.ravel(class_wt);
				sample_wt=np.zeros((len(train_labels),1), dtype=float);
				for s in range(len(train_labels)):
					sample_wt[s, 0]=1/label_freq[train_labels[s],0];
				#svm_model=sklearn.svm.SVC(C=1, kernel='rbf', degree=3, gamma=0.01, coef0=0.0, shrinking=True, probability=False,tol=0.001, cache_size=20000, class_weight=class_wt, verbose=False, max_iter=-1, random_state=None)
				C=1; gamma=0.125;
				svm_model=sklearn.svm.SVC(C=C, kernel='linear', degree=1, gamma='auto', coef0=0.0, shrinking=True, probability=False,tol=0.001, cache_size=20000, class_weight='balanced', verbose=False, max_iter=-1, random_state=None)
				tsta=time.time();
				svm_model.fit(train_data, np.ravel(train_labels))
				score=svm_model.score(train_data, np.ravel(train_labels))
				tsto=time.time(); print('Elapsed Time: {}'.format(tsto-tsta))
			if run_dnn_attack:
				# Create the Estimator
				#classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=train_idir+"convnet_model/output")
				classifier=tf.estimator.Estimator(model_fn=cnn_model_fn)
				print("Estimator created")
				
				
				# Set up logging for predictions
				# Log the values in the "Softmax" tensor with label "probabilities"
				tensors_to_log = {"probabilities": "softmax_tensor"}
				logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
				print("set up logging")
				
				# Train the model
				tr_data=np.array(traceArray).astype(np.float32)
				tr_labels=np.asarray(tempHD, dtype=np.int32);

				print("train_data")
				#train_data = DA_ShiftDeform_trace(tr_data, 500, 500, 5)
				train_data=tr_data; del tr_data;
				print(train_data.shape)
				
				print("train_labels")
				#train_labels = DA_ShiftDeform_labels(tr_labels, 5)
				train_labels=tr_labels; del tr_labels;
				print(train_labels)
				print(train_labels.shape)
				
				print("train starts")

				train_input_fn = tf.estimator.inputs.numpy_input_fn(
				    x={"x": train_data},
				    y=train_labels,
				    batch_size=1000,
				    num_epochs=None,
				    shuffle=True)
				classifier.train(
				    input_fn=train_input_fn,
				    steps=10000, # steps=min(steps_specified, dataset_size/batch_size * num_epochs)
				    hooks=[logging_hook])
				print("train ends")
			

			#testTraceArray_fft=np.zeros([NoofTestTraces, n_w*w_size], dtype=np.float16)
			#if do_mov_mean:	sig_f_allgn=np.convolve(sig_f_allgn, np.ones((N_movmean,))/N_movmean, mode='valid');
			#for w in range(n_w-1):
			#	start_t=w*w_offset; end_t=start_t+w_size;
			#	testTraceArray_fft[:, w*w_size:(w+1)*w_size]=abs(np.fft.fft(testTraceArray[:,start_t:end_t], n=None, norm=None, axis=1)); # axis 1 is row, axis 0 is column

			if normalize_data:
				testTraceArray=(testTraceArray - traceArray_mean)/traceArray_std;
				#testTraceArray=(testTraceArray - traceArray_mean);
			print('testTraceArray.shape: {}'.format(testTraceArray.shape))

			random_i=np.random.randint(NoofTestTraces);
			
			if plot_en2:	plt.plot(traceArray[random_i,:]); plt.plot(testTraceArray[random_i,:]); plt.show();
			startTrace=0;
			endTrace=NoofTestTraces;
			klen=256;

			tempCText = [[]for i in range(0,args.NoofTraces)];
			tempIntermediate = [[]for i in range(0,args.NoofTraces)];
			tempHD = [[]for i in range(0,args.NoofTraces)];
			tempEvHD = [[]for i in range(0,NoofTestTraces)];
			tempTracesHD = [[] for _ in range(9)];
			atkCText0 = [[]for i in range(0,NoofTestTraces)];
			atkCText1 = [[]for i in range(0,NoofTestTraces)];
			atkKey = [[]for i in range(0,NoofTestTraces)];
			rank_key = [[]for i in range(0,NoofTestTraces)];

			for i in range (startTrace,endTrace):
				key = testKeymsgct_dict[i][0];
				pt = testKeymsgct_dict[i][1];
				ct = testKeymsgct_dict[i][2];
				atkCText0[i-startTrace] = np.array(int(''.join(ct[tgtKey*2:tgtKey*2+2]),16));
				atkCText1[i-startTrace] = np.array(int(''.join(ct[sca2.shift_row[tgtKey_mapped]*2:sca2.shift_row[tgtKey_mapped]*2+2]),16));
				if i == startTrace:
					atkKey = np.array(int(''.join(key[2*tgtKey:2*tgtKey+2]),16));
				
				#print (len(testTraceArray));
				#print (len(atkCText0))
			print (atkKey)
			P_k = np.zeros(klen)
			tsta=time.time()
			#start_t=0; end_t=traceArray[0].size
			if run_template_attack:
			       #del traceArray;
			       results_file = odir+'results/results_lastRd_key_'+str(tgtKey)+'_Filter_'+str(int(Fp1/1000000))+'M_'+str(int(Fp2/1000000))+'M_Train_'+ str(args.NoofTraces) + '_Test_' + str(NoofTestTraces)+ '_POIs_'+str(numPOIs) + '_POIspacing_' + str(POIspacing) + '_start_'+str(start_t)+'_end_'+str(end_t)+'_freq_'+str(is_freq)+'_template.txt';f = open(results_file,'w');
			       for j in range(NoofTestTraces):
			       	a = [testTraceArray[j][POIs[i]] for i in range(len(POIs))]
			       	
			       	ct0=atkCText0[j]
			       	ct1=atkCText1[j]
			       	for k in range(klen):
			       		#tstart=time.time()
			       		intermediate = sca2.inv_sbox[ct0 ^ k]
			       		intermediate_bin = '{0:08b}'.format(int(intermediate));
			       		ct1_bin = '{0:08b}'.format(int(ct1));
			       		if isHD_Dist:
			       			HD= sca2.hamming2(intermediate_bin, ct1_bin)
			       		else:
			       			HD = sca2.hamming2(intermediate_bin, '00000000')
			       		#tstop=time.time(); print('elapsed time for hamming distance computation: {}'.format(tstop-tstart)); tstart=time.time()
			       		rv = multivariate_normal(meanMatrix[HD], covMatrix[HD])
			
			       		p_kj = rv.pdf(a)
			       	
			       		P_k[k] += np.log(p_kj)
			       		#tstop=time.time(); print('elapsed time for log likelihood computation: {}'.format(tstop-tstart)); tstart=time.time()
			       	temp=P_k.argsort()
			       	ranks=np.empty(len(P_k),int)
			       	ranks[temp]=np.arange(len(P_k))
			       	rank_correct_key=255-ranks[correctKeyDec[tgtKey]]+1
			       	rank_key[j]=rank_correct_key
			       	#tstop=time.time(); print('elapsed time for post processing: {}'.format(tstop-tstart)); tstart=time.time()
			       	
			       	if((j+1)%1000==0):
			       		if (np.any(P_k.argsort()[-10:] == correctKeyDec[tgtKey])):
			       			print ('start: {}, end: {}, Fp1: {}, Fp2: {}, tgtKey: {}, tgtKeyDec: {}, trace: {}, top10: {}, rank: {}, YES!'.format(start_t, end_t, Fp1/1e6, Fp2/1e6, tgtKey, correctKeyDec[tgtKey], j+1, P_k.argsort()[-10:], rank_correct_key))
			       		else:
			       			print ('start: {}, end: {}, Fp1: {}, Fp2: {}, tgtKey: {}, tgtKeyDec: {}, trace: {}, top10: {}, rank: {}, NO!'.format(start_t, end_t, Fp1/1e6, Fp2/1e6, tgtKey, correctKeyDec[tgtKey], j+1, P_k.argsort()[-10:], rank_correct_key))
			       			#print (tgtKey, correctKeyDec[tgtKey], j, P_k.argsort()[-10:], rank_correct_key, 'NO!')
			       	f.writelines(str(tgtKey) + ' ' + str(correctKeyDec[tgtKey]) + ' '+ str(j) + ' ' + str(P_k.argsort()[-10:]) + ' rank ' + str(rank_correct_key) + '\n');
			       mtd=0;
			       for r_i in range(len(rank_key)):
			       	if(rank_key[r_i]==1):
			       		mtd=mtd;
			       	else:
			       		mtd=r_i;
			       mtd=mtd+1;
			       print(str(tgtKey) + ' ' + str(correctKeyDec[tgtKey]) + ' MTD '+ str(mtd) + ' Max_Traces ' + str(NoofTestTraces) + '\n');
			       f.writelines(str(tgtKey) + ' ' + str(correctKeyDec[tgtKey]) + ' MTD '+ str(mtd) + ' Max_Traces ' + str(NoofTestTraces) + '\n');
			       tsto=time.time(); print('Elapsed Time: {}'.format(tsto-tsta))
			       if plot_en2:	plt.plot(rank_key);  plt.grid(); plt.show(); 
			mtd_array[tgtKey-args.startKey,0]=mtd;
