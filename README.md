# fastSCA

Requirements:
	1.	Python 3.6.4 from Anaconda3 distribution 
	2.	pandas 0.22.0
	3.	numpy, scipy, matplotlib, importlib, argparse, etc
	
The python code provided in this package includes following functions --

	1. incremental correlation power analysis - cpa (inc_cpa)
	2. incremental differential power analysis (inc_cpa) - same function as in 1, dpa analysis is controlled with a 'is_dpa' variable. TODO - add power model generation for dpa.
	3. template attack implemented in the main script - controlled with run_template_attack variable - implements multi-variate gaussian distribution based template attacks

All these attacks can be conducted in time domain (usual time-domain CPA/DPA) and in freq. domain to break desynchronization/misalignment based countermeasures.  

NOTE 1: contains vectorized (matrix) operations to speed up the correlation computation. Uses numpy's einsum function optimized for vector/matrix multiply/other operations. Please read this article for more details - https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/

HOW TO RUN:

Steps:
	1. Download public/template database for power signatures based on SASEBO-GII from dpa contest v2 website (http://www.dpacontest.org/v2/download.php)
	2. Define paths for parent dir to the downloaded power traces in def_design.py (variables 'idir' and 'odir')
	3. All other variables should remain same in def_design.py if you are using dpa contest v2 dir structure
	4. collectTraces and generatePowerModel variables should be 1 for running the attack for the first time. Can be made 0 when trace arrays are created and power models are generated from the 1st run. 
NOTE 2: look into computational aspects of correlation power analysis attack paper for more details about incremental CPA (https://link.springer.com/article/10.1007/s13389-016-0122-9)

Sample command for CPA based on dpa contest v2 public database:

	python -i sca_analysis_mem.py --attackType public --NoofTraces 20000 --mtd_start_trace 2000 --mtd_npts 10 --single_band 1 --start_band 0 --end_band 5 --keyIndex 0 --collectTraces 0 --generatePowerModel 0 --run_cpa_attack 1 --is_filter 1 --is_dpa 0 --startKey 0 --endKey 16

Sample command for template attack (TA) based on dpa contest v2 template database:

	python -i sca_analysis_mem.py --attackType template --NoofTraces 200000 --mtd_start_trace 2000 --mtd_npts 10 --single_band 1 --start_band 0 --end_band 5 --keyIndex 0 --collectTraces 0 --generatePowerModel 0 --run_cpa_attack 0 --is_filter 1 --is_dpa 0 --startKey 0 --endKey 16

NOTE 3: for template attacks, its good idea to look at points of interest only as we are using all 1 million power traces for creating the templates. For the dpa contest v2 database, samples from 2500 to 3000 contain the last round (samples of interest) so can be chosen to reduce memory and compute complexity. Same thing can be applied for CPA.

Files Description:

	1.	def_design.py - define the paths to public/template database
	2.	global_settings.py - global settings, such as no of traces in public/template database, no of key guesses, etc
	3.	my_aes.py - functions related to AES encryption scheme - used to derive power model/verify functionality
	4.	sca2.py - contains all the function definitions 
	5.	sca_analysis_mem.py - main script which is run for the attack

I'd like to thank chipwhisper (https://wiki.newae.com/Template_Attacks) for their article on template attacks which helped us a lot to implement the attack for our design/dpa contest v2.  

We are working on implementing SVM, HMM, DNN based attacks. Code will be updated shortly once the implementation is completed and verified.

Please contact me at rathorearvind19@gmail.com for any questions. thanks
