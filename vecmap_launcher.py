import sys
import os
from argparse import ArgumentParser

def get_all_filepaths(root,forbidden_suffix='_mapped.vec'):
	for path, subdirs, files in os.walk(root):
		for name in files:
			if not name.endswith(forbidden_suffix):
				yield os.path.join(path, name)

def get_config(path):

	if 'word2vec_s' in path:
		prefix = 'word2vec_s'
	elif 'fasttext_s' in path:
		prefix = 'fasttext_s'

	size = int(path.split(prefix)[1][:3])
	mincount = int(path.split('_mc')[1][:1])
	window = int(path.split('_w')[1][:1])
	return prefix,size,mincount,window

if __name__ == '__main__':

	parser = ArgumentParser()
	parser.add_argument('-td','--traindict', help='Training dictionary path', required=True)
	parser.add_argument('-sv','--source-vectors-folder', help='Source embeddings folder', required=True)
	parser.add_argument('-tv','--target-vectors-folder', help='Target embeddings folder', required=True)

	args = parser.parse_args()


	srcvecs = sorted(list(get_all_filepaths(args.source_vectors_folder,forbidden_suffix='_mapped.vec')))
	trvecs = sorted(list(get_all_filepaths(args.target_vectors_folder,forbidden_suffix='_mapped.vec')))

	out = []
	for srcv in srcvecs:
		src_pr, src_s, src_mc, src_w = get_config(srcv)
		for trv in trvecs:
			tr_pr, tr_s, tr_mc, tr_w = get_config(trv)
			if src_s==tr_s and src_mc == tr_mc and src_w == tr_w and src_pr == tr_pr:
				print('SRC: ',srcv)
				print('TGT: ',trv)
				
				runstr = "python3 vecmap/map_embeddings.py --supervised "+\
				args.traindict+' "'+srcv+'" "'+trv+'" "'+\
				srcv+'_mapped.vec" "'+trv+'_mapped.vec"'

				print('=== CMD ===')
				print(runstr)
				os.system(runstr)
				out.append(runstr)

	outfile = 'cmds_log.txt'
	with open(outfile,'w') as outf:
		for x in out:
			outf.write(x+'\n')