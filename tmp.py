import os
import re

class allele:
    def __init__(self, chr, position, ref, alt) -> None:
        self.chr = chr
        self.position = position
        self.ref = ref
        self.alt = alt
        self.id = f'chr{chr}:{position}:{ref}:{alt}'

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def mkdir(dirname):
    if dirname.strip() != '':
        os.makedirs(dirname, exist_ok=True)


def preprocess(path):
    for i in range(1,2):
        header = f'##fileformat=VCFv4.2\n\
##FILTER=<ID=PASS,Description="All filters passed">\n\
##filedate=20210628\n\
##source="beagle.27Jan18.7e1.jar (version 4.1)"\n\
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n\
##contig=<ID=chr{i}>\n'
    with open('samples.txt', 'r') as fp:
        samples = '\t'.join(fp.read().split())
        header += '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t' + samples
    output_prefix = f'output/output.chr{i}.vcf'
    if os.path.exists(output_prefix):
        os.remove(output_prefix)
    mkdir(os.path.dirname(output_prefix))
    with open(output_prefix, 'w') as result_file:
        result_file.write(header)
        result_file.write("\n")
        with open(os.path.join(path,f'chr{i}.txt'), 'r') as fp:
            tmp = fp.readline()
            for line in fp:
                items = line.rstrip().split()
                method_1 = allele(*items[0:4])
                method_2 = allele(*items[4:8])
                method_3 = allele(*items[8:12])
                gt = '\t'.join(items[12:])
                # print(gt)
                for index, method in enumerate([method_1, method_2, method_3]):
                    if 'NA' not in method.id:
                        snp = f'chr{method.chr}\t{method.position}\t{method.id}\t{method.ref}\t{method.alt}\t.\t.\t.\tGT\t{gt}\n'
                        result_file.write(snp)
                        break
    os.system(f'gzip output/output.chr{i}.vcf')
    print(f"Done output/output.chr{i}.vcf")
if __name__ == '__main__':
    path = 'consensus'
    preprocess(path)
