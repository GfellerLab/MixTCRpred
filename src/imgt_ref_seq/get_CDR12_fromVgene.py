from Bio import SeqIO
import natsort

## download allele data from https://www.imgt.org/vquest/refseqh.html
# for TRAV and TRBV (for CDR1, CDR2) download F+ORF+in-frame P with IMGT gaps
# for TRAJ and TRBJ (for CDR1, CDR2) you can download F+ORF+in-frame P (not aligned but not used now)


### here for CDR1,2 definition
#https://www.imgt.org/IMGTScientificChart/Nomenclature/IMGT-FRCDRdefinition.html

####################################################################################################
##### Homo Sapiens
####################################################################################################

#TRAV
gene_name = 'TRAV'
all_gene = []
all_cdr1 = []
all_cdr2 = []
for seq_record in SeqIO.parse("./{0}_HomoSapiens.fasta".format(gene_name), "fasta"):
    name = seq_record.id
    vgene = name.split("|")[1]
    seq = seq_record.seq
    break
    cdr1 = str(seq[26:38]).replace("-","")
    cdr2 = str(seq[55:65]).replace("-","")
    all_cdr1.append(cdr1)
    all_cdr2.append(cdr2)
    #print(vgene, cdr1, cdr2)
    #print("\"{0}\":\"{1}\"".format(vgene, cdr1), end=',')
    print("\"{0}\":\"{1}\"".format(vgene, cdr2), end=',')

####################################################################################################
#TRBV
gene_name = 'TRBV'
all_gene = []
all_cdr1 = []
all_cdr2 = []
for seq_record in SeqIO.parse("./{0}_HomoSapiens.fasta".format(gene_name), "fasta"):
    name = seq_record.id
    seq = seq_record.seq
    vgene = name.split("|")[1]
    cdr1 = str(seq[26:38]).replace("-","")
    cdr2 = str(seq[55:65]).replace("-","")
    all_cdr1.append(cdr1)
    all_cdr2.append(cdr2)
    #print(name, cdr1, cdr2)
    #print("\"{0}\":\"{1}\"".format(vgene, cdr1), end=',')
    print("\"{0}\":\"{1}\"".format(vgene, cdr2), end=',')


####################################################################################################
##### Mus Musculus
####################################################################################################

#TRAV
gene_name = 'TRAV'
all_gene = []
all_cdr1 = []
all_cdr2 = []
for seq_record in SeqIO.parse("./{0}_MusMusculus.fasta".format(gene_name), "fasta"):
    name = seq_record.id
    vgene = name.split("|")[1]
    seq = seq_record.seq
    print(len(seq))
    cdr1 = str(seq[26:38]).replace("-","")
    cdr2 = str(seq[55:65]).replace("-","")
    all_cdr1.append(cdr1)
    all_cdr2.append(cdr2)
    #print(vgene, cdr1, cdr2)
    print("\"{0}\":\"{1}\"".format(vgene, cdr1), end=',')
    #print("\"{0}\":\"{1}\"".format(vgene, cdr2), end=',')

####################################################################################################
#TRBV
gene_name = 'TRBV'
all_gene = []
all_cdr1 = []
all_cdr2 = []
for seq_record in SeqIO.parse("./{0}_MusMusculus.fasta".format(gene_name), "fasta"):
    name = seq_record.id
    seq = seq_record.seq
    vgene = name.split("|")[1]
    cdr1 = str(seq[26:38]).replace("-","")
    cdr2 = str(seq[55:65]).replace("-","")
    all_cdr1.append(cdr1)
    all_cdr2.append(cdr2)
    #print(name, cdr1, cdr2)
    #print("\"{0}\":\"{1}\"".format(vgene, cdr1), end=',')
    print("\"{0}\":\"{1}\"".format(vgene, cdr2), end=',')

