
input = 'Homo_sapiens.GRCh38.cdna.all.fa'
concat = 'human_DNA_concat.fasta'
trimmed = 'human_DNA_trimmed.fasta'

def concat_fasta(input, output):
    with open(input, 'r') as f_input, open(output, 'w') as f_output:
        block = []

        for line in f_input:
            if line.startswith('>'):
                if block:
                    f_output.write(''.join(block) + '\n')
                    block = []
                f_output.write(line)
            else:
                block.append(line.strip())

        if block:
            f_output.write(''.join(block) + '\n')

nuc = ['A', 'T', 'C', 'G']
with open(concat, 'r') as f_input, open(trimmed, 'w') as f_output:
    for line in f_input:
        if any(line.startswith(x) for x in nuc) and len(line) < 2000 and len(line) > 500:
            seq = ' '.join(list(line))
            f_output.write(seq)
