
input = 'human.1.rna.fna'
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

concat_fasta(input, concat)


with open(concat, 'r') as f_input, open(trimmed, 'w') as f_output:
    for line in f_input:
        if line.startswith('ATG') and len(line) % 3 == 0 and len(line) < 2000:
            seq = ' '.join([line[i:i+3] for i in range(0, len(line)+1,3)])
            f_output.write(seq)

