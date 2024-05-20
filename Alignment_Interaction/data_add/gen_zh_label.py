# Define the input and output file paths
input_file_path = '/home/wzl/ea/selfea_a40/data/dbp15k/zh_en/ent_ids_1'
output_file_path = '/home/wzl/ea/selfea_a40/data/dbp15k/zh_en/label_1'

def generate_label_file(input_file_path, output_file_path):
    output_lines = []

    with open(input_file_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            parts = line.strip().split()
            if len(parts) == 2:
                url = parts[1]
                resource_name = url.split('/')[-1].replace('_', ' ')
                output_lines.append(f'{url}\t{resource_name}')

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        outfile.write('\n'.join(output_lines))

    print('File label_1 generated successfully.')
generate_label_file(input_file_path, output_file_path)
