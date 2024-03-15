def replace_zeros_with_values(file1_path, file2_path, output_file_path):
    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2, open(output_file_path, 'w') as out_file:
        for line1, line2 in zip(f1, f2):
            value1 = float(line1.strip())
            value2 = float(line2.strip())

            if value1 == 0:
                out_file.write(f"{value2}\n")
            else:
                out_file.write(f"{value1}\n")

replace_zeros_with_values("output/pcf_values_llama.txt", "output/examples_pcf_values.txt", "output/pcf_values_llama_combined.txt")
