import pandas as pd

# run acquire 100 times, require that pixels are not firing in less than 50% of runs
total_runs = 100
noise_fraction = 0.5
count_threshold = total_runs * noise_fraction

# load data with pandas and assume file format
# COL ROW HIT TOT COUNT
columns = ['col', 'row', 'hit', 'tot', 'count']
# first six rows are comments by peary, individual acquisitions are indexed with === number === and need to be filtered out
df = pd.read_csv('data/test.csv', comment='=', skiprows=6, names=columns)
df = df.groupby(['col', 'row'])['count'].sum().to_frame()

# find noisy pixels
df_noisy = df[df['count'] > count_threshold]

# write to output file
outfile_mask = open('data/mask.csv', 'w')
for index, row in df_noisy.iterrows():
    outfile_mask.write(f"{index[0]},{index[1]}\n")
