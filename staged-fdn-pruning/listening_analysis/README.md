# listening analysis

anonymised scores and analysis for the MUSHRA-style listening study in the paper
(34 participants, N in {16, 32, 64, 128}).

- `analyse_listening.py` - screening (BS.1534 hidden-reference rule), repeat
  averaging, group means and sample standard deviations
- `plot_listening.py` - optional paired-difference diagnostic (not a manuscript figure)
- `listening_table.tex` - the supplementary table

the analysis unit is one session; repeats are averaged within participant before
any group statistic, giving every configuration n = 34. per-session anonymised
ratings are available from the authors on request.
