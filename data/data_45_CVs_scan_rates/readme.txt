These substance#.pt files coontain simulation CV data for four different sets of electrochemical parameters:
        0: {"alpha": 0.3, "d": 2, "k0": 0.001},  # Substance A
        1: {"alpha": 0.7, "d": 2, "k0": 0.001},  # Substance B
        2: {"alpha": 0.3, "d": 2, "k0": 10},     # Substance C
        3: {"alpha": 0.7, "d": 2, "k0": 0.01},     # Substance D

 task1.pt contains substance0 + substance1 data
 task2.pt contains 0, 1, 2
 task3.pt contains 0, 1, 2, 3

 The data should be used for joint training baseline only and testing the none baseline!

 task#_none.pt contain datasets for none baseline training, so:
 task1_none.pt has only data for 0, 1
 task2_none.pt has only data for 2
 task3_none.pt has only data for 3